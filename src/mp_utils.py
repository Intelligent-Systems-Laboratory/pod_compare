# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import shutil
import os
import tqdm
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

import multiprocessing as mp
import numpy as np

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, MetadataCatalog

from detectron2.data.build import DatasetMapper, get_detection_dataset_dicts
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.samplers import TrainingSampler, InferenceSampler

import core.datasets.metadata as metadata

from probabilistic_inference.probabilistic_inference import build_predictor
from train_utils import ActiveTrainer, compute_cls_entropy, compute_cls_max_conf

import concurrent.futures
import time

from probabilistic_inference.inference_utils import instances_to_json

def split_list(dataset, n):
    a = list(range(len(dataset)))
    k, m = divmod(len(a), n)
    idx_lists = list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    dataset_list = []
    for idx_list in idx_lists:
        temp_list = []
        for idx in idx_list:
            temp_list.append(dataset[idx])

        dataset_list.append(temp_list)

    return dataset_list


def model_predict(cfg, model_full_path, cat_mapping_dict, dataset, gpu_num):

    torch.cuda.set_device(gpu_num)

    print(f"processing {len(dataset)} images  running in {torch.cuda.current_device()}")

    #load model and weights
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_full_path, resume=False)
    predictor = build_predictor(cfg, model)

    

    #setup scoring configs
    det_cls_score = cfg.ACTIVE_LEARNING.DET_CLS_SCORE
    det_cls_merge_mode = cfg.ACTIVE_LEARNING.DET_CLS_MERGE_MODE

    mapper = DatasetMapper(cfg, False)
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    num_workers = 4
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )

    #warmup
    warmup_img = next(iter(data_loader))

    _ = predictor(warmup_img)

    output_list = []
    cls_score_list = []
    box_score_list = []

    with torch.no_grad():
        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for idx, input_im in enumerate(data_loader):
                outputs = predictor(input_im)
                cls_preds = outputs.pred_cls_probs.cpu().numpy()
                predicted_boxes = outputs.pred_boxes.tensor.cpu().numpy()
                predicted_covar_mats = outputs.pred_boxes_covariance.cpu().numpy()

                # combine parallel processes here
                box_score = np.array([mat.diagonal().prod() for mat in predicted_covar_mats]).mean()
                #mean of the max confidence pre detection
                if det_cls_score == "entropy":
                    cls_score = compute_cls_entropy(cls_preds, det_cls_merge_mode) #entropy, mean default
                elif det_cls_score == "max_conf":
                    cls_score = compute_cls_max_conf(cls_preds, det_cls_merge_mode)
                else:
                    raise ValueError('Invalid det_cls_score {}.'.format(det_cls_score))
                box_score_list.append(box_score)
                cls_score_list.append(cls_score)
                output_list.extend(instances_to_json(outputs, input_im[0]['image_id'], cat_mapping_dict))

                pbar.update(1)
                
    return {'output_list': output_list, 'cls_score_list':cls_score_list, 'box_score_list':box_score_list}

def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

def parallel_predict(cfg, model_path, cat_mapping_dict, dataset, process_gpu_list):
    start = time.perf_counter()

    final_output_list = []
    cls_score_list = []
    box_score_list = []
    # prepare args
    cfgs = [cfg for _ in range(len(process_gpu_list))]
    model_paths = [model_path for _ in range(len(process_gpu_list))]
    cat_mapping_dicts = [cat_mapping_dict for _ in range(len(process_gpu_list))]
    #dataset = dataset[:100]
    datasets = split_list(dataset, len(process_gpu_list))


    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        results = executor.map(model_predict, cfgs, model_paths, cat_mapping_dicts, datasets, process_gpu_list)

        for result in results:
            final_output_list.extend(result['output_list'])
            cls_score_list.extend(result['cls_score_list'])
            box_score_list.extend(result['box_score_list'])

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
    print(f'length od cls_score_list {len(cls_score_list)}')
    
    assert len(cls_score_list) == len(dataset)

    return final_output_list, cls_score_list, box_score_list


#this function is not used
def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        raise ValueError("evaluator is required")
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0


    #inference context eval is removed, because the model has to stay in train mode for the MCDrop
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            #print(inputs)
            #print(outputs)
            evaluator.process(inputs, outputs)
            #return
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results