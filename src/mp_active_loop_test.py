import json
import shutil
import os
import torch
import tqdm
from shutil import copyfile
import multiprocessing as mp
import numpy as np
# Detectron imports
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, MetadataCatalog


from detectron2.data.build import DatasetMapper, get_detection_dataset_dicts
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.samplers import TrainingSampler, InferenceSampler
# Project imports
import core.datasets.metadata as metadata

from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_probabilistic_metrics, compute_calibration_errors
from probabilistic_inference.probabilistic_inference import build_predictor
from probabilistic_inference.inference_utils import instances_to_json

from train_utils import ActiveTrainer, compute_cls_entropy, compute_cls_max_conf

import concurrent.futures
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

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
    num_workers = 16
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )

        
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
    datasets = list(split_list(dataset, len(process_gpu_list)))

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
    
    return final_output_list, cls_score_list, box_score_list

if __name__ == "__main__": 
    ##setup inference args, this also contains all the training args
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args("")
    # Support single gpu inference only.
    args.num_gpus = 1
    args.dataset_dir = '/public-dataset/BDD/bdd100k'
    args.test_dataset = 'bdd_val'
    #args.dataset_dir = '~/datasets/VOC2012'
    #args.test_dataset = 'cocovoc2012_val'
    args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/BDD-Detection/retinanet/active_retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
    #args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/VOC-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
    args.inference_config = '/home/richard.tanai/cvpr2/pod_compare/src/configs/Inference/bayes_od_mc_dropout.yaml'
    args.random_seed = 1000
    args.resume=False
    print("Command Line Args:", args)

    # run this once per session only
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=False)

    #cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings


    inference_output_dir = os.path.join(
            cfg['OUTPUT_DIR'],
            'inference',
            args.test_dataset,
            os.path.split(args.inference_config)[-1][:-5])
    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Get category mapping dictionary:
    #train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    if (train_thing_dataset_id_to_contiguous_id == test_thing_dataset_id_to_contiguous_id) or (
            cfg.DATASETS.TRAIN[0] == 'coco_not_in_voc_2017_train'):
        cat_mapping_dict = dict(
            (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
    else:
        # If not equal, two situations: 1) BDD to KITTI and 2) COCO to PASCAL
        cat_mapping_dict = dict(
            (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
        if 'voc' in args.test_dataset and 'coco' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.COCO_TO_VOC_CONTIGUOUS_ID.items())
        elif 'kitti' in args.test_dataset and 'bdd' in cfg.DATASETS.TRAIN[0]:
            dataset_mapping_dict = dict(
                (v, k) for k, v in metadata.BDD_TO_KITTI_CONTIGUOUS_ID.items())
        else:
            ValueError(
                'Cannot generate category mapping dictionary. Please check if training and inference datasets are compatible.')
        cat_mapping_dict = dict(
            (dataset_mapping_dict[k], v) for k, v in cat_mapping_dict.items())

    # Build predictor
    #model = build_model(cfg)

    #test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)

    output_dir = cfg.ACTIVE_LEARNING.OUT_DIR

    output_results_dir = os.path.join(output_dir,"eval_results")

    os.makedirs(output_results_dir,exist_ok=True)

    model_path_list = os.listdir(output_dir)

    #process_gpu_list = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    process_gpu_list = [0, 0, 1, 1, 2, 2, 3, 3]

    #process_gpu_list = [2, 2, 2, 2, 3, 3, 3, 3]

    #process_gpu_list = [0, 1, 2, 3]

    for model_path in model_path_list:

        if 'checkpoint_step7.pth' not in model_path:
            print(f"{model_path} Not a Model")
            continue

        full_path = os.path.join(output_dir,model_path)
        #DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(full_path, resume=False)


        # load dataset
        dataset = get_detection_dataset_dicts([args.test_dataset],filter_empty=False, proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
        )

        final_output_list, cls_score_list, box_output_list = parallel_predict(cfg, full_path, cat_mapping_dict, dataset, process_gpu_list)

        with open(os.path.join(inference_output_dir, 'coco_instances_results.json'), 'w') as fp:
            json.dump(final_output_list, fp, indent=4,
                      separators=(',', ': '))

    #when running the eval at the same time, make sure that different cfg file is used per process that is run
        out_file = os.path.join(output_results_dir,model_path.split('.')[0]+'_results.txt')
        compute_average_precision.main(args, cfg, to_file=out_file)
        compute_probabilistic_metrics.main(args, cfg, to_file=out_file)
        compute_calibration_errors.main(args, cfg, to_file=out_file)

        shutil.move(os.path.join(inference_output_dir, 'coco_instances_results.json'), os.path.join(output_results_dir, f"{model_path.split('.')[0]}_coco_results.json"))

    # mv coco_results to coco_result_step_1

# epoch is 20, just an arbitrary number loltrain_step = 1
