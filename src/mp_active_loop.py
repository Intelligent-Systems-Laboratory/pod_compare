import json
import numpy as np
import os
import torch
import tqdm
import time
from shutil import copyfile

# Detectron imports
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, MetadataCatalog

from detectron2.utils import comm
# Project imports
import core.datasets.metadata as metadata

from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_probabilistic_metrics, compute_calibration_errors
from probabilistic_inference.probabilistic_inference import build_predictor
from probabilistic_inference.inference_utils import instances_to_json

from train_utils import ActiveTrainer, compute_cls_entropy, compute_cls_max_conf
from mp_utils import parallel_predict

from torch.nn.parallel import DistributedDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(args): 
    # run this once per session only
    cfg = setup_config(args, random_seed=args.random_seed, is_testing=False)

    #cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    #cfg.SOLVER.IMS_PER_BATCH = 1

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
    model = build_model(cfg)

    

    if comm.get_world_size() > 1:
        print(f"Using GPUs {comm.get_local_rank()}")
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
    #
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    #model = DistributedDataParallel(model, device_ids=[1,2,3,4], broadcast_buffers=False)

    trainer = ActiveTrainer(cfg, model)

    train_step = 1
    max_step = cfg.ACTIVE_LEARNING.MAX_STEP
    label_per_step = cfg.ACTIVE_LEARNING.STEP_N
    out_dir = cfg.ACTIVE_LEARNING.OUT_DIR
    det_cls_score = cfg.ACTIVE_LEARNING.DET_CLS_SCORE
    det_cls_merge_mode = cfg.ACTIVE_LEARNING.DET_CLS_MERGE_MODE
    w_cls_score = cfg.ACTIVE_LEARNING.W_CLS_SCORE

    os.makedirs(out_dir, exist_ok=True)

    process_gpu_list = [0, 1, 2, 3]

    start = time.perf_counter()

    while(1):
        print(f"performing train step {train_step}")
        trainer.train()
        model_full_path = f"{out_dir}/checkpoint_step{train_step}.pth"
        torch.save(model.state_dict(), model_full_path)

        if len(trainer.dataset.pool) <= 0 or train_step > max_step:
            print("training completed")
            break
        if comm.is_main_process():
            if det_cls_score == 'random':
                if len(trainer.dataset.pool) >= label_per_step:
                    trainer.dataset.label_randomly(label_per_step)
                elif len(trainer.dataset.pool) > 0:
                    trainer.dataset.label_randomly(len(trainer.dataset.pool))
                else:
                    break

            else:
                pool_dataset = trainer.dataset.pool

                final_output_list, cls_score_list, box_score_list = parallel_predict(cfg, model_full_path, cat_mapping_dict, pool_dataset, process_gpu_list)

                cls_score_rank = np.array(cls_score_list).argsort().argsort()
                box_score_rank = (-np.array(box_score_list)).argsort().argsort()

                #possible weighted fusion can be added here
                total_sort = np.argsort((w_cls_score)*cls_score_rank + (1-w_cls_score)*box_score_rank)

                if len(trainer.dataset.pool) >= label_per_step:
                    idx_to_label = total_sort[:label_per_step].tolist()
                    trainer.dataset.label(idx_to_label)
                elif len(trainer.dataset.pool) > 0:
                    trainer.dataset.label_randomly(len(trainer.dataset.pool))
                else:
                    break

        trainer.rebuild_trainer()
        train_step += 1

    finish = time.perf_counter()
    print(f'Finished Active Learning Loop in {round(finish-start, 2)} second(s)')

##setup inference args, this also contains all the training args
if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args("")
    # Support single gpu inference only.
    args.num_gpus = 4
    #args.dataset_dir = '/public-dataset/BDD/bdd100k'
    #args.test_dataset = 'bdd_val'
    args.dataset_dir = '~/datasets/VOC2012'
    args.test_dataset = 'cocovoc2012_val'
    #args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/BDD-Detection/retinanet/active_retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
    args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/VOC-Detection/retinanet/ent_10.yaml'
    args.inference_config = '/home/richard.tanai/cvpr2/pod_compare/src/configs/Inference/bayes_od_mc_dropout.yaml'
    args.random_seed = 2000
    args.resume=False
    print("Command Line Args:", args)

    #main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )