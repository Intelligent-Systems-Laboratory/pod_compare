import json
import os
import torch
import tqdm
from shutil import copyfile


# Detectron imports|
from detectron2.engine import launch
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import *
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers.distributed_sampler import InferenceSampler

# Project imports
import core.datasets.metadata as metadata

from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_probabilistic_metrics, compute_calibration_errors
from probabilistic_inference.probabilistic_inference import build_predictor
from probabilistic_inference.inference_utils import instances_to_json

import torch.multiprocessing as mp

from baal.active import FileDataset, ActiveLearningDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##todo
# create mp_predict(function)



def load_dataset_and_mapper_from_config(cfg, dataset_name='bdd_val', mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    # use active learning dataset wrapper
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
        
    return dataset, mapper

def data_process(name_list, dataset):
    for data in dataset:
        name_list.append(data['file_name'])

if __name__ == "__main__":
    ##setup inference args, this also contains all the training args
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args("")
    # Support single gpu inference only.
    args.num_gpus = 1
    args.dataset_dir = '/public-dataset/BDD/bdd100k'
    args.test_dataset = 'bdd_val'
    args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
    args.inference_config = '/home/richard.tanai/cvpr2/pod_compare/src/configs/Inference/bayes_od_mc_dropout.yaml'
    print("Command Line Args:", args)

    cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    # Make sure only 1 data point is processed at a time. This simulates
    # deployment.
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.DEVICE = device.type

    # Set up number of cpu threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    dataset, mapper = load_dataset_and_mapper_from_config(cfg, dataset_name='bdd_val', mapper=None)

    data1 = dataset[:20]
    data2 = dataset[20:50]

    num_processes = 2

    list1 = []
    list2 = []

    group_list = [list1, list2]
    data_list = [data1, data2]

    processes = []

    #mp.set_start_method('spawn')

    for rank in range(num_processes):
        p = mp.Process(target=data_process, args=(group_list[rank],data_list[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    print(f"list1 is {len(list1)}")
    print(f"list2 is {len(list2)}")