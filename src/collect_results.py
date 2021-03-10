import json
import shutil
import os
import torch
import tqdm
import csv
from shutil import copyfile

# Detectron imports
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader, MetadataCatalog

# Project imports
import core.datasets.metadata as metadata

from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_average_precision, compute_probabilistic_metrics, compute_calibration_errors
from probabilistic_inference.probabilistic_inference import build_predictor
from probabilistic_inference.inference_utils import instances_to_json

from train_utils import ActiveTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##setup inference args, this also contains all the training args
arg_parser = setup_arg_parser()
args = arg_parser.parse_args("")
# Support single gpu inference only.
args.num_gpus = 1
#args.dataset_dir = '/public-dataset/BDD/bdd100k'
#args.test_dataset = 'bdd_val'
args.dataset_dir = '~/datasets/VOC2012'
args.test_dataset = 'cocovoc2012_val'
#args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/VOC-Detection/retinanet/ex1_rnd_10v2.yaml'
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


output_dir = cfg.ACTIVE_LEARNING.OUT_DIR

output_results_dir = os.path.join(output_dir,"eval_results")

os.makedirs(output_results_dir,exist_ok=True)

results_path_list = os.listdir(output_results_dir)

row_list = []

for results_path in results_path_list:

    if 'results.txt' not in results_path:
        print(f"{results_path} Not a Model")
        continue

    full_path = os.path.join(output_results_dir,results_path)

    with open(full_path, 'r') as results_file:
        data = results_file.readlines()
    
    step = full_path.strip('_results.txt').split('_step')[-1]
    common = data[0].strip().strip('[').strip(']').split(', ')
    e1 = common[0]
    e2 = common[8]
    e3 = common[-1]

    e4 = data[2].strip().split(': ')[1]
    e5 = data[3].strip().split(': ')[1]
    e6 = data[4].strip().split(': ')[1]
    e7 = data[5].strip().split(': ')[1]
    e8 = data[6].strip().split(': ')[1]
    e9 = data[7].strip().split(': ')[1]
    e10 = data[8].strip().split(': ')[1]

    e11 = data[11].strip().split(': ')[1]
    e12 = data[12].strip().split(': ')[1]
    e13 = data[13].strip().split(': ')[1]
    e14 = data[14].strip().split(': ')[1]
    e15 = data[15].strip().split(': ')[1]

    row_list.append([step,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15])

    #when running the eval at the same time, make sure that different cfg file is used per process that is run
out_file = os.path.join(output_results_dir,'results_summary.csv')

with open(out_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(row_list)

    #shutil.move(os.path.join(inference_output_dir, 'coco_instances_results.json'), os.path.join(output_results_dir, f"{results_path.split('.')[0]}_coco_results.json"))

    # mv coco_results to coco_result_step_1


