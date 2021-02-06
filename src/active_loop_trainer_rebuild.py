import json
import os
import torch
import tqdm
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
args.dataset_dir = '/public-dataset/BDD/bdd100k'
args.test_dataset = 'bdd_val'
args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
args.inference_config = '/home/richard.tanai/cvpr2/pod_compare/src/configs/Inference/bayes_od_mc_dropout.yaml'
args.random_seed = 1001
args.resume=False
print("Command Line Args:", args)

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

DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)

#DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#            "/home/richard.tanai/cvpr2/pod_compare/src/outputs2/checkpoint_step6.pth", resume=False)
            

trainer = ActiveTrainer(cfg, model)
#trainer.resume_or_load(resume=True)

#predictor = build_predictor(cfg, model)

# 10000 is already started in the init
# epoch is 20, just an arbitrary number lol
train_step = 1
label_per_step = 10000
while(1):
    print(f"performing train step {train_step}")
    trainer.train()
    torch.save(model.state_dict(), f"outputs3/checkpoint_step{train_step}.pth")
    if len(trainer.dataset.pool) >= label_per_step:
        trainer.dataset.label_randomly(label_per_step)
    elif len(trainer.dataset.pool) > 0:
        trainer.dataset.label_randomly(len(trainer.dataset.pool))
    else:
        break
    trainer.rebuild_trainer()
    train_step += 1

len(trainer.dataset.pool)

#trainer.dataset.label_randomly(len(trainer.dataset.pool))
#trainer.rebuild_trainer()
#trainer.train()
#torch.save(model.state_dict(), f"outputs2/checkpoint_step_last.pth")