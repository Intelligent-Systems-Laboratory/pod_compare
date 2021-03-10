import json
import numpy as np
import os
import torch
import time
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

from train_utils import ActiveTrainer, compute_cls_entropy, compute_cls_max_conf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##setup inference args, this also contains all the training args
arg_parser = setup_arg_parser()
args = arg_parser.parse_args("")
# Support single gpu inference only.
args.num_gpus = 1
args.dataset_dir = '~/datasets/VOC2012'
args.test_dataset = 'cocovoc_2012'
args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/VOC-Detection/retinanet/ex1_ent_d20.yaml'
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


# Build predictor
model = build_model(cfg)

DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)

#DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#            "/home/richard.tanai/cvpr2/pod_compare/src/outputs2/checkpoint_step6.pth", resume=False)
            

trainer = ActiveTrainer(cfg, model)
#trainer.resume_or_load(resume=True)

# 10000 is already started in the init using cfg
# epoch is 20, just an arbitrary number lol

## active learning loop configurables

train_step = 1


max_step = cfg.ACTIVE_LEARNING.MAX_STEP
label_per_step = cfg.ACTIVE_LEARNING.STEP_N
out_dir = cfg.ACTIVE_LEARNING.OUT_DIR
det_cls_score = cfg.ACTIVE_LEARNING.DET_CLS_SCORE
det_cls_merge_mode = cfg.ACTIVE_LEARNING.DET_CLS_MERGE_MODE
w_cls_score = cfg.ACTIVE_LEARNING.W_CLS_SCORE
max_dets = cfg.ACTIVE_LEARNING.MAX_DETS

os.makedirs(out_dir, exist_ok=True)

start = time.perf_counter()

while(1):
    print(f"performing train step {train_step}")
    trainer.train()
    torch.save(model.state_dict(), f"{out_dir}/checkpoint_step{train_step}.pth")

    if len(trainer.dataset.pool) <= 0 or train_step >= max_step:
        print("training completed")
        break

    pool_loader = trainer.build_pool_dataloader()

    final_output_list = []
    cls_score_list = []
    box_score_list = []

    predictor = build_predictor(cfg, model)

    if det_cls_score == 'random':
        if len(trainer.dataset.pool) >= label_per_step:
            trainer.dataset.label_randomly(label_per_step)
        elif len(trainer.dataset.pool) > 0:
            trainer.dataset.label_randomly(len(trainer.dataset.pool))
        else:
            break

    else:
        if not args.eval_only:
            with torch.no_grad():
                with tqdm.tqdm(total=len(pool_loader)) as pbar:
                    for idx, input_im in enumerate(pool_loader):
                        #print(input_im.size)
                        outputs = predictor(input_im)

                        results = outputs

                        cls_preds = results.pred_cls_probs.cpu().numpy()
                        predicted_boxes = results.pred_boxes.tensor.cpu().numpy()
                        predicted_covar_mats = results.pred_boxes_covariance.cpu().numpy()


                        if len(cls_preds) > max_dets:
                            cls_preds = cls_preds[:max_dets]
                            predicted_boxes = predicted_boxes[:max_dets]
                            predicted_covar_mats = predicted_covar_mats[:max_dets]
                        
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

                        pbar.update(1)

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
print(f'Active Learning Loop finished in {round(finish-start, 2)} second(s)')