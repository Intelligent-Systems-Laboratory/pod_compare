import json
import shutil
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
args.dataset_dir = '~/datasets/VOC2012'
args.test_dataset = 'voc_2012_test'
args.config_file = '/home/richard.tanai/cvpr2/pod_compare/src/configs/VOC-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml'
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
# Build predictor
model = build_model(cfg)

test_data_loader = build_detection_test_loader(
        cfg, dataset_name=args.test_dataset)

# the models are stored in this directory
output_dir = "outputs_10k_cls1_only"

output_results_dir = os.path.join(output_dir,"eval_results")

os.makedirs(output_results_dir,exist_ok=True)

model_path_list = os.listdir(output_dir)
for model_path in model_path_list:
    full_path = os.path.join(output_dir,model_path)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            full_path, resume=False)
    
    predictor = build_predictor(cfg, model)

    #results_dir = inference_output_dir + "/" + model_path

    final_output_list = []
    if not args.eval_only:
        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data_loader)) as pbar:
                for idx, input_im in enumerate(test_data_loader):
                    #print(input_im.size)
                    outputs = predictor(input_im)

                    final_output_list.extend(
                        instances_to_json(
                            outputs,
                            input_im[0]['image_id'],
                            cat_mapping_dict))
                    pbar.update(1)

                        


        with open(os.path.join(inference_output_dir, 'coco_instances_results.json'), 'w') as fp:
            json.dump(final_output_list, fp, indent=4,
                      separators=(',', ': '))

    out_file = os.path.join(output_results_dir,model_path.split('.')[0]+'_results.txt')
    compute_average_precision.main(args, cfg, to_file=out_file)
    compute_probabilistic_metrics.main(args, cfg, to_file=out_file)
    compute_calibration_errors.main(args, cfg, to_file=out_file)

    shutil.move(os.path.join(inference_output_dir, 'coco_instances_results.json'), os.path.join(output_results_dir, f"{model_path.split('.')[0]}_coco_results.json"))

    # mv coco_results to coco_result_step_1

# epoch is 20, just an arbitrary number loltrain_step = 1
