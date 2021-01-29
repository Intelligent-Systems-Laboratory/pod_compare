import logging
import math
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn, distributions

# Detectron Imports
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, RetinaNetHead, permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes


# From Probabilistic Inference
# Detectron Imports

import cv2
import numpy as np
import os

from abc import ABC, abstractmethod
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from core.visualization_tools.probabilistic_visualizer import ProbabilisticVisualizer

# Project Imports
from probabilistic_inference import inference_utils
from probabilistic_modeling.modeling_utils import covariance_output_to_cholesky


@META_ARCH_REGISTRY.register()
class ProbabilisticRetinaNet(RetinaNet):
    """
    Probabilistic retinanet class.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
                # Probabilistic Inference Init
        self.inference_mode = self.cfg.PROBABILISTIC_INFERENCE.INFERENCE_MODE
        self.mc_dropout_enabled = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE
        self.num_mc_dropout_runs = self.cfg.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS
        self.dropout_rate = self.cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0
        
        self.sample_box2box_transform = inference_utils.SampleBox2BoxTransform(
            self.cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != 'none'
        self.cls_var_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        self.bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        self.bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        if self.bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10


        self.current_step = 0
        self.annealing_step = cfg.SOLVER.STEPS[1]

        # Define custom probabilistic head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.head_in_features]
        self.head = ProbabilisticRetinaNetHead(
            cfg,
            self.use_dropout,
            self.dropout_rate,
            self.compute_cls_var,
            self.compute_bbox_cov,
            self.bbox_cov_dims,
            feature_shapes)

        # Send to device
        self.to(self.device)

    def forward(
            self,
            batched_inputs,
            return_anchorwise_output=False,
            num_mc_dropout_runs=-1):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_anchorwise_output (bool): returns raw output for probabilistic inference

            num_mc_dropout_runs (int): perform efficient monte-carlo dropout runs by running only the head and
            not full neural network.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # Preprocess image
        images = self.preprocess_image(batched_inputs)

        # Extract features and generate anchors
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        anchors = self.anchor_generator(features)

        # MC_Dropout inference forward
        if num_mc_dropout_runs > 1:
            anchors = anchors*num_mc_dropout_runs
            features = features*num_mc_dropout_runs
            output_dict = self.produce_raw_output(anchors, features)
            return output_dict

        # Regular inference forward
        if return_anchorwise_output:
            return self.produce_raw_output(anchors, features)

        # Training and validation forward
        pred_logits, pred_anchor_deltas, pred_logits_vars, pred_anchor_deltas_vars = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_vars]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [permute_to_N_HWA_K(x, self.bbox_cov_dims) for x in pred_anchor_deltas_vars]

        if self.training:  #change to if mc_dropout_enabled
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_classes, gt_boxes = self.label_anchors(
                anchors, gt_instances)

            self.anchors = torch.cat(
                [Boxes.cat(anchors).tensor for i in range(len(gt_instances))], 0)

            # Loss is computed based on what values are to be estimated by the neural
            # network
            losses = self.losses(
                anchors,
                gt_classes,
                gt_boxes,
                pred_logits,
                pred_anchor_deltas,
                pred_logits_vars,
                pred_anchor_deltas_vars)

            self.current_step += 1

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)
            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
            self,
            anchors,
            gt_classes,
            gt_boxes,
            pred_class_logits,
            pred_anchor_deltas,
            pred_class_logits_var=None,
            pred_bbox_cov=None):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas`, `pred_class_logits_var` and `pred_bbox_cov`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_classes)
        gt_labels = torch.stack(gt_classes)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.

        # Transform per-feature layer lists to a single tensor
        pred_class_logits = cat(pred_class_logits, dim=1)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)

        if pred_class_logits_var is not None:
            pred_class_logits_var = cat(
                pred_class_logits_var, dim=1)

        if pred_bbox_cov is not None:
            pred_bbox_cov = cat(
                pred_bbox_cov, dim=1)

        gt_classes_target = torch.nn.functional.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
                           :, :-1
                           ].to(pred_class_logits[0].dtype)  # no loss for the last (background) class

        # Classification losses
        if self.compute_cls_var:
            # Compute classification variance according to:
            # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
            if self.cls_var_loss == 'loss_attenuation':
                num_samples = self.cls_var_num_samples
                # Compute standard deviation
                pred_class_logits_var = torch.sqrt(torch.exp(
                    pred_class_logits_var[valid_mask]))

                pred_class_logits = pred_class_logits[valid_mask]

                # Produce normal samples using logits as the mean and the standard deviation computed above
                # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
                # COCO dataset.
                univariate_normal_dists = distributions.normal.Normal(
                    pred_class_logits, scale=pred_class_logits_var)

                pred_class_stochastic_logits = univariate_normal_dists.rsample(
                    (num_samples,))
                pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                    (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
                pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(
                    2)

                # Produce copies of the target classes to match the number of
                # stochastic samples.
                gt_classes_target = torch.unsqueeze(gt_classes_target, 0)
                gt_classes_target = torch.repeat_interleave(
                    gt_classes_target, num_samples, dim=0).view(
                    (gt_classes_target.shape[1] * num_samples, gt_classes_target.shape[2], -1))
                gt_classes_target = gt_classes_target.squeeze(2)

                # Produce copies of the target classes to form the stochastic
                # focal loss.
                loss_cls = sigmoid_focal_loss_jit(
                    pred_class_stochastic_logits,
                    gt_classes_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / (num_samples * max(1, self.loss_normalizer))
            else:
                raise ValueError(
                    'Invalid classification loss name {}.'.format(
                        self.bbox_cov_loss))
        else:
            # Standard loss computation in case one wants to use this code
            # without any probabilistic inference.
            loss_cls = sigmoid_focal_loss_jit(
                pred_class_logits[valid_mask],
                gt_classes_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        # Compute Regression Loss
        pred_anchor_deltas = pred_anchor_deltas[pos_mask]
        gt_anchors_deltas = gt_anchor_deltas[pos_mask]
        if self.compute_bbox_cov:
            if self.bbox_cov_loss == 'negative_log_likelihood':
                if self.bbox_cov_type == 'diagonal':
                    # Compute regression variance according to:
                    # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017

                    # This is the log of the variance. We have to clamp it else negative
                    # log likelihood goes to infinity.
                    pred_bbox_cov = torch.clamp(
                        pred_bbox_cov[pos_mask], -7.0, 7.0)

                    loss_box_reg = 0.5 * torch.exp(-pred_bbox_cov) * smooth_l1_loss(
                        pred_anchor_deltas,
                        gt_anchors_deltas,
                        beta=self.smooth_l1_beta)

                    loss_covariance_regularize = 0.5 * pred_bbox_cov
                    loss_box_reg += loss_covariance_regularize

                    loss_box_reg = torch.sum(
                        loss_box_reg) / max(1, self.loss_normalizer)
            else:
                raise ValueError(
                    'Invalid regression loss name {}.'.format(
                        self.bbox_cov_loss))

            # Perform loss annealing.
            standard_regression_loss = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)
            probabilistic_loss_weight = min(1.0, self.current_step/self.annealing_step)
            probabilistic_loss_weight = (100**probabilistic_loss_weight-1.0)/(100.0-1.0)
            loss_box_reg = (1.0 - probabilistic_loss_weight)*standard_regression_loss + probabilistic_loss_weight*loss_box_reg
        else:
            # Standard regression loss in case no variance is needed to be
            # estimated
            loss_box_reg = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def produce_raw_output(self, anchors, features):
        """
        Given anchors and features, produces raw pre-nms output to be used for custom fusion operations.
        """
        # Perform inference run
        pred_logits, pred_anchor_deltas, pred_logits_vars, pred_anchor_deltas_vars = self.head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits_vars]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [permute_to_N_HWA_K(x, self.bbox_cov_dims) for x in pred_anchor_deltas_vars]

        # Create raw output dictionary
        raw_output = {'anchors': anchors}

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.
        raw_output.update({'box_cls': pred_logits,
                           'box_delta': pred_anchor_deltas,
                           'box_cls_var': pred_logits_vars,
                           'box_reg_var': pred_anchor_deltas_vars})
        return raw_output
    
    ## new method for the inference call
    def predict_forward(self, input_im):
        if self.mc_dropout_enabled:
            self.train()
        else:
            self.eval()

        if self.inference_mode == 'standard_nms':
            results = self.post_processing_standard_nms(input_im)
        elif self.inference_mode == 'mc_dropout_ensembles':
            results = self.post_processing_mc_dropout_ensembles(
                input_im)
        elif self.inference_mode == 'anchor_statistics':
            results = self.post_processing_anchor_statistics(
                input_im)
        #elif self.inference_mode == 'ensembles':
        #    results = self.post_processing_ensembles(input_im, self.model_list)
        elif self.inference_mode == 'bayes_od':
            results = self.post_processing_bayes_od(input_im)
        else:
            raise ValueError(
                'Invalid inference mode {}.'.format(
                    self.inference_mode))

        # Perform post processing on detector output.
        height = input_im[0].get("height", results.image_size[0])
        width = input_im[0].get("width", results.image_size[1])
        results = inference_utils.probabilistic_detector_postprocess(results,
                                                                     height,
                                                                     width)
        return results

    def retinanet_probabilistic_inference(
            self,
            input_im,
            outputs=None,
            ensemble_inference=False,
            outputs_list=None):
        """
        General RetinaNet probabilistic anchor-wise inference. Preliminary inference step for many post-processing
        based inference methods such as standard_nms, anchor_statistics, and bayes_od.
        Args:
            input_im (list): an input im list generated from dataset handler.
            outputs (list): outputs from model.forward. Will be computed internally if not provided.
            ensemble_inference (bool): True if ensembles are used for inference. If set to true, outputs_list must be externally provided.
            outputs_list (list): List of model() outputs, usually generated from ensembles of models.
        Returns:
            all_predicted_boxes,
            all_predicted_boxes_covariance (Tensor): Nx4x4 vectors used
            all_predicted_prob (Tensor): Nx1 scores which represent max of all_pred_prob_vectors. For usage in NMS and mAP computation.
            all_classes_idxs (Tensor): Nx1 Class ids to be used for NMS.
            all_predicted_prob_vectors (Tensor): NxK tensor where K is the number of classes.
        """
        is_epistemic = ((self.mc_dropout_enabled and self.num_mc_dropout_runs > 1)
                        or ensemble_inference) and outputs is None
        if is_epistemic:
            if self.mc_dropout_enabled and self.num_mc_dropout_runs > 1:
                outputs_list = self.forward(
                    input_im,
                    return_anchorwise_output=True,
                    num_mc_dropout_runs=self.num_mc_dropout_runs)
                n_fms = len(self.head_in_features)
                outputs_list = [{key: value[i * n_fms:(i + 1) * n_fms] if value is not None else value for key, value in outputs_list.items()} for i in range(self.num_mc_dropout_runs)]

            outputs = {'anchors': outputs_list[0]['anchors']}

            # Compute box classification and classification variance means
            box_cls = [output['box_cls'] for output in outputs_list]

            box_cls_mean = box_cls[0]
            for i in range(len(box_cls) - 1):
                box_cls_mean = [box_cls_mean[j] + box_cls[i][j]
                                for j in range(len(box_cls_mean))]
            box_cls_mean = [
                box_cls_f_map /
                len(box_cls) for box_cls_f_map in box_cls_mean]
            outputs.update({'box_cls': box_cls_mean})

            if outputs_list[0]['box_cls_var'] is not None:
                box_cls_var = [output['box_cls_var']
                               for output in outputs_list]
                box_cls_var_mean = box_cls_var[0]
                for i in range(len(box_cls_var) - 1):
                    box_cls_var_mean = [
                        box_cls_var_mean[j] +
                        box_cls_var[i][j] for j in range(
                            len(box_cls_var_mean))]
                box_cls_var_mean = [
                    box_cls_var_f_map /
                    len(box_cls_var) for box_cls_var_f_map in box_cls_var_mean]
            else:
                box_cls_var_mean = None
            outputs.update({'box_cls_var': box_cls_var_mean})

            # Compute box regression epistemic variance and mean, and aleatoric
            # variance mean
            box_delta_list = [output['box_delta']
                              for output in outputs_list]
            box_delta_mean = box_delta_list[0]
            for i in range(len(box_delta_list) - 1):
                box_delta_mean = [
                    box_delta_mean[j] +
                    box_delta_list[i][j] for j in range(
                        len(box_delta_mean))]
            box_delta_mean = [
                box_delta_f_map /
                len(box_delta_list) for box_delta_f_map in box_delta_mean]
            outputs.update({'box_delta': box_delta_mean})

            if outputs_list[0]['box_reg_var'] is not None:
                box_reg_var = [output['box_reg_var']
                               for output in outputs_list]
                box_reg_var_mean = box_reg_var[0]
                for i in range(len(box_reg_var) - 1):
                    box_reg_var_mean = [
                        box_reg_var_mean[j] +
                        box_reg_var[i][j] for j in range(
                            len(box_reg_var_mean))]
                box_reg_var_mean = [
                    box_delta_f_map /
                    len(box_reg_var) for box_delta_f_map in box_reg_var_mean]
            else:
                box_reg_var_mean = None
            outputs.update({'box_reg_var': box_reg_var_mean})

        elif outputs is None:
            outputs = self.forward(input_im, return_anchorwise_output=True)

        all_anchors = []
        all_predicted_deltas = []
        all_predicted_boxes_cholesky = []
        all_predicted_prob = []
        all_classes_idxs = []
        all_predicted_prob_vectors = []
        all_predicted_boxes_epistemic_covar = []

        for i, anchors in enumerate(outputs['anchors']):
            box_cls = outputs['box_cls'][i][0]
            box_delta = outputs['box_delta'][i][0]

            # If classification aleatoric uncertainty available, perform
            # monte-carlo sampling to generate logits.
            if outputs['box_cls_var'] is not None:
                box_cls_var = outputs['box_cls_var'][i][0]
                box_cls_dists = torch.distributions.normal.Normal(
                    box_cls, scale=torch.sqrt(torch.exp(box_cls_var)))
                box_cls = box_cls_dists.rsample(
                    (self.cls_var_num_samples,))
                box_cls = torch.mean(box_cls.sigmoid_(), 0)
            else:
                box_cls = box_cls.sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.test_topk_candidates, box_delta.size(0))
            predicted_prob, classes_idxs = torch.max(box_cls, 1)
            predicted_prob, topk_idxs = predicted_prob.topk(num_topk)
            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]
            anchor_idxs = topk_idxs
            classes_idxs = classes_idxs[topk_idxs]

            box_delta = box_delta[anchor_idxs]
            anchors = anchors[anchor_idxs]

            cholesky_decomp = None

            if outputs['box_reg_var'] is not None:
                box_reg_var = outputs['box_reg_var'][i][0][anchor_idxs]
                # Construct cholesky decomposition using diagonal vars
                cholesky_decomp = covariance_output_to_cholesky(box_reg_var)

            # In case dropout is enabled, we need to compute aleatoric
            # covariance matrix and add it here:
            box_reg_epistemic_covar = None
            if is_epistemic:
                # Compute epistemic box covariance matrix
                box_delta_list_i = [
                    self.box2box_transform.apply_deltas(
                        box_delta_i[i][0][anchor_idxs],
                        anchors.tensor) for box_delta_i in box_delta_list]

                _, box_reg_epistemic_covar = inference_utils.compute_mean_covariance_torch(
                    box_delta_list_i)

            all_predicted_deltas.append(box_delta)
            all_predicted_boxes_cholesky.append(cholesky_decomp)
            all_anchors.append(anchors.tensor)
            all_predicted_prob.append(predicted_prob)
            all_predicted_prob_vectors.append(box_cls[anchor_idxs])
            all_classes_idxs.append(classes_idxs)
            all_predicted_boxes_epistemic_covar.append(box_reg_epistemic_covar)

        box_delta = cat(all_predicted_deltas)
        anchors = cat(all_anchors)

        if isinstance(all_predicted_boxes_cholesky[0], torch.Tensor):
            # Generate multivariate samples to be used for monte-carlo simulation. We can afford much more samples
            # here since the matrix dimensions are much smaller and therefore
            # have much less memory footprint. Keep 100 or less to maintain
            # reasonable runtime speed.
            cholesky_decomp = cat(all_predicted_boxes_cholesky)

            multivariate_normal_samples = torch.distributions.MultivariateNormal(
                box_delta, scale_tril=cholesky_decomp)

            # Define monte-carlo samples
            distributions_samples = multivariate_normal_samples.rsample(
                (1000,))
            distributions_samples = torch.transpose(
                torch.transpose(distributions_samples, 0, 1), 1, 2)
            samples_anchors = torch.repeat_interleave(
                anchors.unsqueeze(2), 1000, dim=2)

            # Transform samples from deltas to boxes
            t_dist_samples = self.sample_box2box_transform.apply_samples_deltas(
                distributions_samples, samples_anchors)

            # Compute samples mean and covariance matrices.
            all_predicted_boxes, all_predicted_boxes_covariance = inference_utils.compute_mean_covariance_torch(
                t_dist_samples)
            if isinstance(
                    all_predicted_boxes_epistemic_covar[0],
                    torch.Tensor):
                epistemic_covar_mats = cat(
                    all_predicted_boxes_epistemic_covar)
                all_predicted_boxes_covariance += epistemic_covar_mats
        else:
            # This handles the case where no aleatoric uncertainty is available
            if is_epistemic:
                all_predicted_boxes_covariance = cat(
                    all_predicted_boxes_epistemic_covar)
            else:
                all_predicted_boxes_covariance = []

            # predict boxes
            all_predicted_boxes = self.box2box_transform.apply_deltas(
                box_delta, anchors)

        return all_predicted_boxes, all_predicted_boxes_covariance, cat(
            all_predicted_prob), cat(all_classes_idxs), cat(all_predicted_prob_vectors)

    def post_processing_standard_nms(self, input_im):
        """
        This function produces results using standard non-maximum suppression. The function takes into
        account any probabilistic modeling method when computing the results. It can combine aleatoric uncertainty
        from heteroscedastic regression and epistemic uncertainty from monte-carlo dropout for both classification and
        regression results.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        outputs = self.retinanet_probabilistic_inference(input_im)

        return inference_utils.general_standard_nms_postprocessing(
            input_im, outputs, self.test_nms_thresh, self.max_detections_per_image)

    def post_processing_anchor_statistics(self, input_im):
        """
        This function produces box covariance matrices using anchor statistics. Uses the fact that multiple anchors are
        regressed to the same spatial location for clustering and extraction of box covariance matrix.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        outputs = self.retinanet_probabilistic_inference(input_im)

        return inference_utils.general_anchor_statistics_postprocessing(
            input_im,
            outputs,
            self.test_nms_thresh,
            self.max_detections_per_image,
            self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_mc_dropout_ensembles(self, input_im):
        """
        This function produces results using multiple runs of MC dropout, through fusion before or after
        the non-maximum suppression step.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT.BOX_MERGE_MODE == 'pre_nms':
            return self.post_processing_standard_nms(input_im)
        else:
            outputs_dict = self.forward(
                input_im,
                return_anchorwise_output=False,
                num_mc_dropout_runs=self.num_mc_dropout_runs)
            n_fms = len(self.head_in_features)
            outputs_list = [{key: value[i * n_fms:(i + 1) * n_fms] if value is not None else value for key,
                             value in outputs_dict.items()} for i in range(self.num_mc_dropout_runs)]

            # Merge results:
            results = [
                inference_utils.general_standard_nms_postprocessing(
                    input_im,
                    self.retinanet_probabilistic_inference(
                        input_im,
                        outputs=outputs),
                    self.test_nms_thresh,
                    self.max_detections_per_image) for outputs in outputs_list]

            # Append per-ensemble outputs after NMS has been performed.
            ensemble_pred_box_list = [
                result.pred_boxes.tensor for result in results]
            ensemble_pred_prob_vectors_list = [
                result.pred_cls_probs for result in results]
            ensembles_class_idxs_list = [
                result.pred_classes for result in results]
            ensembles_pred_box_covariance_list = [
                result.pred_boxes_covariance for result in results]

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.test_nms_thresh,
                self.max_detections_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_bayes_od(self, input_im):
        """
        This function produces results using forms of bayesian inference instead of NMS for both category and box results.

        Args:
            input_im (list): an input im list generated from dataset handler.

        Returns:
            result (instances): object instances

        """
        box_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE
        cls_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE

        outputs = self.retinanet_probabilistic_inference(input_im)

        predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors = outputs

        keep = batched_nms(
            predicted_boxes,
            predicted_prob,
            classes_idxs,
            self.test_nms_thresh)

        keep = keep[: self.max_detections_per_image]

        match_quality_matrix = pairwise_iou(
            Boxes(predicted_boxes), Boxes(predicted_boxes))

        box_clusters_inds = match_quality_matrix[keep, :]
        box_clusters_inds = box_clusters_inds > self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD

        # Compute mean and covariance for every cluster.

        predicted_prob_vectors_list = []
        predicted_boxes_list = []
        predicted_boxes_covariance_list = []

        predicted_prob_vectors_centers = predicted_prob_vectors[keep]
        for box_cluster, predicted_prob_vectors_center in zip(
                box_clusters_inds, predicted_prob_vectors_centers):
            cluster_categorical_params = predicted_prob_vectors[box_cluster]
            center_binary_score, center_cat_idx = torch.max(
                predicted_prob_vectors_center, 0)
            cluster_binary_scores, cat_idx = cluster_categorical_params.max(
                1)
            class_similarity_idx = cat_idx == center_cat_idx
            if cls_merge_mode == 'bayesian_inference':
                predicted_prob_vectors_list.append(
                    cluster_categorical_params.mean(0).unsqueeze(0))
            else:
                predicted_prob_vectors_list.append(
                    predicted_prob_vectors_center.unsqueeze(0))

            # Switch to numpy as torch.inverse is too slow.
            cluster_means = predicted_boxes[box_cluster,
                                            :][class_similarity_idx].cpu().numpy()
            cluster_covs = predicted_boxes_covariance[box_cluster, :][class_similarity_idx].cpu(
            ).numpy()

            predicted_box, predicted_box_covariance = inference_utils.bounding_box_bayesian_inference(
                cluster_means, cluster_covs, box_merge_mode)
            predicted_boxes_list.append(
                torch.from_numpy(np.squeeze(predicted_box)))
            predicted_boxes_covariance_list.append(
                torch.from_numpy(predicted_box_covariance))

        # Switch back to cuda for the remainder of the inference process.
        result = Instances(
            (input_im[0]['image'].shape[1],
             input_im[0]['image'].shape[2]))

        if len(predicted_boxes_list) > 0:
            if cls_merge_mode == 'bayesian_inference':
                predicted_prob_vectors = torch.cat(
                    predicted_prob_vectors_list, 0)
                predicted_prob, classes_idxs = torch.max(
                    predicted_prob_vectors, 1)
            elif cls_merge_mode == 'max_score':
                predicted_prob_vectors = predicted_prob_vectors[keep]
                predicted_prob = predicted_prob[keep]
                classes_idxs = classes_idxs[keep]
            result.pred_boxes = Boxes(
                torch.stack(
                    predicted_boxes_list,
                    0).to(self.device))
            result.scores = predicted_prob
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.stack(
                predicted_boxes_covariance_list, 0).to(self.device)
        else:
            result.pred_boxes = Boxes(predicted_boxes)
            result.scores = torch.zeros(
                predicted_boxes.shape[0]).to(
                self.device)
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.empty(
                (predicted_boxes.shape + (4,))).to(self.device)
        return result

class ProbabilisticRetinaNetHead(RetinaNetHead):
    """
    The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg,
                 use_dropout,
                 dropout_rate,
                 compute_cls_var,
                 compute_bbox_cov,
                 bbox_cov_dims,
                 input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.cfg = cfg
        # Extract config information
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims

        # For consistency all configs are grabbed from original RetinaNet
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(nn.ReLU())

            if self.use_dropout:
                cls_subnet.append(nn.Dropout(p=self.dropout_rate))
                bbox_subnet.append(nn.Dropout(p=self.dropout_rate))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels,
            num_anchors *
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)

        for modules in [
                self.cls_subnet,
                self.bbox_subnet,
                self.cls_score,
                self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        # Create subnet for classification variance estimation.
        if self.compute_cls_var:
            self.cls_var = nn.Conv2d(
                in_channels,
                num_anchors *
                num_classes,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.cls_var.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, -10.0)

        # Create subnet for bounding box covariance estimation.
        if self.compute_bbox_cov:
            self.bbox_cov = nn.Conv2d(
                in_channels,
                num_anchors * self.bbox_cov_dims,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.bbox_cov.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            logits_var (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the variance of the logits modeled as a univariate
                Gaussian distribution at each spatial position for each of the A anchors and K object
                classes.

            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.

            bbox_cov (list[Tensor]): #lvl tensors, each has shape (N, Ax4 or Ax10, Hi, Wi).
                The tensor predicts elements of the box
                covariance values for every anchor. The dimensions of the box covarianc
                depends on estimating a full covariance (10) or a diagonal covariance matrix (4).
        """
        logits = []
        bbox_reg = []

        logits_var = []
        bbox_cov = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
            if self.compute_cls_var:
                logits_var.append(self.cls_var(self.cls_subnet(feature)))
            if self.compute_bbox_cov:
                bbox_cov.append(self.bbox_cov(self.bbox_subnet(feature)))

        return_vector = [logits, bbox_reg]

        if self.compute_cls_var:
            return_vector.append(logits_var)
        else:
            return_vector.append(None)

        if self.compute_bbox_cov:
            return_vector.append(bbox_cov)
        else:
            return_vector.append(None)

        return return_vector
