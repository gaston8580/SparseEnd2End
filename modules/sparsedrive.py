# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch

from inspect import signature
from modules.cnn.base_detector import BaseDetector
from modules.backbone import *
from modules.neck import *
from modules.head import *
from modules.head.blocks.core_blocks import *

from tool.runner.fp16_utils import force_fp32, auto_fp16

try:
    from .ops import feature_maps_format

    DFA_VALID = True
except:
    DFA_VALID = False

__all__ = ["SparseDrive"]


class SparseDrive(BaseDetector):
    def __init__(
        self,
        task_config: dict,
        img_backbone,
        img_neck,
        det_head,
        map_head=None,
        motion_plan_head=None,
        depth_branch=None,
        use_grid_mask=True,
        use_deformable_func=False,
        init_cfg=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        self.task_config = task_config

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.img_backbone = build_module(img_backbone)
        if img_neck is not None:
            self.img_neck = build_module(img_neck)
        if self.task_config['with_det']:
            self.det_head = build_module(det_head)
        if self.task_config['with_map']:
            self.map_head = build_module(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_module(motion_plan_head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DFA_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_module(depth_branch)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]  # (1, 6, 3, 256, 704)
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)  # [1*6, 3, 256, 704]
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        det_model_outs = self.det_head(feature_maps, data)

        # map head forward
        if self.task_config['with_map']:
            map_model_outs = self.map_head(feature_maps, data)

        # motion head forward
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.motion_plan_head(
                det_model_outs, 
                map_model_outs, 
                feature_maps,
                data,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.instance_bank.anchor_handler,
            )
        else:
            motion_output, planning_output = None, None
        
        losses = dict()
        det_output = self.det_head.loss(det_model_outs, data)
        losses.update(det_output)
        if self.task_config['with_map']:
            map_losses_output = self.map_head.loss(map_model_outs, data)
            losses.update(map_losses_output)
        if self.task_config['with_motion_plan']:
            motion_loss_cache = dict(indices=self.det_head.sampler.indices,)
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
        
        if depths is not None and "gt_depth" in data:
            losses["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return losses

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        # det head forward
        model_outs = self.det_head(feature_maps, data)
        det_results = self.det_head.post_process(model_outs)
        batch_size = len(det_results)

        # map head forward
        if self.task_config['with_map']:
            map_model_outs = self.map_head(feature_maps, data)
            map_results = self.map_head.post_process(map_model_outs)
        
        # motion head forward
        if self.task_config['with_motion_plan']:
            agent_hs = model_outs['instance_feature']
            map_hs = map_model_outs['instance_feature'] if self.map_head is not None else None
            motion_model_outs = self.motion_plan_head(agent_hs, map_hs, data)
            motion_results = self.motion_plan_head.post_process(motion_model_outs, data)

        output = [dict()] * batch_size
        for i in range(batch_size):
            output[i].update(det_results[i])
            if self.map_head is not None:
                output[i].update(map_results[i])
            if self.motion_plan_head is not None:
                output[i].update(motion_results[i])
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
