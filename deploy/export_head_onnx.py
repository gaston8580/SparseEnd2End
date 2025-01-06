# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import copy
import logging
import argparse

import onnx
from onnxsim import simplify

import torch
from torch import nn

from modules.sparsedrive import *
from modules.head.blocks.instance_bank import topk
from modules.ops import deformable_aggregation_function as DAF

from tool.utils.config import read_cfg
from typing import Optional, Dict, Any

from tool.utils.logger import set_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Head!")
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x4_bs22_256x704_VAD.py",
        help="deploy config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="e2e_worklog/sparse4d_temporal_r50_1x4_bs22_256x704_VAD/iter_330.pth",
        help="deploy ckpt path",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/export_head_onnx.log",
    )
    parser.add_argument(
        "--save_onnx1",
        type=str,
        default="deploy/onnx/SparseE2E1st.onnx",
    )
    parser.add_argument(
        "--save_onnx2",
        type=str,
        default="deploy/onnx/SparseE2E2nd.onnx",
    )
    parser.add_argument(
        "--export_2nd", default=1, action="store_true", help="export sparse4dhead2nd or sparse4dhead1nd onnx."
    )
    parser.add_argument(
        "--export_bank", default=0, action="store_true", help="whether export instance_bank onnx model."
    )
    args = parser.parse_args()
    return args


class Sparse4DHead1st(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead1st, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(
        self,
        feature_maps,
        image_wh,
        lidar2img,
    ):  
        # Instance bank get inputs
        cached_instance_feature = None
        temp_anchor_embed = None
        (instance_feature, anchor, _, time_interval, _) = self.instance_bank.get_cacahe_trt()

        # DFA inputs
        metas = {"image_wh": image_wh, "lidar2img": lidar2img,}

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        tmp_outs = []
        for i, op in enumerate(self.operation_order):
            print("i: ", i, "\top: ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    cached_instance_feature,
                    cached_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(instance_feature, anchor_embed, metas)
                points_2d = (
                    self.layers[i].project_points(
                        key_points,
                        metas["lidar2img"],  # lidar2img
                        metas.get("image_wh"),
                    ).permute(0, 2, 3, 1, 4).reshape(bs, num_anchor, self.layers[i].num_pts, self.layers[i].num_cams, 2)
                )
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
                        bs,
                        num_anchor,
                        self.layers[i].num_pts,
                        self.layers[i].num_cams,
                        self.layers[i].num_levels,
                        self.layers[i].num_groups,
                    )
                )

                features = DAF(*feature_maps, points_2d, weights)
                features = features.reshape(bs, num_anchor, self.layers[i].embed_dims)
                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(features)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
        
        (temp_confidence, cached_confidence, cached_instance_feature, cached_anchor) = \
            self.instance_bank.cache_trt(instance_feature, anchor, cls)
        
        prev_id = torch.tensor(0, dtype=torch.int64).cuda()
        prev_id, track_id, cached_track_id = self.instance_bank.get_track_id_trt(cls, temp_confidence, prev_id)

        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id,
            cached_track_id,
            cached_confidence,
            cached_instance_feature,
            cached_anchor,
            prev_id,
        )

    def forward(
        self,
        img,
        image_wh,
        lidar2img,
        ego_his_trajs,
    ):
        feature_maps = self.model.extract_feat(img)  # feature, spatial_shapes, level_start_index

        # det head forward
        head = self.model.head
        (
            det_instance_feature,
            det_anchor,
            det_cls,
            det_qt,
            det_track_id,
            det_cached_track_id,
            det_cached_confidence,
            det_cached_instance_feature,
            det_cached_anchor,
            det_prev_id,  # 300
        ) = self.head_forward(
            head,
            feature_maps,
            image_wh,
            lidar2img,
        )

        # map head forward
        map_head = self.model.map_head
        (
            map_instance_feature,
            map_anchor,
            map_cls,
            map_qt,
            map_track_id,
            map_cached_track_id,
            map_cached_confidence,
            map_cached_instance_feature,
            map_cached_anchor,
            map_prev_id,  # 67
        ) = self.head_forward(
            map_head,
            feature_maps,
            image_wh,
            lidar2img,
        )

        # motion head forward
        motion_head = self.model.motion_head
        data = {'ego_his_trajs': ego_his_trajs}
        outputs = motion_head(det_instance_feature, map_instance_feature, data)
        plan_traj = outputs['ego_fut_preds']
        prob = outputs['prob']

        return (
            ## det outputs
            det_cached_track_id,
            det_cached_confidence,
            det_cached_instance_feature,
            det_cached_anchor,
            ## map outputs
            map_cached_track_id,
            map_cached_confidence,
            map_cached_instance_feature,
            map_cached_anchor,
            ## motion outputs
            plan_traj,
            prob,
        )


class Sparse4DHead2nd(nn.Module):
    def __init__(self, model):
        super(Sparse4DHead2nd, self).__init__()
        self.model = model

    @staticmethod
    def head_forward(
        self,
        feature_maps,
        time_interval,
        image_wh,
        lidar2img,
        cached_instance_feature,  # cache
        cached_anchor,            # cache
        cached_track_id,
        metas_global2lidar,
        his_metas_lidar2global,
        cached_confidence,
        prev_id,
    ):
        (instance_feature, anchor, cached_anchor, time_interval, mask) = \
            self.instance_bank.get_cacahe_trt(cached_anchor, time_interval, metas_global2lidar, his_metas_lidar2global)

        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(cached_anchor)

        # DAF inputs
        metas = {"lidar2img": lidar2img, "image_wh": image_wh,}

        prediction = []
        tmp_outs = []
        for i, op in enumerate(self.operation_order):
            print("op:  ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    cached_instance_feature,
                    cached_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(instance_feature, anchor_embed, metas)
                points_2d = (self.layers[i].project_points(
                        key_points,
                        metas["lidar2img"],  # lidar2img
                        metas.get("image_wh"),
                    ).permute(0, 2, 3, 1, 4).reshape(bs, num_anchor, self.layers[i].num_pts, self.layers[i].num_cams, 2)
                )
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
                        bs,
                        num_anchor,
                        self.layers[i].num_pts,
                        self.layers[i].num_cams,
                        self.layers[i].num_levels,
                        self.layers[i].num_groups,
                    )
                )

                features = DAF(*feature_maps, points_2d, weights)
                features = features.reshape(bs, num_anchor, self.layers[i].embed_dims)
                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
                tmp_outs.append(features)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)

                # update in head refine
                if len(prediction) == self.num_single_frame_decoder:
                    N = (self.instance_bank.num_anchor - self.instance_bank.num_temp_instances)
                    cls = cls.max(dim=-1).values
                    _, (selected_feature, selected_anchor) = topk(cls, N, instance_feature, anchor)
                    selected_feature = torch.cat([cached_instance_feature, selected_feature], dim=1)
                    selected_anchor = torch.cat([cached_anchor, selected_anchor], dim=1)
                    instance_feature = torch.where(mask[:, None, None], selected_feature, instance_feature)
                    anchor = torch.where(mask[:, None, None], selected_anchor, anchor)
                    cached_confidence = torch.where(mask[:, None], cached_confidence, cached_confidence.new_tensor(0))
                    cached_track_id = torch.where(mask[:, None], cached_track_id, cached_track_id.new_tensor(-1))

                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder:
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

        (temp_confidence, cached_confidence, cached_instance_feature, cached_anchor) = \
            self.instance_bank.cache_trt(instance_feature, anchor, cls, cached_confidence)
        
        prev_id, track_id, cached_track_id = \
            self.instance_bank.get_track_id_trt(cls, temp_confidence, prev_id, cached_track_id)

        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id,
            cached_track_id,
            cached_confidence,
            cached_instance_feature,
            cached_anchor,
            prev_id,
        )

    def forward(
        self,
        img,
        time_interval,
        image_wh,
        lidar2img,
        ego_his_trajs,
        det_cached_instance_feature,
        det_cached_anchor,
        det_cached_track_id,
        metas_global2lidar,
        his_metas_lidar2global,
        det_cached_confidence,
        det_prev_id,
        map_cached_instance_feature,
        map_cached_anchor,
        map_cached_track_id,
        map_cached_confidence,
        map_prev_id,
    ):
        feature_maps = self.model.extract_feat(img)  # feature, spatial_shapes, level_start_index

        # det head forward
        head = self.model.head
        (
            det_instance_feature,
            det_anchor,
            det_cls,
            det_qt,
            det_track_id,
            det_cached_track_id,
            det_cached_confidence,
            det_cached_instance_feature,
            det_cached_anchor,
            det_prev_id,
        ) = self.head_forward(
            head,
            feature_maps,
            time_interval,
            image_wh,
            lidar2img,
            det_cached_instance_feature,
            det_cached_anchor,
            det_cached_track_id,
            metas_global2lidar,
            his_metas_lidar2global,
            det_cached_confidence,
            det_prev_id,
        )

        # map head forward
        map_head = self.model.map_head
        (
            map_instance_feature,
            map_anchor,
            map_cls,
            map_qt,
            map_track_id,
            map_cached_track_id,
            map_cached_confidence,
            map_cached_instance_feature,
            map_cached_anchor,
            map_prev_id,
        ) = self.head_forward(
            map_head,
            feature_maps,
            time_interval,
            image_wh,
            lidar2img, 
            map_cached_instance_feature,
            map_cached_anchor,
            map_cached_track_id,
            metas_global2lidar,
            his_metas_lidar2global,
            map_cached_confidence,
            map_prev_id,
        )
        
        # motion head forward
        motion_head = self.model.motion_head
        data = {'ego_his_trajs': ego_his_trajs}
        outputs = motion_head(det_instance_feature, map_instance_feature, data)
        plan_traj = outputs['ego_fut_preds']
        prob = outputs['prob']

        return (
            # det outputs
            det_cached_track_id,
            det_cached_confidence,
            det_cached_instance_feature,
            det_cached_anchor,
            det_prev_id,
            # map outputs
            map_cached_track_id,
            map_cached_confidence,
            map_cached_instance_feature,
            map_cached_anchor,
            map_prev_id,
            # motion outputs
            plan_traj,
            prob,
        )


class InstanceBank(nn.Module):
    def __init__(self, model):
        super(InstanceBank, self).__init__()
        self.model = model
    
    ## get cacahe
    # def forward(
    #     self,
    #     cached_anchor,
    #     time_interval,
    #     metas_global2lidar,
    #     his_metas_lidar2global,
    # ):
    #     (instance_feature, anchor, cached_anchor, time_interval,) = \
    #         self.model.head.instance_bank.get_cacahe_trt(cached_anchor, time_interval, metas_global2lidar, 
    #                                                      his_metas_lidar2global)
    #     return (instance_feature, anchor, cached_anchor, time_interval,)
    
    ## process cache
    # def forward(
    #     self,
    #     instance_feature,
    #     anchor,
    #     cls,
    #     cached_confidence,
    # ):
    #     (temp_confidence, cached_confidence, cached_feature, cached_anchor) = \
    #     self.model.head.instance_bank.cache_trt(instance_feature, anchor, cls, cached_confidence)
    #     return (temp_confidence, cached_confidence, cached_feature, cached_anchor,)
    
    ## get track_id
    def forward(
        self,
        cls,
        temp_confidence,
        prev_id,
        cached_track_id,
    ):
        prev_id, track_id, cached_track_id = \
            self.model.head.instance_bank.get_track_id_trt(cls, temp_confidence, prev_id, cached_track_id)
        return prev_id, track_id, cached_track_id


def dummpy_input(
    model,
    bs: int,
    nums_cam: int,
    input_h: int,
    input_w: int,
    nums_query=900,
    nums_topk=600,
    embed_dims=256,
    anchor_dims=11,
    first_frame=True,
    logger=None,
):
    h_4x, w_4x = input_h // 4, input_w // 4
    h_8x, w_8x = input_h // 8, input_w // 8
    h_16x, w_16x = input_h // 16, input_w // 16
    h_32x, w_32x = input_h // 32, input_w // 32
    feature_size = nums_cam * (
        h_4x * w_4x + h_8x * w_8x + h_16x * w_16x + h_32x * w_32x
    )
    dummy_feature = torch.randn(bs, feature_size, embed_dims).float().cuda()

    dummy_spatial_shapes = (
        torch.tensor([[h_4x, w_4x], [h_8x, w_8x], [h_16x, w_16x], [h_32x, w_32x]])
        .int()
        .unsqueeze(0)
        .repeat(nums_cam, 1, 1)
        .cuda()
    )

    scale_start_index = dummy_spatial_shapes[..., 0] * dummy_spatial_shapes[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0).int()
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    dummy_level_start_index = scale_start_index.reshape(nums_cam, 4)

    instance_feature = model.head.instance_bank.instance_feature  # (900, 256)
    dummy_instance_feature = (
        instance_feature[None].repeat((bs, 1, 1)).cuda()
    )  # (bs, 900, 256)

    anchor = model.head.instance_bank.anchor  # (900, 11)
    dummy_anchor = anchor[None].repeat((bs, 1, 1)).cuda()  # (bs, 900, 11)

    dummy_time_interval = torch.tensor(
        [model.head.instance_bank.default_time_interval] * bs
    ).cuda()

    dummy_temp_instance_feature = (
        torch.zeros((bs, nums_topk, embed_dims)).float().cuda()
    )
    dummy_temp_anchor = torch.zeros((bs, nums_topk, anchor_dims)).float().cuda()
    dummy_mask = torch.randint(0, 2, size=(bs,)).int().cuda()
    dummy_track_id = -1 * torch.ones((bs, nums_query)).int().cuda()

    dummy_image_wh = (
        torch.tensor([input_w, input_h])
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bs, nums_cam, 1)
        .to(dummy_feature)
    )

    dummy_lidar2img = torch.randn(bs, nums_cam, 4, 4).to(dummy_feature)

    logger.debug(f"Dummy input : hape&Type&Device Msg >>>>>>")
    roi_x = [
        "dummy_feature",
        "dummy_spatial_shapes",
        "dummy_level_start_index",
        "dummy_instance_feature",
        "dummy_anchor",
        "dummy_time_interval",
        "dummy_image_wh",
        "dummy_lidar2img",
    ]
    for x in roi_x:
        logger.debug(
            f"{x}\t:\tshape={eval(x).shape},\tdtype={eval(x).dtype},\tdevice={eval(x).device}"
        )

    if first_frame:
        logger.debug(f"Frame > 1: Extra dummy input is needed >>>>>>>")
        roi_y = [
            "dummy_temp_instance_feature",
            "dummy_temp_anchor",
            "dummy_mask",
            "dummy_track_id",
        ]
        for y in roi_y:
            logger.debug(
                f"{y}\t:\tshape={eval(y).shape},\tdtype={eval(y).dtype},\tdevice={eval(y).device}"
            )

    return (
        dummy_feature,
        dummy_spatial_shapes,
        dummy_level_start_index,
        dummy_instance_feature,
        dummy_anchor,
        dummy_time_interval,
        dummy_temp_instance_feature,
        dummy_temp_anchor,
        dummy_mask,
        dummy_track_id,
        dummy_image_wh,
        dummy_lidar2img,
    )


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_onnx1), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, True)
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.cuda().eval()

    BS = 1
    NUMS_CAM = 6
    C = 3
    INPUT_H = 256
    INPUT_W = 704
    first_frame = True
    (
        dummy_feature,
        dummy_spatial_shapes,
        dummy_level_start_index,
        dummy_instance_feature,
        dummy_anchor,
        dummy_time_interval,
        dummy_temp_instance_feature,
        dummy_temp_anchor,
        dummy_mask,
        dummy_track_id,
        dummy_image_wh,
        dummy_lidar2img,
    ) = dummpy_input(
        model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=first_frame, logger=logger
    )
    dummy_img = torch.randn(BS, NUMS_CAM, C, INPUT_H, INPUT_W).cuda()

    img = dummy_img
    time_interval = dummy_time_interval
    image_wh = dummy_image_wh
    lidar2img = dummy_lidar2img
    ego_his_trajs = torch.rand(1, 2, 2).cuda()
    metas_global2lidar = torch.rand(1, 4, 4).cuda()
    his_metas_lidar2global = torch.rand(1, 4, 4).cuda()
    # det inputs
    num_det, num_det_cache = 900, 600
    det_cached_instance_feature = dummy_temp_instance_feature
    det_cached_anchor = dummy_temp_anchor
    det_cached_track_id = dummy_track_id
    det_cached_confidence = torch.rand(1, num_det_cache).cuda()
    det_prev_id = torch.tensor(num_det - num_det_cache).cuda()
    # map inputs
    num_map, num_map_cache = 100, 33
    map_cached_instance_feature = torch.rand(1, num_map_cache, 256).cuda()
    map_cached_anchor = torch.rand(1, num_map_cache, 40).cuda()
    map_cached_track_id = -1 * torch.ones((1, num_map)).int().cuda()
    map_cached_confidence = torch.rand(1, num_map_cache).cuda()
    map_prev_id = torch.tensor(num_map - num_map_cache).cuda()
    
    if not args.export_bank:
        if not args.export_2nd:
            backbone_first_frame_head = Sparse4DHead1st(copy.deepcopy(model))
            logger.info("Export Sparse4DHead1st Onnx >>>>>>>>>>>>>>>>")
            time.sleep(2)
            with torch.no_grad():
                torch.onnx.export(
                    backbone_first_frame_head,
                    (
                        img,                
                        image_wh,
                        lidar2img,
                        ego_his_trajs,
                    ),
                    args.save_onnx1,
                    input_names=[
                        "img",
                        "image_wh",
                        "lidar2img",
                        "ego_his_trajs",
                    ],
                    output_names=[
                        "det_cached_track_id",
                        "det_cached_confidence",
                        "det_cached_instance_feature",
                        "det_cached_anchor",
                        "map_cached_track_id",
                        "map_cached_confidence",
                        "map_cached_instance_feature",
                        "map_cached_anchor",
                        "plan_traj",
                        "prob",
                    ],
                    # output_names=None,
                    opset_version=15,
                    do_constant_folding=False,
                    export_params=True,
                    verbose=True,
                )

                os.system(f'onnxsim {args.save_onnx1} {args.save_onnx1}')
                os.system(f'trtexec --onnx={args.save_onnx1} --saveEngine={args.save_onnx1.replace(".onnx", ".engine")} '
                          '--plugins=deploy/dfa_plugin/lib/deformableAttentionAggr.so')
        else:
            backbone_second_frame_head = Sparse4DHead2nd(copy.deepcopy(model))
            logger.info("Export Sparse4DHead2nd Onnx >>>>>>>>>>>>>>>>")
            time.sleep(2)
            with torch.no_grad():
                torch.onnx.export(
                    backbone_second_frame_head,
                    (
                        img,
                        time_interval,
                        image_wh,
                        lidar2img,
                        ego_his_trajs,
                        det_cached_instance_feature,
                        det_cached_anchor,
                        det_cached_track_id,
                        metas_global2lidar,
                        his_metas_lidar2global,
                        det_cached_confidence,
                        det_prev_id,
                        map_cached_instance_feature,
                        map_cached_anchor,
                        map_cached_track_id,
                        map_cached_confidence,
                        map_prev_id,
                    ),
                    args.save_onnx2,
                    input_names=[
                        "img",
                        "time_interval",
                        "image_wh",
                        "lidar2img",
                        "ego_his_trajs",
                        "det_cached_instance_feature",
                        "det_cached_anchor",
                        "det_cached_track_id",
                        "metas_global2lidar",
                        "his_metas_lidar2global",
                        "det_cached_confidence",
                        "det_prev_id",
                        "map_cached_instance_feature",
                        "map_cached_anchor",
                        "map_cached_track_id",
                        "map_cached_confidence",
                        "map_prev_id",
                    ],
                    output_names=[
                        "det_cached_track_id",
                        "det_cached_confidence",
                        "det_cached_instance_feature",
                        "det_cached_anchor",
                        "det_prev_id",
                        "map_cached_track_id",
                        "map_cached_confidence",
                        "map_cached_instance_feature",
                        "map_cached_anchor",
                        "map_prev_id",
                        "plan_traj",
                        "prob",
                    ],
                    opset_version=15,
                    do_constant_folding=False,
                    export_params=True,
                    verbose=True,
                )

                os.system(f'onnxsim {args.save_onnx2} {args.save_onnx2}')
                os.system(f'trtexec --onnx={args.save_onnx2} --saveEngine={args.save_onnx2.replace(".onnx", ".engine")} '
                          '--plugins=deploy/dfa_plugin/lib/deformableAttentionAggr.so')
    else:
        instance_bank = InstanceBank(copy.deepcopy(model))
        print("Export InstanceBank Onnx >>>>>>>>>>>>>>>>")
        
        # time_interval = torch.tensor([0.5]).cuda()
        # metas_global2lidar = torch.rand(1, 4, 4).cuda()
        # his_metas_lidar2global = torch.rand(1, 4, 4).cuda()
        # inutputs = (dummy_temp_anchor, time_interval, metas_global2lidar, his_metas_lidar2global)
        # input_names = ['cached_anchor', 'time_interval', 'metas_global2lidar', 'his_metas_lidar2global']
        # output_names = ['instance_feature', 'anchor', 'cached_anchor', 'time_interval']
        # onnx_save_path = "deploy/onnx/bank_get.onnx"
        # with torch.no_grad():
        #     torch.onnx.export(instance_bank, inutputs, onnx_save_path, input_names=input_names, output_names=output_names, 
        #                       opset_version=11, export_params=True, verbose=True)
        #     os.system(f'onnxsim {onnx_save_path} {onnx_save_path}')
        #     os.system(f'trtexec --onnx={onnx_save_path} --saveEngine={onnx_save_path.replace(".onnx", ".engine")}')
        
        # cls = torch.rand(1, 900, 10).cuda()
        # cached_confidence = torch.rand(1, 600).cuda()
        # inutputs = (dummy_instance_feature, dummy_anchor, cls, cached_confidence)
        # input_names = ['instance_feature', 'anchor', 'cls', 'cached_confidence']
        # output_names = ['temp_confidence', 'cached_confidence', 'cached_feature', 'cached_anchor']
        # onnx_save_path = "deploy/onnx/bank_cache.onnx"
        # with torch.no_grad():
        #     torch.onnx.export(instance_bank, inutputs, onnx_save_path, input_names=input_names, output_names=output_names, 
        #                       opset_version=15, export_params=True, verbose=True)
        #     os.system(f'onnxsim {onnx_save_path} {onnx_save_path}')
        #     os.system(f'trtexec --onnx={onnx_save_path} --saveEngine={onnx_save_path.replace(".onnx", ".engine")}')

        cls = torch.rand(1, 900, 10).cuda()
        temp_confidence = torch.rand(1, 900).cuda()
        prev_id = torch.tensor(900).cuda()
        cached_track_id = torch.ones(1, 900).cuda()
        inutputs = (cls, temp_confidence, prev_id, cached_track_id)
        input_names = ['cls', 'temp_confidence', 'prev_id', 'cached_track_id']
        output_names = ['prev_id', 'track_id', 'cached_track_id']
        onnx_save_path = "deploy/onnx/bank_get_track_id.onnx"
        with torch.no_grad():
            torch.onnx.export(instance_bank, inutputs, onnx_save_path, input_names=input_names, output_names=output_names, 
                                opset_version=15, export_params=True, verbose=True)
            os.system(f'onnxsim {onnx_save_path} {onnx_save_path}')
            os.system(f'trtexec --onnx={onnx_save_path} --saveEngine={onnx_save_path.replace(".onnx", ".engine")}')