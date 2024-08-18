# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import argparse
import numpy as np

import onnx
import onnxruntime as ort

import logging
from tool.utils.logger import set_logger

from modules.sparse4d_detector import *
from typing import Optional, Dict, Any
from tool.utils.config import read_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Onnx Model Inference Consistency Check!"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnxlog/onnx_consistencycheck.log",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="deploy/onnxlog/sparse4dbackbone.onnx",
    )
    args = parser.parse_args()
    return args


def model_infer(model, dummy_img):
    with torch.no_grad():
        feature, spatial_shapes, level_start_index = model.extract_feat(dummy_img)
    return feature.cpu().numpy()


def onnx_infer(dummy_img, args):
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {session.get_inputs()[0].name: dummy_img}
    ort_outs = session.run(["feature"], ort_inputs)
    return ort_outs[0]


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


if __name__ == "__main__":
    args = parse_args()
    logger, console_handler, file_handler = set_logger(args.log)
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    logger.info("Sparse4d Backbone Onnx Inference Consistency Check!...")

    cfg = read_cfg(args.cfg)
    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)

    np.random.seed(1)

    BS = 1
    NUM_CAMS = 6
    C = 3
    H = 256
    W = 704
    dummy_img = np.random.rand(BS, NUM_CAMS, C, H, W).astype(np.float32)

    output1 = model_infer(model.cuda().eval(), torch.from_numpy(dummy_img).cuda())
    output2 = onnx_infer(dummy_img, args)

    cosine_distance = 1 - np.dot(output1.flatten(), output2.flatten()) / (
        np.linalg.norm(output1.flatten()) * np.linalg.norm(output2.flatten())
    )
    print(type(cosine_distance))
    logger.info(f"cosine_distance = {float(cosine_distance)}")
    logger.info(f"max(abs())      = {float((np.abs(output1 - output2)).max())}")
