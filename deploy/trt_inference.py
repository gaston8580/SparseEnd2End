# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import torch
import ctypes
import argparse, json, copy
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from tool.utils.logging import ProgressBar
from typing import Optional, Dict, Any
from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.dist_utils import get_dist_info
from tool.utils.data_parallel import E2EDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint
from dataset.dataloader_wrapper import *
from tool.tester.test_sdk import *
from dataset import *
from modules.sparse4d_detector import *
from export_head_onnx import SparseE2E1st, SparseE2E2nd

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)  # 不使用科学计数法
# for compatibility with older numpy versions
if not hasattr(np, 'bool'):
    np.bool = np.bool_


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def load_torch_model(cfg, args):
    # build torch model
    model = build_module(cfg["model"])
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # model = E2EDataParallel(model, device_ids=[0])
    model.eval()
    return model


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def TRTEngineInit(trt_path):
    engine = load_engine(trt_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, input_name2idx_dict, output_name2idx_dict = [], [], [], {}, {}

    num_bindings = engine.num_bindings
    for binding_idx in range(num_bindings):
        binding_name = engine.get_binding_name(binding_idx)
        size = trt.volume(engine.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配device内存
        bindings.append(device_mem)
        if engine.binding_is_input(binding_idx):
            inputs.append(host_mem)
            input_name2idx_dict[binding_name] = binding_idx
        else:
            outputs.append(host_mem)
            output_name2idx_dict[binding_name] = binding_idx
    for name in output_name2idx_dict.keys():
        output_name2idx_dict[name] = output_name2idx_dict[name] - len(inputs)
    return inputs, outputs, bindings, context, input_name2idx_dict, output_name2idx_dict


def e2e_torch_inference(model, data_loader, args):
    results = []
    dataset = data_loader.dataset
    model = E2EDataParallel(model, device_ids=[0])
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if i >= args.compare_num:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        # results.extend(result)
        results.append(result[0]['plan_traj'])

        if args.vis:
            filename = data['img_metas'].data[0][0]['filename'][0]
            save_result_path_prefix = filename.split("/clip")[0]
            save_result_path_suffix = filename.split("/")[-2]
            save_result_path_dir = os.path.join(save_result_path_prefix, "samples_results")
            os.makedirs(save_result_path_dir, exist_ok=True)
            save_result_path = os.path.join(save_result_path_dir, save_result_path_suffix + ".json")
            # print("infer result writes to", save_result_path)
            def tensor_to_list(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [tensor_to_list(item) for item in obj]
                return obj
            result_serializable = {k: tensor_to_list(v) for k, v in result[0].items()}
            with open(save_result_path, "w") as f:
                json.dump(result_serializable, f, indent=True, ensure_ascii=False)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def e2e_trt_inference(model, data_loader, args):
    trt_torch_model1st = SparseE2E1st(copy.deepcopy(model))
    trt_torch_model2nd = SparseE2E2nd(copy.deepcopy(model))
    trt_torch_model1st.eval()
    trt_torch_model2nd.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = ProgressBar(len(dataset))

    det_cached_instance_feature, det_cached_anchor, det_cached_track_id, det_cached_confidence = None, None, None, None
    map_cached_instance_feature, map_cached_anchor, map_cached_track_id, map_cached_confidence = None, None, None, None
    metas_global2lidar, his_metas_lidar2global, det_prev_id, map_prev_id = None, None, None, None
    history_ts = None
    for i, data in enumerate(data_loader):
        if i >= args.compare_num:
            break
        with torch.no_grad():
            img, image_wh, lidar2img, ego_his_trajs = data['img'].data[0].cuda(), data['image_wh'].cuda(), \
                data['lidar2img'].cuda(), data['ego_his_trajs'].cuda()
            if i == 0:
                ## inference
                inputs1st = (img, image_wh, lidar2img, ego_his_trajs.float())
                result = trt_torch_model1st(*inputs1st)
            else:
                time_interval = data["timestamp"].cuda() - history_ts
                metas_global2lidar = data["img_metas"].data[0][0]["global2lidar"]
                metas_global2lidar = torch.from_numpy(metas_global2lidar).unsqueeze(0).cuda()

                ## inference
                inputs2nd = (img, time_interval.float(), image_wh, lidar2img, ego_his_trajs.float(), 
                             det_cached_instance_feature, det_cached_anchor, det_cached_track_id.int(), 
                             metas_global2lidar.float(), his_metas_lidar2global.float(), det_cached_confidence, 
                             det_prev_id, map_cached_instance_feature, map_cached_anchor, map_cached_track_id.int(), 
                             map_cached_confidence, map_prev_id)
                result = trt_torch_model2nd(*inputs2nd)

            det_prev_id = torch.tensor(300).int().cuda() if i == 0 else result['det_prev_id']
            map_prev_id = torch.tensor(67).int().cuda() if i == 0 else result['map_prev_id']
            det_cached_instance_feature, det_cached_anchor, det_cached_track_id, det_cached_confidence = \
                result['det_cached_instance_feature'], result['det_cached_anchor'], result['det_cached_track_id'], \
                result['det_cached_confidence']
            map_cached_instance_feature, map_cached_anchor, map_cached_track_id, map_cached_confidence = \
                result['map_cached_instance_feature'], result['map_cached_anchor'], result['map_cached_track_id'], \
                result['map_cached_confidence']
            
            history_ts = data["timestamp"].cuda()
            his_metas_lidar2global = data["img_metas"].data[0][0]["lidar2global"]
            his_metas_lidar2global = torch.from_numpy(his_metas_lidar2global).unsqueeze(0).cuda()
            results.append(result)

        if args.vis:
            filename = data['img_metas'].data[0][0]['filename'][0]
            save_result_path_prefix = filename.split("/clip")[0]
            save_result_path_suffix = filename.split("/")[-2]
            save_result_path_dir = os.path.join(save_result_path_prefix, "samples_results")
            os.makedirs(save_result_path_dir, exist_ok=True)
            save_result_path = os.path.join(save_result_path_dir, save_result_path_suffix + ".json")
            # print("infer result writes to", save_result_path)
            def tensor_to_list(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [tensor_to_list(item) for item in obj]
                return obj
            result_serializable = {k: tensor_to_list(v) for k, v in result[0].items()}
            with open(save_result_path, "w") as f:
                json.dump(result_serializable, f, indent=True, ensure_ascii=False)
        prog_bar.update()
    return results


def e2e_trt_engine_inference(data_loader, args):
    import pycuda.autoinit
    ctypes.CDLL(args.plugin_path)  # 加载plugin
    inputs1st, outputs1st, bindings1st, context1st, input_name2idx_dict1st, output_name2idx_dict1st = \
        TRTEngineInit(args.trt1st_path)

    results = []
    prog_bar = ProgressBar(len(data_loader.dataset))

    det_cached_instance_feature, det_cached_anchor, det_cached_track_id, det_cached_confidence = None, None, None, None
    map_cached_instance_feature, map_cached_anchor, map_cached_track_id, map_cached_confidence = None, None, None, None
    metas_global2lidar, his_metas_lidar2global, det_prev_id, map_prev_id = None, None, None, None
    history_ts, result = None, {}
    for i, data in enumerate(data_loader):
        if i >= args.compare_num:
            break
        with torch.no_grad():
            img, image_wh, lidar2img, ego_his_trajs = data['img'].data[0].cuda(), data['image_wh'].cuda(), \
                data['lidar2img'].cuda(), data['ego_his_trajs'].cuda()
            if i == 0:
                ## inference
                onnx_inputs1st = {"img": img, "image_wh": image_wh, "lidar2img": lidar2img, 
                                  "ego_his_trajs": ego_his_trajs}
                for key in input_name2idx_dict1st.keys():
                    input_data = onnx_inputs1st[key].cpu().numpy()
                    cp_idx = input_name2idx_dict1st[key]
                for key in input_name2idx_dict1st.keys():
                    input_data = onnx_inputs1st[key].cpu().numpy()
                    cp_idx = input_name2idx_dict1st[key]
                    np.copyto(inputs1st[cp_idx], input_data.ravel())  # 可验证input精度是否正确
                for inp, src in zip(bindings1st[:len(inputs1st)], inputs1st):
                    cuda.memcpy_htod_async(inp, src)  # cpu to gpu
                context1st.execute_v2(bindings1st)  # inference
                for out, dst in zip(outputs1st, bindings1st[len(inputs1st):]):
                    cuda.memcpy_dtoh_async(out, dst)  # gpu to cpu
                for name in output_name2idx_dict1st.keys():
                    idx = output_name2idx_dict1st[name]
                    result[name] = outputs1st[idx]
            else:
                time_interval = data["timestamp"].cuda() - history_ts
                metas_global2lidar = data["img_metas"].data[0][0]["global2lidar"]
                metas_global2lidar = torch.from_numpy(metas_global2lidar).unsqueeze(0).cuda()

                ## inference
                inputs2nd = (img, time_interval.float(), image_wh, lidar2img, ego_his_trajs, det_cached_instance_feature, \
                             det_cached_anchor, det_cached_track_id, metas_global2lidar, his_metas_lidar2global, \
                             det_cached_confidence, det_prev_id, map_cached_instance_feature, map_cached_anchor, \
                             map_cached_track_id, map_cached_confidence, map_prev_id)
                result = trt_torch_model2nd(*inputs2nd)

            det_prev_id = torch.tensor(300).int().cuda() if i == 0 else result['det_prev_id']
            map_prev_id = torch.tensor(67).int().cuda() if i == 0 else result['map_prev_id']
            det_cached_instance_feature, det_cached_anchor, det_cached_track_id, det_cached_confidence = \
                result['det_cached_instance_feature'], result['det_cached_anchor'], result['det_cached_track_id'], \
                result['det_cached_confidence']
            map_cached_instance_feature, map_cached_anchor, map_cached_track_id, map_cached_confidence = \
                result['map_cached_instance_feature'], result['map_cached_anchor'], result['map_cached_track_id'], \
                result['map_cached_confidence']
            
            history_ts = data["timestamp"].cuda()
            his_metas_lidar2global = data["img_metas"].data[0][0]["lidar2global"]
            his_metas_lidar2global = torch.from_numpy(his_metas_lidar2global).unsqueeze(0).cuda()
            results.append(result)

        if args.vis:
            filename = data['img_metas'].data[0][0]['filename'][0]
            save_result_path_prefix = filename.split("/clip")[0]
            save_result_path_suffix = filename.split("/")[-2]
            save_result_path_dir = os.path.join(save_result_path_prefix, "samples_results")
            os.makedirs(save_result_path_dir, exist_ok=True)
            save_result_path = os.path.join(save_result_path_dir, save_result_path_suffix + ".json")
            # print("infer result writes to", save_result_path)
            def tensor_to_list(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [tensor_to_list(item) for item in obj]
                return obj
            result_serializable = {k: tensor_to_list(v) for k, v in result[0].items()}
            with open(save_result_path, "w") as f:
                json.dump(result_serializable, f, indent=True, ensure_ascii=False)
        prog_bar.update()
    return results


def trt_precision_check():
    args = parse_args()
    cfg = read_cfg(args.config)  # dict
    cfg["model"]["img_backbone"]["init_cfg"] = {}
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)
    torch_model = load_torch_model(cfg, args)

    dataset = build_module(cfg["data"]["test"])
    data_loader = dataloader_wrapper_without_dist(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg["data"]["workers_per_gpu"],
        dist=False,
        shuffle=False,
    )

    # inference
    # torch_outputs = e2e_torch_inference(torch_model, data_loader, args)
    trt_outputs = e2e_trt_inference(torch_model, data_loader, args)
    trt_engine_outputs = e2e_trt_engine_inference(data_loader, args)

    if args.save_results:
        kwargs = {}
        eval_kwargs = cfg.get("evaluation", {}).copy()
        for key in ["interval"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        if eval_kwargs.get("jsonfile_prefix", None) is None:
            eval_kwargs["jsonfile_prefix"] = os.path.join("./e2e_worklog", 
                                                          os.path.splitext(os.path.basename(args.config))[0], "eval",)
        print("\n", eval_kwargs)
        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Train E2E detector")
    parser.add_argument("--config", default="dataset/config/sparsee2e_bs1_stage2_no_aug_zdrive.py", help="train config file path",)
    parser.add_argument("--checkpoint", default="ckpt/iter_796.pth", help="checkpoint file",)
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--deterministic", action="store_true", help="whether to set deterministic options for CUDNN backend.",)
    parser.add_argument("--vis", action="store_true", default=False, help="whether to visulize for outputs.",)
    parser.add_argument("--eval", type=str, nargs="+", default="bbox", 
                        help='evaluation metrics, which depends on the dataset, e.g., "bbox"',)
    parser.add_argument("--save_results", default=False, help='whether to save inference results.',)
    parser.add_argument("--compare_num", default=1, help='compare number of torch and trt outputs.',)
    parser.add_argument("--trt1st_path", default='deploy/onnx/SparseE2E1st.engine', help='SparseE2E1st path.',)
    parser.add_argument("--trt2nd_path", default='deploy/onnx/SparseE2E2nd.engine', help='SparseE2E2nd path.',)
    parser.add_argument("--plugin_path", default='deploy/dfa_plugin/lib/deformableAttentionAggr.so', 
                        help='SparseE2E2nd path.',)

    return parser.parse_args()


if __name__ == "__main__":
    
    trt_precision_check()
