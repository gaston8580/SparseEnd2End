# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import time
import torch
import pickle
import shutil
import tempfile
import numpy as np
import pycocotools.mask as mask_util
import torch.distributed as dist
import json
from tool.utils.logging import ProgressBar
from tool.utils.dist_utils import get_dist_info


__all__ = ["single_gpu_test", "custom_multi_gpu_test"]


def single_gpu_test(model, data_loader, is_vis):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        results.extend(result)

        if is_vis:
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


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    def _collect_results_cpu(result_part, size, tmpdir=None):
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
            if rank == 0:
                os.makedirs(".dist_test", exist_ok=True)
                tmpdir = tempfile.mkdtemp(dir=".dist_test")
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
                )
                dir_tensor[: len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            os.makedirs(tmpdir)
        # dump the part result to the dir
        pickle.dump(result_part, os.path.join(tmpdir, f"part_{rank}.pkl"))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                part_file = os.path.join(tmpdir, f"part_{i}.pkl")
                part_list.append(pickle.load(part_file))
            # sort the results
            ordered_results = []
            """
            bacause we change the sample of the evaluation stage to make sure that
            each gpu will handle continuous sample,
            """
            # for res in zip(*part_list):
            for res in part_list:
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)
            return ordered_results

    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if "bbox_results" in result.keys():
                    bbox_result = result["bbox_results"]
                    batch_size = len(result["bbox_results"])
                    bbox_results.extend(bbox_result)
                if (
                    "mask_results" in result.keys()
                    and result["mask_results"] is not None
                ):
                    mask_result = _custom_encode_mask_results(result["mask_results"])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    # if gpu_collect:
    #     bbox_results = collect_results_gpu(bbox_results, len(dataset))
    #     if have_mask:
    #         mask_results = collect_results_gpu(mask_results, len(dataset))
    #     else:
    #         mask_results = None
    # else:
    bbox_results = _collect_results_cpu(bbox_results, len(dataset), tmpdir)
    tmpdir = tmpdir + "_mask" if tmpdir is not None else None
    if have_mask:
        mask_results = _collect_results_cpu(mask_results, len(dataset), tmpdir)
    else:
        mask_results = None

    if mask_results is None:
        return bbox_results
    return {"bbox_results": bbox_results, "mask_results": mask_results}


def _custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8")
            )[0]
        )  # encoded with RLE
    return [encoded_mask_results]
