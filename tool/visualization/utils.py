import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import cv2

def get_camera_name(filename):
    if "frontwide" in filename or "front_wide" in filename:
        camera_name = "camera0"
    elif "frontmain" in filename or "front_main" in filename:
        camera_name = "camera1"
    elif "leftfront" in filename or "left_front" in filename:
        camera_name = "camera2"
    elif "leftrear" in filename or "left_rear" in filename:
        camera_name = "camera3"
    elif "rightfront" in filename or "right_front" in filename:
        camera_name = "camera4"
    elif "rightrear" in filename or "right_rear" in filename:
        camera_name = "camera5"
    elif "rearmain" in filename or "rear_main" in filename:
        camera_name = "camera6"

    elif "fisheyeleft" in filename or "fisheye_left" in filename:
        camera_name = "camera7"
    elif "fisheyerear" in filename or "fisheye_rear" in filename:
        camera_name = "camera8"
    elif "fisheyefront" in filename or "fisheye_front" in filename:
        camera_name = "camera9"
    elif "fisheyeright" in filename or "fisheye_right" in filename:
        camera_name = "camera10"
    else:
        raise ValueError
    return camera_name


def load_ins_ex(ins_ex_path: str):
    """
    读取相机内外参
    Args:
      - ins_ex_path(str): 内外参文件所在目录
                          e.g. /data/sfs_turbo/hyq/data/B550M0/data_collection/
    Returns:
      list[dict[camera_name]]:
        - K(np.ndarray): shape=(3, 3)内参
        - D(np.ndarray): 畸变系数shape=(5,) pinhole dist: k1,k2,p1,p2,k3
                       or shape=(4,) fisheye dist: k1, k2, k3, k4
        - R_T(np.ndarray): shape=(4, 4)外参, lidar -> camera
    """
    ins_ex = {}
    # extrinsics
    extrinsic_path = os.path.join(ins_ex_path, "extrinsics/lidar2camera")
    for extrinsic_filename in os.listdir(extrinsic_path):
        # if 'fisheye' in extrinsic_filename or extrinsic_filename.startswith('_') or (not (extrinsic_filename.endswith('.yaml') or extrinsic_filename.endswith('.yml'))):
        if extrinsic_filename.startswith("_") or (
            not (
                extrinsic_filename.endswith(".yaml")
                or extrinsic_filename.endswith(".yml")
            )
        ):
            continue
        d = {}
        camera_name = get_camera_name(extrinsic_filename)
        with open(os.path.join(extrinsic_path, extrinsic_filename)) as f:
            ex = yaml.safe_load(f)
        d["R_T"] = np.array(ex["transform"])
        ins_ex[camera_name] = d

    # intrinsics & dist coeff
    intrinsic_path = os.path.join(ins_ex_path, "intrinsics")
    for intrinsic_filename in os.listdir(intrinsic_path):
        # if 'fisheye' in intrinsic_filename or intrinsic_filename.startswith('_') or (not (intrinsic_filename.endswith('.yaml') or intrinsic_filename.endswith('.yml'))):
        if intrinsic_filename.startswith("_") or (
            not (
                intrinsic_filename.endswith(".yaml")
                or intrinsic_filename.endswith(".yml")
            )
        ):
            continue
        camera_name = get_camera_name(intrinsic_filename)
        with open(os.path.join(intrinsic_path, intrinsic_filename)) as f:
            ins = yaml.safe_load(f)
        K = np.eye(3)
        K[0, 0] = ins["K"][0]
        K[1, 1] = ins["K"][1]
        K[0, 2] = ins["K"][2]
        K[1, 2] = ins["K"][3]
        ins_ex[camera_name]["K"] = K
        ins_ex[camera_name]["D"] = np.array(ins["D"])
        if len(ins["D"]) in [5, 8]:  # pinhole
            ins_ex[camera_name]["camera_model"] = "pinhole"
        elif len(ins["D"]) in [4]:  # fisheye
            ins_ex[camera_name]["camera_model"] = "fisheye"
        else:
            raise ValueError(
                f"{intrinsic_path}/{intrinsic_filename} distortion coefficients error"
            )

    try:
        lidar2imu_path = os.path.join(
            ins_ex_path, "extrinsics/lidar2imu/lidar2imu.yaml"
        )
        with open(lidar2imu_path) as f:
            data = yaml.safe_load(f)
        lidar2imu = np.eye(4)
        q = R.from_quat(
            [
                data["transform"]["rotation"]["x"],
                data["transform"]["rotation"]["y"],
                data["transform"]["rotation"]["z"],
                data["transform"]["rotation"]["w"],
            ]
        )
        lidar2imu[:3, :3] = q.as_matrix()
        lidar2imu[:3, 3] = [
            data["transform"]["translation"]["x"],
            data["transform"]["translation"]["y"],
            data["transform"]["translation"]["z"],
        ]
        ins_ex["lidar2imu"] = lidar2imu  # shape=(4, 4)
    except Exception:
        pass
    return ins_ex


def prepare_images_for_vconcat(img1, img2):
    if img1 is None or img2 is None:
        raise ValueError("One of the images is None.")

    shape1 = img1.shape
    shape2 = img2.shape
    
    if shape1[1] != shape2[1]:
        max_width = max(shape1[1], shape2[1])
        img1 = cv2.resize(img1, (max_width, shape1[0]))
        img2 = cv2.resize(img2, (max_width, shape2[0]))

    if len(shape1) == 3 and len(shape2) == 3 and shape1[2] != shape2[2]:
        if shape1[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    elif len(shape1) != len(shape2):
        if len(shape1) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    dtype1 = img1.dtype
    dtype2 = img2.dtype
    if dtype1 != dtype2:
        img2 = img2.astype(dtype1)

    return img1, img2