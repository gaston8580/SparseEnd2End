# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import json
import numpy as np
import math
from tqdm import tqdm
from pyquaternion import Quaternion
import copy
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
#from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor
from .map_utils.nuscmap_extractor import NuscMapExtractor

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


ego_width, ego_length = 1.85, 4.084

def geom2anno(map_geoms):
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords).tolist()
                vectors[label].append(line)
    return vectors


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = (R.T).tolist()  # points @ R.T + T # numpy(3,3)
    sweep["sensor2lidar_translation"] = T.tolist()  # numpy(3)
    return sweep


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i



def _fill_trainval_infos(nusc, nusc_map_extractor, nusc_can_bus, train_scenes, val_scenes, test=False, max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    ##Future  3s##
    fut_ts = 6
    ##History 2s
    his_ts = 1
    # train_nusc_infos = []
    # val_nusc_infos = []
    trainscene_nusc_infos = dict()
    valscene_nusc_infos = dict()
    frame_idx = 0
    cat2idx = {}


    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    for t in train_scenes:
        trainscene_nusc_infos[t] = list()
    for t in val_scenes:
        valscene_nusc_infos[t] = list()

    for sample in tqdm(nusc.sample, desc="Processing Sample"):
        lidar_token = sample["data"]["LIDAR_TOP"]
        # sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']


        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)


        if not os.path.isfile(lidar_path):
            raise FileNotFoundError('file "{}" does not exist'.format(lidar_path))

        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        ##

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            'prev': sample['prev'],
            'next': sample['next'],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],  # listlen=3
            "lidar2ego_rotation": cs_record["rotation"],  # listlen=4
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1



        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix  # numpy(3,3)
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix  # numpy(3,3)

        # extract map annos
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        lidar2global = ego2global @ lidar2ego

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)

        map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation)
        map_annos = geom2anno(map_geoms)
        info['map_annos'] = map_annos

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic.tolist())
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc,
                    sd_rec["prev"],
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                )  # dict
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = (
                np.array(
                    [
                        (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                        for anno in annotations
                    ],
                    dtype=np.int32,
                )
                .reshape(-1)
                .tolist()
            )
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            # names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            # info["instance_inds"] = np.array(
            #     [
            #         nusc.getind("instance", x["instance_token"])
            #         for x in annotations
            #     ]
            # )

            info["instance_inds"] = [
                nusc.getind("instance", x["instance_token"]) for x in annotations
            ]



            # get future coords for each box
            # [num_box, fut_ts*2]
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_yaw = np.zeros((num_box, fut_ts))
            gt_fut_masks = np.zeros((num_box, fut_ts))
            gt_boxes_yaw = -(gt_boxes[:, 6] + np.pi / 2)
            # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
            agent_lcf_feat = np.zeros((num_box, 9))
            gt_fut_goal = np.zeros((num_box))
            for i, anno in enumerate(annotations):
                cur_box = boxes[i]
                cur_anno = anno
                agent_lcf_feat[i, 0:2] = cur_box.center[:2]
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                agent_lcf_feat[i, 3:5] = velocity[i]
                agent_lcf_feat[i, 5:8] = anno['size']  # width,length,height
                agent_lcf_feat[i, 8] = cat2idx[anno['category_name']] if anno['category_name'] in cat2idx.keys() else -1
                for j in range(fut_ts):
                    if cur_anno['next'] != '':
                        anno_next = nusc.get('sample_annotation', cur_anno['next'])
                        box_next = Box(
                            anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                        )
                        # Move box to ego vehicle coord system.
                        box_next.translate(-np.array(pose_record['translation']))
                        box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                        #  Move box to sensor coord system.
                        box_next.translate(-np.array(cs_record['translation']))
                        box_next.rotate(Quaternion(cs_record['rotation']).inverse)
                        gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                        gt_fut_masks[i, j] = 1
                        # add yaw diff
                        _, _, box_yaw = quart_to_rpy([cur_box.orientation.x, cur_box.orientation.y,
                                                      cur_box.orientation.z, cur_box.orientation.w])
                        _, _, box_yaw_next = quart_to_rpy([box_next.orientation.x, box_next.orientation.y,
                                                           box_next.orientation.z, box_next.orientation.w])
                        gt_fut_yaw[i, j] = box_yaw_next - box_yaw
                        cur_anno = anno_next
                        cur_box = box_next
                    else:
                        gt_fut_trajs[i, j:] = 0
                        break
                # get agent goal
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                if coord_diff.max() < 1.0:  # static
                    gt_fut_goal[i] = 9
                else:
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

            # get ego history traj (offset format)
            ego_his_trajs = np.zeros((his_ts + 1, 3))
            ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
            sample_cur = sample
            for i in range(his_ts, -1, -1):
                if sample_cur is not None:
                    pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                    ego_his_trajs[i] = pose_mat[:3, 3]
                    has_prev = sample_cur['prev'] != ''
                    has_next = sample_cur['next'] != ''
                    if has_next:
                        sample_next = nusc.get('sample', sample_cur['next'])
                        pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                        ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                        sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
                else:
                    ego_his_trajs[i] = ego_his_trajs[i + 1] - ego_his_trajs_diff[i + 1]
                    ego_his_trajs_diff[i] = ego_his_trajs_diff[i + 1]

            # global to ego at lcf
            ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            # ego to lidar at lcf
            ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]

            # get ego futute traj (offset format)
            ego_fut_trajs = np.zeros((fut_ts + 1, 3))
            ego_fut_masks = np.zeros((fut_ts + 1))
            sample_cur = sample
            for i in range(fut_ts + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[i + 1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            # global to ego at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # drive command according to final fut step offset from lcf
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            # offset from lcf -> per-step offset
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

            ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
            ego_lcf_feat = np.zeros(9)
            # 根据odom推算自车速度及加速度
            _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
            ego_pos = np.array(pose_record['translation'])
            if pose_record_prev is not None:
                _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
                ego_pos_prev = np.array(pose_record_prev['translation'])
            if pose_record_next is not None:
                _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
                ego_pos_next = np.array(pose_record_next['translation'])
            assert (pose_record_prev is not None) or (
                        pose_record_next is not None), 'prev token and next token all empty'
            if pose_record_prev is not None:
                ego_w = (ego_yaw - ego_yaw_prev) / 0.5
                ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi / 2), ego_v * math.sin(ego_yaw + np.pi / 2)
            else:
                ego_w = (ego_yaw_next - ego_yaw) / 0.5
                ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi / 2), ego_v * math.sin(ego_yaw + np.pi / 2)

            ref_scene = nusc.get("scene", sample['scene_token'])
            try:
                pose_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'pose')
                steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
                pose_uts = [msg['utime'] for msg in pose_msgs]
                steer_uts = [msg['utime'] for msg in steer_msgs]
                ref_utime = sample['timestamp']
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                steer_index = locate_message(steer_uts, ref_utime)
                steer_data = steer_msgs[steer_index]
                # initial speed
                v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
                # curvature (positive: turn left)
                steering = steer_data["value"]
                # flip x axis if in left-hand traffic (singapore)
                flip_flag = True if map_location.startswith('singapore') else False
                if flip_flag:
                    steering *= -1
                Kappa = 2 * steering / 2.588
            except:
                delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                v0 = np.sqrt(delta_x ** 2 + delta_y ** 2)
                Kappa = 0

            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])  # can_bus[13:15]
            ego_lcf_feat[2:4] = can_bus[7:9]
            ego_lcf_feat[4] = ego_w  # can_bus[12]
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = v0
            ego_lcf_feat[8] = Kappa






            info["gt_boxes"] = gt_boxes.tolist()
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2).tolist()
            info["num_lidar_pts"] = [a["num_lidar_pts"] for a in annotations]

            info["num_radar_pts"] = [a["num_radar_pts"] for a in annotations]
            info["valid_flag"] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts * 2).astype(np.float32).tolist()
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32).tolist()
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32).tolist()
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32).tolist()
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32).tolist()
            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32).tolist()
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32).tolist()
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32).tolist()
            info['gt_ego_fut_cmd'] = command.astype(np.float32).tolist()
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32).tolist()

        if sample["scene_token"] in train_scenes:
            info.update(scene_token=sample["scene_token"])
            trainscene_nusc_infos[sample["scene_token"]].append(info)
            # train_nusc_infos.append(info)
        else:
            info.update(scene_token=sample["scene_token"])
            valscene_nusc_infos[sample["scene_token"]].append(info)
            # val_nusc_infos.append(info)

    return trainscene_nusc_infos, valscene_nusc_infos


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.
    nusc.scene:
        'token'
        'log_token'
        'nbr_samples'
        'first_sample_token'
        'last_sample_token'
        'name' : str="scene-0061"
        'description'
    nusc.get("sample", token)
        'token'
        'timestamp' : 1532402927647951
        'prev'
        'next'
        'scene_token'
        'data': {} 记录各个传感器当前帧率数据对应的token
                'RADAR_FRONT'
                'RADAR_FRONT_LEFT'
                'RADAR_FRONT_RIGHT'
                'RADAR_BACK_LEFT'
                'RADAR_BACK_RIGHT'
                'LIDAR_TOP'
                'CAM_FRONT'
                'CAM_FRONT_RIGHT'
                'CAM_BACK_RIGHT'
                'CAM_BACK'
                'CAM_BACK_LEFT'
                'CAM_FRONT_LEFT'
        'anns'
    nusc.get("sample_data", token): 存储传感器关键帧内外参和pose, 同时记录前一帧和后一帧非关键帧的token
                                    nusc.get_sample_data(cam_token)获取标注数据
        'token': 记录当前传感器关键帧标注数据对应的token
        'sample_token'
        'ego_pose_token'
        'calibrated_sensor_token'
        'timestamp': 关键帧时间戳
        'fileformat'
        'is_key_frame'
        'height'
        'width'
        'filename'
        'prev'
        'next'
        'sensor_modality'
        'channel'
    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)  # 这里和scene内容保持一致
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not isinstance(lidar_path, str):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            print("[WARNING] cur scene not exist...")
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    roi_size = (30, 60)
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    nusc_can_bus = NuScenesCanBus(dataroot=root_path)

    # filter existing scenes.根据lidartop的数据路径来判断
    available_scenes = get_available_scenes(nusc)  # list
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )

    # 解析nuscenes原始数据
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_map_extractor, nusc_can_bus, list(train_scenes), list(val_scenes), test, max_sweeps=max_sweeps
    )

    # metadata = dict(version=version)
    if test:
        total_test_sample = 0
        for k, v in train_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "test", k + ".json")
            total_test_sample += len(v)
            json.dump(v, open(cache_path, "w"), indent=True, ensure_ascii=False)
        print(
            "test sample: {}, val sample: {}".format(
                total_train_sample, total_val_sample
            )
        )
    else:
        total_train_sample = 0
        for k, v in train_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "train", k + ".json")
            total_train_sample += len(v)
            json.dump(v, open(cache_path, "w"), indent=True, ensure_ascii=False)

        total_val_sample = 0
        for k, v in val_nusc_infos.items():
            cache_path = os.path.join(info_prefix, "val", k + ".json")
            total_val_sample += len(v)
            json.dump(v, open(cache_path, "w"), indent=True, ensure_ascii=False)

        print(
            "train sample: {}, val sample: {}".format(
                total_train_sample, total_val_sample
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument("--root_path", type=str, default="./data/nuscenes")
    parser.add_argument("--info_prefix", type=str, default="./data/nusc_anno_dumpjson")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--max_sweeps", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.info_prefix + "/train", exist_ok=True)
    os.makedirs(args.info_prefix + "/val", exist_ok=True)
    os.makedirs(args.info_prefix + "/test", exist_ok=True)

    versions = args.version.split(",")
    for version in versions:
        create_nuscenes_infos(
            args.root_path,
            args.info_prefix,
            version=version,
            max_sweeps=args.max_sweeps,
        )
