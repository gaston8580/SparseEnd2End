import os
import mmcv
import tempfile
from utils import *
import cv2
import numpy as np
#import scipy.spatial.transform.Rotation as R
from scipy.spatial.transform import Rotation as R
import random
from misc import *
import torch

PALETTE_COLOR = [(15, 255, 107), (255, 60, 25), (255, 36, 229), (18, 5, 255), (31, 169, 255)]


X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners

def draw_3d_box(img, boxes3d, labels=None, color=(0, 255, 0), intrinsic=None, extrinsic=np.eye(4), dist=None, fov=None,
                thickness=2, draw_2d=False, return_box2d=False):
    '''
    绘制3d box

    Args:
      - img: image
      - boxes3d(np.ndarray|np.ndarray): nx7, 3d box(x,y,z,l,w,h,yaw)
      - intrinsic(np.ndarray): 3x3
      - extrinsic(np.ndarray): 4x4, xxx -> camera
      - dist(np.ndarray): distortion
      - fov: camera fov
      - draw_2d: 是否绘制3d框的外接2d框
    Returns:
      -
    '''
    assert (intrinsic is not None)
    img_draw = np.copy(img)
    h, w = img.shape[:2]
    # 8个顶点
    n = len(boxes3d)
    corners3d = box3d_to_corners3d(boxes3d)  # (n, 8, 3)
    corners3d_homo = to_homo(corners3d)  # (n, 8, 4)
    cam_corners3d_homo = corners3d_homo @ (extrinsic.T)
    mask = (cam_corners3d_homo[..., 2] > 0)  # (n, 8)
    cam_corners3d = cam_corners3d_homo[..., :3]  # (n, 8, 3)

    if dist is None:
        corners2d_homo = cam_corners3d @ (intrinsic.T)
        corners2d = corners2d_homo[..., :2] / corners2d_homo[..., 2:3]  # (n, 8, 2)
    else:
        n, m, c = corners3d.shape  # (n, 8, 3)
        rvec, _ = cv2.Rodrigues(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3]
        if len(dist) == 4:  # k1, k2, k3, k4 -> 鱼眼畸变
            corners2d, _ = cv2.fisheye.projectPoints(corners3d.reshape(-1, 1, 3), rvec, tvec, intrinsic, dist)
        else:  # 针孔畸变
            corners2d, _ = cv2.projectPoints(corners3d.reshape(-1, 1, 3), rvec, tvec, intrinsic, dist)
        corners2d = corners2d.reshape(n, m, -1)

    if return_box2d:
        box2d_list = [None for _ in range(n)]

    for i, corner2d in enumerate(corners2d):
        if (not draw_2d) and mask[i].sum() < 8:  # z > 0 的点少于4个
            continue
        corner2d = corner2d.astype(np.int32)  # (8, 2)

        if draw_2d:
            dense_corner3d = box3d_to_dense_corners3d(boxes3d[i:i + 1], step=0.1, on_surface=True)[0]  # (k, 3)
            dense_corner3d_homo = to_homo(dense_corner3d)  # (k, 4)
            dense_cam_corner3d_homo = dense_corner3d_homo @ (extrinsic.T)
            dense_mask = (dense_cam_corner3d_homo[..., 2] > 0)
            if fov is not None:
                _, fov_mask = filter_fov_points(dense_cam_corner3d_homo[..., :3], fov)
                dense_mask &= fov_mask

            dense_corner3d = dense_corner3d[dense_mask & fov_mask][..., :3]
            dense_cam_corner3d = dense_cam_corner3d_homo[dense_mask & fov_mask][..., :3]
            if len(dense_corner3d) == 0:
                continue

            if dist is None:
                dense_corner2d_homo = dense_cam_corner3d[:, :3] @ (intrinsic.T)
                dense_corner2d = dense_corner2d_homo[..., :2] / dense_corner2d_homo[..., 2:3]  # (k, 2)
            else:
                k, c = dense_corner3d.shape  # (k, 3)
                if len(dist) == 4:  # k1, k2, k3, k4 -> 鱼眼畸变
                    dense_corner2d, _ = cv2.fisheye.projectPoints(dense_corner3d.reshape(-1, 1, 3), rvec, tvec,
                                                                  intrinsic, dist)
                else:  # 针孔畸变
                    dense_corner2d, _ = cv2.projectPoints(dense_corner3d.reshape(-1, 1, 3), rvec, tvec, intrinsic, dist)
                dense_corner2d = dense_corner2d.reshape(k, 2).astype(np.int32)
            xmin = int(dense_corner2d[:, 0].min())
            ymin = int(dense_corner2d[:, 1].min())
            xmax = int(dense_corner2d[:, 0].max())
            ymax = int(dense_corner2d[:, 1].max())
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w - 1, xmax)
            ymax = min(h - 1, ymax)
            box2d_list[i] = [xmin, ymin, xmax, ymax]
            c = random.choice(PALETTE_COLOR)
            cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color=c, thickness=thickness)
            if labels is not None:
                cv2.putText(img_draw, str(labels[i]), (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 1, c,
                            thickness=thickness)
        else:
            # up & down plane
            cv2.polylines(img_draw, [corner2d[:4].reshape(-1, 1, 2)], True, color, thickness)
            cv2.polylines(img_draw, [corner2d[4:].reshape(-1, 1, 2)], True, color, thickness)

            # left & right plane
            cv2.polylines(img_draw, [corner2d[[0, 4, 5, 1]].reshape(-1, 1, 2)], True, color, thickness)
            cv2.polylines(img_draw, [corner2d[[3, 7, 6, 2]].reshape(-1, 1, 2)], True, color, thickness)

            # front
            color2 = (200, 60, 0)
            # cv2.fillPoly(img_draw, [corner2d[[0, 4, 7, 3]].reshape(-1, 1, 2)], color2)
            cv2.polylines(img_draw, [corner2d[[0, 4, 7, 3]].reshape(-1, 1, 2)], True, color, thickness)

            # back
            cv2.polylines(img_draw, [corner2d[[1, 5, 6, 2]].reshape(-1, 1, 2)], True, color, thickness)

            # direction cross
            color2 = (255, 255, 0)
            cv2.line(img_draw, tuple(corner2d[0]), tuple(corner2d[7]), color2, thickness)
            cv2.line(img_draw, tuple(corner2d[3]), tuple(corner2d[4]), color2, thickness)
            if labels is not None:
                cv2.putText(img_draw, str(labels[i]), tuple(corner2d[0] - [40, 10]), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), thickness=thickness)

    if return_box2d:
        return img_draw, box2d_list
    return img_draw


if __name__ == '__main__':

    clips_path = '/home/ma-user/work/data/ali_odd'
    for clip in os.listdir(clips_path):

        if clip!= 'clip_1730513152300':
            continue
        os.makedirs(os.path.join(clips_path, clip, 'vis_3d_lane_resize'), exist_ok=True)
        ##########Get the lane#########
        maps = mmcv.load(os.path.join(clips_path, 'label3d_line_lidarCS', 'LNNACDDV5PDA30339', clip, 'local_map',
                                    'line_3d_lidarCS.json'))
        ######Get MOD Type####
        mod_annos = mmcv.load(os.path.join(clips_path, clip+'.json'))

        #######Get Extrinsics and Intrinsics###
        info_json = mmcv.load(os.path.join(clips_path, clip, 'info.json'))
        plate_no = info_json['plate_no']
        with tempfile.TemporaryDirectory() as temp:
            exit_code = os.system(
                f'python /home/ma-user/work/calibration/load_calibration.py -v {plate_no} -p {clip} -c /home/ma-user/work/calibration/ -s {str(temp)}')
            if exit_code != 0:
                raise RuntimeError(f'{clip} get calibration failed !!!')
            ins_ex_path = str(temp)
            lidar2imu_path = os.path.join(ins_ex_path, 'extrinsics/lidar2imu/lidar2imu.yaml')
            with open(lidar2imu_path) as f:
                data_imu = yaml.safe_load(f)
            ins_ex = load_ins_ex(ins_ex_path)
        #####################
        sample_list = []
        for file in os.listdir(os.path.join(clips_path, clip)):
            if file.startswith('sample_'):
                sample_list.append(file)

        ##################
        for sample in sample_list:
            os.makedirs(os.path.join(clips_path, clip, 'vis_3d_bbox', sample), exist_ok=True)

            for g in mod_annos['frames']:
                if g['frame_name'] == sample:
                    g_tmp = g
            objs = g_tmp['annotated_info']['3d_city_object_detection_annotated_info']['annotated_info'][
                '3d_object_detection_info']['3d_object_detection_anns_info']

            boxes3d = np.array([[*obj['obj_center_pos'],
                                *obj['size'],
                                R.from_quat(obj['obj_rotation']).as_euler('zyx')[0]]
                                for obj in objs])
            labels = []
            for obj in objs:
                # if obj["category"] in ["unknown", "barrier", "traffic_warning"]:
                #  continue
                labels.append(f"{obj['track_id']}")

            labels = np.array(labels)

            for sensor_file in os.listdir(os.path.join(clips_path, clip, sample)):
                if sensor_file.startswith('camera0') and sensor_file.find('undist') != -1:
                        cam_type = 'camera0'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _ = draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)

                if sensor_file.startswith('camera1') and sensor_file.find('undist') != -1 and sensor_file.startswith('camera10')==False:
                        cam_type = 'camera1'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)
                if sensor_file.startswith('camera2') and sensor_file.find('undist') != -1:
                        cam_type = 'camera2'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)


                if sensor_file.startswith('camera3') and sensor_file.find('undist') != -1:
                        cam_type = 'camera3'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)

                if sensor_file.startswith('camera4') and sensor_file.find('undist') != -1:
                        cam_type = 'camera4'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)

                if sensor_file.startswith('camera5') and sensor_file.find('undist') != -1:
                        cam_type = 'camera5'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)

                if sensor_file.startswith('camera6') and sensor_file.find('undist') != -1:
                        cam_type = 'camera6'
                        img = cv2.imread(os.path.join(clips_path, clip, sample, sensor_file))
                        img_resize =  cv2.resize(img, dsize=(640, 480))
                        #img[int(1080-640):int(1080 + 640), int(1920 - 960):int(1920 + 960)]
                        ###The extrinsics are the same###
                        intrinsics = ins_ex[cam_type]['K']
                        scale = np.eye(3)
                        scale[0][0] = float(640 / img.shape[1])
                        scale[1][1] = float(480 / img.shape[0])
                        dist = None
                        fov =  60/180*np.pi
                        img_w_3d_boxes, _= draw_3d_box(img_resize, boxes3d, labels, intrinsic= scale@intrinsics, extrinsic=ins_ex[cam_type]['R_T'],
                                                    dist=dist, fov=fov, thickness=2, draw_2d=False, return_box2d=True)
                        cv2.imwrite(os.path.join(clips_path, clip, 'vis_3d_bbox', sample, sensor_file), img_w_3d_boxes)