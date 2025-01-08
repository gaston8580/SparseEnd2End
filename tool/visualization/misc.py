# -*- coding: utf-8 -*-
import os
import json
import yaml
import numpy as np
import cv2
import bisect
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


def to_homo(pts):
    """转化为齐次坐标"""
    shape = pts.shape
    homo_pts = np.concatenate([pts, np.ones([*shape[:-1], 1])], axis=-1)
    return homo_pts


def filter_fov_points(pts, fov):
    """过滤camera fov范围以外的
    Notes: pts要是camera坐标系下的点
    Args:
      - pts: np.ndarray, shape=(n, c), camera坐标系下的点, x,y,z,...
      - fov: float, 水平fov

    Returns:
      - new_pts: 过滤之后的点云
    """
    pts_fov = np.abs(np.arctan2(pts[:, 0], pts[:, 2]))
    pts_mask = pts_fov <= fov / 2
    return pts[pts_mask], pts_mask


def cartesian2spherical(pts):
    """
    笛卡尔坐标系 -> 球面坐标系
    x=rsinθcosφ.
    y=rsinθsinφ.
    z=rcosθ.
    Args:
      - pts: np.ndarray, shape=(..., c)

    Returns:
      - pts: np.ndarray, shape=(..., c)
    """
    # 转换到球面坐标系
    spherical_pts = np.zeros_like(pts)  # r, θ, φ.
    r = np.linalg.norm(pts[..., :3], axis=-1)
    theta = np.arccos(pts[..., 2] / r)  # [0, pi]
    fi = np.arctan2(pts[..., 1], pts[..., 0])  # [-pi, pi]
    fi = np.where(fi < 0, fi + 2 * np.pi, fi)  # [0, 2*np.pi]
    spherical_pts[..., 0] = r
    spherical_pts[..., 1] = theta
    spherical_pts[..., 2] = fi
    spherical_pts[..., 3:] = pts[..., 3:]
    return spherical_pts


def spherical2cartesian(pts):
    """
    球面坐标系 -> 笛卡尔坐标系
    x=rsinθcosφ.
    y=rsinθsinφ.
    z=rcosθ.
    Args:
      - pts: np.ndarray, shape=(..., c)

    Returns:
      - pts: np.ndarray, shape=(..., c)
    """
    # 转换到球面坐标系
    cartesian_pts = np.zeros_like(pts)  # r, θ, φ.
    r = pts[..., 0]
    theta = pts[..., 1]
    fi = pts[..., 2]
    x = r * np.sin(theta) * np.cos(fi)
    y = r * np.sin(theta) * np.sin(fi)
    z = r * np.cos(theta)
    cartesian_pts[..., 0] = x
    cartesian_pts[..., 1] = y
    cartesian_pts[..., 2] = z
    cartesian_pts[..., 3:] = pts[..., 3:]
    return cartesian_pts


def farthest_point_sample(xyz, npoint):
    """最远点采样 copy from pointnet
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        #centroids: sampled pointcloud index, [B, npoint]
        new_xyz: 新的采样点 [B, npoint, 3]
    """

    # xyz = xyz.transpose(0,2,1)
    B, N, C = xyz.shape

    centroids = np.zeros((B, npoint), dtype=np.int32)  # 采样点矩阵（B, npoint）
    distance = np.ones((B, N)) * 1e10  # 采样点到所有点距离（B, N）

    batch_indices = np.arange(B)  # batch_size 数组

    barycenter = np.sum((xyz), 1)  # 计算重心坐标 及 距离重心最远的点
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, C)  # numpy中的reshape相当于torch中的view

    dist = np.sum((xyz - barycenter) ** 2, -1)
    farthest = np.argmax(
        dist, 1
    )  # 将距离重心最远的点作为第一个点，这里跟torch.max不一样

    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].reshape(
            B, 1, C
        )  # 取出这个最远点的xyz坐标
        dist = np.sum(
            (xyz - centroid) ** 2, -1
        )  # 计算点集中的所有点到这个最远点的欧式距离，-1消掉了xyz那个维度
        # print("dist    : ", dist)
        mask = dist < distance
        # print("mask %i : %s" % (i,mask))
        distance[mask] = dist[
            mask
        ]  # 更新distance，记录样本中每个点距离所有已出现的采样点（已采样集合中的点）的最小距离
        # print("distance: ", distance)

        farthest = np.argmax(distance, -1)  # 返回最远点索引
    # return centroids
    batch_indices = batch_indices.repeat(centroids.shape[1]).reshape(
        (-1, centroids.shape[1])
    )  # (batch_size,npoints)
    new_xyz = xyz[batch_indices, centroids, :]
    return new_xyz


def get_yaw(quat):
    """
    从四元数转成yaw角
    Args:
      - quat: np.ndarray|list of 4, x, y, z, w
    Returns:
      - yaw:
    """
    r = R.from_quat(quat)
    zyx = r.as_euler("zyx")
    return zyx[0]


def yaw_to_quaternion(yaw, degrees=False):
    """
    yaw角转化为四元数
    Args:
      - yaw:
    Returns:
      - Quaternion: w, x, y, z
    """
    if degrees:
        yaw = yaw / 180.0 * np.pi
    r = R.from_euler("zyx", [yaw, 0, 0]).as_quat()  # x, y, z, w
    q = Quaternion(r[[3, 0, 1, 2]])  # w, x, y, z
    return q


def box3d_to_corners3d(boxes3d: np.ndarray):
    """
    将3dbox (x, y, z, l, w, h, yaw) 转还为8个顶点的3d坐标
    bev视角下:
              x                          z   x
    (4)0------------3(7)                 |   /
       |      |     |                    |  /
       |      |     |                    | /
      y|-------     |           y-----------------
       |            |                   /|
    (5)1------------2(6)               / |

    Args:
      - boxes3d(np.ndarray): shape=(n, 7), x, y, z, l, w, h, yaw
    Returns:
      - corners3d(np.ndarray): shape=(n, 8, 3)
    """
    if len(boxes3d) == 0:
        return np.zeros(shape=(0, 8, 3), dtype=boxes3d.dtype)
    xs, ys, zs, ls, ws, hs, yaws = boxes3d.T  # (n, )
    corners3d = np.array(
        [
            [ls / 2, ws / 2, hs / 2],  # p0
            [-ls / 2, ws / 2, hs / 2],  # p1
            [-ls / 2, -ws / 2, hs / 2],  # p2
            [ls / 2, -ws / 2, hs / 2],  # p3
            [ls / 2, ws / 2, -hs / 2],  # p4
            [-ls / 2, ws / 2, -hs / 2],  # p5
            [-ls / 2, -ws / 2, -hs / 2],  # p6
            [ls / 2, -ws / 2, -hs / 2],  # p7
        ],
        dtype=np.float32,
    )  # (8, 3, n)
    corners3d = corners3d.transpose(2, 0, 1)  # (n, 8, 3)

    zeros = np.zeros_like(yaws)
    ones = np.ones_like(yaws)
    rot_mat = np.array(
        [
            [np.cos(yaws), -np.sin(yaws), zeros],
            [np.sin(yaws), np.cos(yaws), zeros],
            [zeros, zeros, ones],
        ],
        dtype=np.float32,
    )  # (3, 3, n)
    t = np.stack([xs, ys, zs], axis=1)  # (n, 3)

    rot_mat = rot_mat.transpose(2, 0, 1)  # (n, 3, 3)
    corners3d = corners3d @ (rot_mat.transpose(0, 2, 1))  # (n, 8, 3)
    corners3d = corners3d + np.expand_dims(t, 1)  # (n, 8, 3)
    return corners3d


def box3d_to_dense_corners3d(boxes3d: np.ndarray, step=0.1, on_surface=True):
    """
    box3d_to_corners3d改进版 每维度按照step等距离密集采样
    bev视角下:
              x                          z   x
    (4)0------------3(7)                 |   /
       |      |     |                    |  /
       |      |     |                    | /
      y|-------     |           y-----------------
       |            |                   /|
    (5)1------------2(6)               / |

    Args:
      - boxes3d(np.ndarray): shape=(n, 7), x, y, z, l, w, h, yaw
      - step(float): 步长,
      - on_surface(bool): 是否只在3d框表面上采样点, default: True
    Returns:
      - corners3d_list(List[np.ndarray]): shape is [(m1, 3), (m2, 3), ... (mn, 3)]
    """
    if len(boxes3d) == 0:
        return np.zeros(shape=(0, 0, 3), dtype=boxes3d.dtype)
    xs, ys, zs, ls, ws, hs, yaws = boxes3d.T  # (n, )
    corners3d_list = []
    for x, y, z, l, w, h, yaw in zip(xs, ys, zs, ls, ws, hs, yaws):
        if on_surface:
            # 4个侧面(正常情况)
            xxx1, yyy1, zzz1 = np.meshgrid(
                np.array([-(l + step) / 2, (l + step) / 2]),
                np.arange(-(w + step) / 2, (w + step) / 2, step),
                np.arange(-(h + step) / 2, (h + step) / 2, step),
            )
            corners3d1 = np.stack([xxx1, yyy1, zzz1], axis=-1).reshape(-1, 3)
            xxx2, yyy2, zzz2 = np.meshgrid(
                np.arange(-(l + step) / 2, (l + step) / 2, step),
                np.array([-(w + step) / 2, (w + step) / 2]),
                np.arange(-(h + step) / 2, (h + step) / 2, step),
            )
            corners3d2 = np.stack([xxx2, yyy2, zzz2], axis=-1).reshape(-1, 3)
            corners3d = np.concatenate([corners3d1, corners3d2], axis=0)
        else:
            xxx, yyy, zzz = np.meshgrid(
                np.arange(-(l + step) / 2, (l + step) / 2, step),
                np.arange(-(w + step) / 2, (w + step) / 2, step),
                np.arange(-(h + step) / 2, (h + step) / 2, step),
            )
            corners3d = np.stack([xxx, yyy, zzz], axis=-1).reshape(-1, 3)  # (m, 3)
        rot_mat = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )  # (3, 3)
        tran = np.array([x, y, z])  # (3)

        corners3d = corners3d @ rot_mat.T + tran
        corners3d_list.append(corners3d)
    return corners3d_list


def points_in_3dboxes(points: np.ndarray, boxes3d: np.ndarray):
    """获取在box3d中的所有点
    定义3dbox的6个面的法向量朝外, 通过向量内积正负号判断
    bev视角下
              x                          z   x
    (4)0------------3(7)                 |   /
       |      |     |                    |  /
       |      |     |                    | /
      y|-------     |           y-----------------
       |            |                   /|
    (5)1------------2(6)               / |
    Args:
      - points(np.ndarray): shape=(n, 3)
      - boxes3d(np.ndarray): shape=(m, 7)
    """
    n = points.shape[0]
    m = boxes3d.shape[0]
    corners3d = box3d_to_corners3d(boxes3d)  # (m, 8, 3)
    # 上下左右前后6个面定义为plane{0~5}, 法向量定义为normal_vector{0~5}, 方向朝外
    normal_vector0 = np.cross(
        corners3d[:, 1] - corners3d[:, 0], corners3d[:, 3] - corners3d[:, 0]
    )
    normal_vector1 = np.cross(
        corners3d[:, 7] - corners3d[:, 4], corners3d[:, 5] - corners3d[:, 4]
    )
    normal_vector2 = np.cross(
        corners3d[:, 0] - corners3d[:, 1], corners3d[:, 5] - corners3d[:, 1]
    )
    normal_vector3 = np.cross(
        corners3d[:, 6] - corners3d[:, 2], corners3d[:, 3] - corners3d[:, 2]
    )
    normal_vector4 = np.cross(
        corners3d[:, 1] - corners3d[:, 2], corners3d[:, 6] - corners3d[:, 2]
    )
    normal_vector5 = np.cross(
        corners3d[:, 7] - corners3d[:, 3], corners3d[:, 0] - corners3d[:, 3]
    )

    normal_vector = np.stack(
        [
            normal_vector0,
            normal_vector2,
            normal_vector5,
            normal_vector1,
            normal_vector3,
            normal_vector4,
        ],
        axis=1,
    )  # (m, 6, 3) 每个3d框每个面的法向量
    vec1 = (
        points[None, ...] - corners3d[:, None, 0]
    )  # 取plane{0, 2, 5}上的点0构成向量 (m, n, 3)
    vec2 = points[None, ...] - corners3d[:, None, 6]  # 取plane{1, 3, 4}上的点6构成向量
    points_vector = np.stack(
        [vec1, vec1, vec1, vec2, vec2, vec2], axis=1
    )  # (m, 6, n, 3)

    # dot_product: (m, 6, n, 1)
    dot_product = np.matmul(points_vector, normal_vector[..., None])
    t = np.all(dot_product[..., 0] < 0, axis=1)  # (m, n)
    result = []
    for i in range(m):
        # result.append(points[t[i]>0]) # 具体的点坐标
        result.append((t[i] > 0))  # bool 索引, 每个点是否在该box内
    return result


def get_pose_mat(poses: list, poses_timestamp: list, timestamp: float):
    """
    获取特定时间戳的pose矩阵, 若有必要需要进行插值
    Args:
      - poses: list[dict] 自车poses
      - poses_timestamp: pose的时间戳
      - timestamp: float, 要获取的时间戳
    Returns:
      - pose_mat: np.ndarray, shape=(4, 4)
    """

    idx = bisect.bisect(poses_timestamp, timestamp)
    if idx > 0 and idx < len(poses_timestamp):
        pose0 = poses[idx - 1]
        pose1 = poses[idx]

        q, t = pose_slerp(pose0, pose1, timestamp)
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = q.rotation_matrix
        pose_mat[:3, 3] = t
    else:
        pose = poses[idx]
        pose_mat = pose2matrix(pose)
    return pose_mat


def pose_slerp(pose0: dict, pose1: dict, timestamp: float):
    """对pose进行插值

    Args:
      - pose0: 起始pose
      - pose1: 结束pose
      - timestamp: 中间时刻
    Returns:
      - q: 插值后的四元数
      - q: 插值后的平移量
    """
    q0 = Quaternion(
        [
            pose0["pose"]["orientation"]["qw"],
            pose0["pose"]["orientation"]["qx"],
            pose0["pose"]["orientation"]["qy"],
            pose0["pose"]["orientation"]["qz"],
        ]
    )
    t0 = np.array(
        [
            pose0["pose"]["position"]["x"],
            pose0["pose"]["position"]["y"],
            pose0["pose"]["position"]["z"],
        ]
    )
    timestamp0 = pose0["timestamp"]
    q1 = Quaternion(
        [
            pose1["pose"]["orientation"]["qw"],
            pose1["pose"]["orientation"]["qx"],
            pose1["pose"]["orientation"]["qy"],
            pose1["pose"]["orientation"]["qz"],
        ]
    )
    t1 = np.array(
        [
            pose1["pose"]["position"]["x"],
            pose1["pose"]["position"]["y"],
            pose1["pose"]["position"]["z"],
        ]
    )
    timestamp1 = pose1["timestamp"]

    amount = (timestamp - timestamp0) / (timestamp1 - timestamp0 + 1e-7)
    t = t0 + amount * (t1 - t0)
    q = Quaternion.slerp(q0, q1, amount)
    return q, t


def pose2matrix(pose):
    """
    把pose信息变成np.ndarray=4x4的矩阵 imu -> utm
    """
    m = np.eye(4)
    quat = R.from_quat(
        [
            pose["pose"]["orientation"]["qx"],
            pose["pose"]["orientation"]["qy"],
            pose["pose"]["orientation"]["qz"],
            pose["pose"]["orientation"]["qw"],
        ]
    )
    m[:3, :3] = quat.as_matrix()
    m[:3, 3] = [
        pose["pose"]["position"]["x"],
        pose["pose"]["position"]["y"],
        pose["pose"]["position"]["z"],
    ]
    return m


def box3d_interp(box0: np.ndarray, box1: np.ndarray, t0: float, t1: float, t: float):
    """对box进行插值

    Args:
      - box0: np.ndarray|list, (10,) [xc, yc, zc, l, w, h, qx, qy, qz, qw]
      - box1: np.ndarray|list, (10,) [xc, yc, zc, l, w, h, qx, qy, qz, qw]
      - t0: box0的时间戳
      - t1: box1的时间戳
      - t: 当前的时间戳
    Returns:
      - box: 插值后的box [xc, yc, zc, l, w, h, qx, qy, qz, qw]
    """
    quat0 = Quaternion([box0[-1], box0[-4], box0[-3], box0[-2]])
    tran0 = np.array(box0[:3])
    quat1 = Quaternion([box1[-1], box1[-4], box1[-3], box1[-2]])
    tran1 = np.array(box1[:3])

    amount = (t - t0) / (t1 - t0)
    tran = tran0 + amount * (tran1 - tran0)
    quat = Quaternion.slerp(quat0, quat1, amount)
    box = np.array([*tran, *box0[3:6], quat.x, quat.y, quat.z, quat.w])
    return box


def matrix_interp(m0: np.ndarray, m1: np.ndarray, t0: float, t1: float, t: float):
    """对坐标变换矩阵进行插值

    Args:
      - m0: np.ndarray, 4x4
      - m1: np.ndarray, 4x4
      - t0: box0的时间戳
      - t1: box1的时间戳
      - t: 当前的时间戳
    Returns:
      - m: 插值后的变换矩阵, 4x4
    """
    rot_mat0 = m0[:3, :3]
    tran0 = m0[:3, 3]
    quat0 = Quaternion(matrix=rot_mat0)

    rot_mat1 = m1[:3, :3]
    tran1 = m1[:3, 3]
    quat1 = Quaternion(matrix=rot_mat1)

    amount = (t - t0) / (t1 - t0)
    tran = tran0 + amount * (tran1 - tran0)
    quat = Quaternion.slerp(quat0, quat1, amount)
    m = np.eye(4)
    m[:3, :3] = quat.rotation_matrix
    m[:3, 3] = tran
    return m


def iou(boxes0: np.ndarray, boxes1: np.ndarray, return_iou=True):
    """计算多个边界框和多个边界框的交并比

    Parameters
    ----------
    boxes0: `~np.ndarray` of shape `(A, 4)`
        边界框

    boxes1: `~np.ndarray` of shape `(B, 4)`
        边界框

    return_iou: 返回IoU if True, 否则返回inter_area/B
    Returns
    -------
    iou: `~np.ndarray` of shape `(A, B)`
        交并比
    """
    A = boxes0.shape[0]
    B = boxes1.shape[0]

    xy_max = np.minimum(
        boxes0[:, np.newaxis, 2:].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, 2:], (A, B, 2)),
    )
    xy_min = np.maximum(
        boxes0[:, np.newaxis, :2].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, :2], (A, B, 2)),
    )

    # 计算交集面积
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, :, 0] * inter[:, :, 1]

    # 计算每个矩阵的面积
    area_0 = ((boxes0[:, 2] - boxes0[:, 0]) * (boxes0[:, 3] - boxes0[:, 1]))[
        :, np.newaxis
    ].repeat(B, axis=1)
    area_1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))[
        np.newaxis, :
    ].repeat(A, axis=0)
    if return_iou:
        return inter / (area_0 + area_1 - inter)
    else:
        return inter / (area_1)
