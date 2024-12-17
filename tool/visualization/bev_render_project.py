import os
import numpy as np
import cv2
import json
import torch
import matplotlib
import matplotlib.pyplot as plt
import pyquaternion
import copy
from shapely.geometry import LineString

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

EGO_BBOX = [[0,
    0,
    0,
    5,
    2,
    2,
    0]]

def project_points(key_points, lidar2img, image_wh=None):
    """
    Args:
        key_points : Shape[1, 1220, 13, 3].
    Return:
        key_points2d : Shape[1, 6, 1220, 13, 2].


    """
    bs, num_anchor, num_pts = key_points.shape[:3]
    print("*************************")
    print(key_points.shape, lidar2img.shape)
    pts_extend = torch.cat(
        [key_points, torch.ones_like(key_points[..., :1])], dim=-1
    )
    # points_2d = torch.matmul(
    #     lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
    # ).squeeze(-1)
    points_2d = torch.matmul(
        lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
    )[..., 0]
    points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
    print("*************************")
    # torch.Size([1, 1, 8, 3]) torch.Size([1, 6, 4, 4]) torch.Size([1, 6, 1, 8, 2])
    print(key_points.shape, lidar2img.shape, points_2d.shape)
    if image_wh is not None:
        points_2d = points_2d / image_wh[:, :, None, None]
    return points_2d

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


CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.02
MAP_SCORE_THRESH = 0.3
color_mapping = np.asarray([
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255


class BEVRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
        xlim = 50,
        ylim = 50,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        self.axes.set_xlim(- self.xlim, self.xlim)
        self.axes.set_ylim(- self.ylim, self.ylim)
        self.axes.axis('off')

    def render(
        self,
        data, 
        result,
        index,
        with_infer = True,
    ):
        self.reset_canvas()
        self.draw_detection_gt(data)
        # self.draw_motion_gt(data)
        self.draw_map_gt(data)
        self.draw_planning_gt(data)
        self._render_command(data)
        self._render_gt_title()
        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)

        if with_infer:
            # origin_path_pre = '/home/ma-user/work/data/ali_odd'
            # local_path_pre = '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd'
            # img_path = data['cams']['CAM_FRONT_WIDE']['data_path']
            # origin_path_pre = './data/nuscenes'
            # local_path_pre = '/home/chengjiafeng/work/data/nuscene/nuscenes'
            origin_path_pre = './data/nuscenes/samples'
            local_path_pre = '/home/chengjiafeng/work/data/nuscene/nuscenes_8clips_cam'
            img_path = data['cams']['CAM_FRONT']['data_path']
            img_path = img_path.replace(origin_path_pre, local_path_pre)
            # save_result_path_prefix = img_path.split("/sample")[0]
            # save_result_path_suffix = img_path.split("/camera0")[1].split(".jpg")[0]
            save_result_path_prefix = img_path.split("/CAM_FRONT/")[0]
            save_result_path_suffix = img_path.split("/CAM_FRONT/")[1].split(".jpg")[0]
            save_result_path_dir = os.path.join(save_result_path_prefix, "samples_results_nuscene_8clips")
            save_result_path = os.path.join(save_result_path_dir, save_result_path_suffix + ".json")
            result = json.load(open(save_result_path, "r"))
            
            self.reset_canvas()
            # self.draw_detection_pred(result)
            lidar2img = self.get_data_info(data)
            self.draw_detection_project(result, data, lidar2img)
            # self.draw_track_pred(result)
            # self.draw_motion_pred(result)
            # self.draw_map_pred(result)
            # self.draw_planning_pred(data, result)
            self._render_command(data)
            self._render_pred_title()
        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_pred)

        return save_path_gt, save_path_pred

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det']:
            return

        # for i in range(data['gt_labels_3d'].shape[0]):
        #     label = data['gt_labels_3d'][i]
        for i in range(len(data['gt_boxes'])):
            # label = data['gt_boxes'][i]
            # if label == -1: 
            #     continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(np.array(data['gt_boxes']))[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = np.array(result['boxes_3d'])
        for i in range(len(result['labels_3d'])):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[i % len(color_mapping)]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_detection_project(self, result, data, lidar2img):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bbox3d = np.array(data["gt_boxes"])
        bbox3d = box3d_to_corners(np.array(data['gt_boxes']))
        bbox3d = torch.tensor(bbox3d).unsqueeze(0).double()
        bboxes_2d = project_points(bbox3d, torch.tensor(lidar2img).unsqueeze(0).double())

        bboxes_2d = bboxes_2d.permute(0, 2, 3, 1, 4)[:, :, :, 0, :].squeeze(0)
        print("$$$$$$$$$$$$$$$$$$")
        print(bboxes_2d)
        for i in range(bboxes_2d.shape[0]):
            color = color_mapping[i % len(color_mapping)]

            x = bboxes_2d[i, :, 0]
            y = bboxes_2d[i, :, 1]

            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(np.array(anno))
                map_geoms[label].append(geom)
        return map_geoms

    def get_data_info(self, data):
        """format data dict fed to pipeline.
        img_filename: List[str] length=6(v)
        """

        info = data

        input_dict = dict(
            sample_scene=info["scene_token"],
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,  # 单位为秒
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],

            #prev_idx=info['prev'],
            #next_idx=info['next'],
            #can_bus= np.array(info['can_bus']),
            #frame_idx=info['frame_idx'],
            map_annos = info['map_annos'],
            fut_valid_flag=info['fut_valid_flag'],
            #map_location=info['map_location'],
            ego_his_trajs=np.array(info['gt_ego_his_trajs']),
            ego_fut_trajs=np.array(info['gt_ego_fut_trajs']),
            ego_fut_masks= np.array(info['gt_ego_fut_masks']),
            ego_fut_cmd=np.array(info['gt_ego_fut_cmd']),
            ego_lcf_feat= np.array(info['gt_ego_lcf_feat'])
        )

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])

        input_dict["lidar2global"] = ego2global @ lidar2ego

        map_geoms = self.anno2geom(info["map_annos"])
        input_dict["map_geoms"] = map_geoms

        image_paths = []
        lidar2img_rts = []
        cam_intrinsic = []
        for cam_type, cam_info in info["cams"].items():
            data_path = cam_info["data_path"].replace('./data/nuscenes/samples', '/home/chengjiafeng/work/data/nuscene/nuscenes_8clips_cam')
            # data_path = cam_info["data_path"].replace('/home/ma-user/work/data/ali_odd', '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd')

            print("==============")
            print(data_path)
            image_paths.append(data_path)
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(np.array(cam_info["sensor2lidar_rotation"]))
            lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T  # todo

            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = copy.deepcopy(np.array(cam_info["cam_intrinsic"]))
            cam_intrinsic.append(intrinsic)
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
        return lidar2img_rts


    def draw_track_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['track'] and "anchor_queue" in result):
            return
        
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(len(data['gt_names'])):
            label = data['gt_names'][i]
            # "movable_object.barrier": "barrier",
            # "vehicle.bicycle": "bicycle",
            # "vehicle.bus.bendy": "bus",
            # "vehicle.bus.rigid": "bus",
            # "vehicle.car": "car",
            # "vehicle.construction": "construction_vehicle",
            # "vehicle.motorcycle": "motorcycle",
            # "human.pedestrian.adult": "pedestrian",
            # "human.pedestrian.child": "pedestrian",
            # "human.pedestrian.construction_worker": "pedestrian",
            # "human.pedestrian.police_officer": "pedestrian",
            # "movable_object.trafficcone": "traffic_cone",
            # "vehicle.trailer": "trailer",
            # "vehicle.truck": "truck",
            color = color_mapping[i % len(color_mapping)]
            no_vehicle_name_list = ['barrier', 'motorcycle', 'bicycle', 'traffic_cone', 'pedestrian']
            if label not in no_vehicle_name_list:
                dot_size = 150
            else:
                dot_size = 25

            center = np.array(data['gt_boxes'])[i, :2]
            masks = np.array(data['gt_agent_fut_masks'])[i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = np.array(data['gt_agent_fut_trajs'])[i].reshape(-1, 2)[masks]
            trajs = trajs.cumsum(axis=0) + center
            # trajs = trajs + center
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)
            
            self._render_traj(trajs, traj_score=1.0,
                            colormap='winter', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return
        
        bboxes = result['boxes_3d']
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj_score = result['trajs_score'][i].numpy()
            traj = result['trajs_3d'][i].numpy()
            num_modes = len(traj_score)
            center = bboxes[i, :2][None, None].repeat(num_modes, 1, 1).numpy()
            traj = np.concatenate([center, traj], axis=1)

            sorted_ind = np.argsort(traj_score)[::-1]
            sorted_traj = traj[sorted_ind, :, :2]
            sorted_score = traj_score[sorted_ind]
            norm_score = np.exp(sorted_score[0])

            for j in range(top_k - 1, -1, -1):
                viz_traj = sorted_traj[j]
                traj_score = np.exp(sorted_score[j])/norm_score
                self._render_traj(viz_traj, traj_score=traj_score,
                                colormap='winter', dot_size=dot_size)
    
    def draw_map_gt(self, data):
        if not self.plot_choices['map']:
            return
        vectors = data['map_annos']
        if isinstance(vectors, list):
            for vector in vectors:
                color = COLOR_VECTORS[0]
                pts = np.array(vector)[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)
        elif isinstance(vectors, dict):
            for label, vector_list in vectors.items():
                color = COLOR_VECTORS[int(label)]
                for vector in vector_list:
                    if len(vector) == 0:
                        continue
                    pts = np.array(vector)[:, :2]
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result):
            return

        for i in range(len(result['scores'])):
            score = result['scores'][i]
            if  score < MAP_SCORE_THRESH:
                continue
            color = COLOR_VECTORS[result['labels'][i]]
            pts = np.array(result['vectors'])[i]
            x = pts[:, 0]
            y = pts[:, 1]
            plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_planning_gt(self, data):
        if not self.plot_choices['planning']:
            return

        # draw planning gt
        masks = np.array(data['gt_ego_fut_masks']).astype(bool)
        if masks[0] != 0:
            plan_traj = np.array(data['gt_ego_fut_trajs'])[masks]
            cmd = data['gt_ego_fut_cmd']
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0,
                colormap='autumn', dot_size=50)

        # draw corners
        gt_ego_lcf_feat = data['gt_ego_lcf_feat']
        EGO_BBOX[0][-1] = np.arctan2(gt_ego_lcf_feat[1], gt_ego_lcf_feat[0])
        corners = box3d_to_corners(np.array(EGO_BBOX))[0, [0, 3, 7, 4, 0]]
        x = corners[:, 0]
        y = corners[:, 1]
        color = [0, 0, 0]
        self.axes.plot(x, y, color=color, linewidth=6, linestyle='--')


    def draw_planning_pred(self, data, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "plan_traj" in result):
            return

        if self.plot_choices['track'] and "ego_anchor_queue" in result:
            ego_temp_bboxes = result["ego_anchor_queue"]
            ego_period = result["ego_period"]
            for j in range(ego_period[0]):
                # draw corners
                corners = box3d_to_corners(ego_temp_bboxes[:, -1-j])[0, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color='mediumseagreen', linewidth=2, linestyle='-')
        # import ipdb; ipdb.set_trace()
        # plan_trajs = result['plan_traj'].cpu().numpy()
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_score = result['planning_score'].cpu().numpy()

        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]

        # sorted_ind = np.argsort(plan_score)[::-1]
        # sorted_traj = plan_trajs[sorted_ind, :, :2]
        # sorted_score = plan_score[sorted_ind]
        # norm_score = np.exp(sorted_score[0])

        # for j in range(top_k - 1, -1, -1):
        #     viz_traj = sorted_traj[j]
        #     traj_score = np.exp(sorted_score[j]) / norm_score
        #     self._render_traj(viz_traj, traj_score=traj_score,
        #                     colormap='autumn', dot_size=50)
        plan_traj = np.array(result['plan_traj'])
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = plan_traj.cumsum(axis=0)
        plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
        self._render_traj(plan_traj, traj_score=1.0,
            colormap='autumn', dot_size=50)

    def _render_traj(
        self, 
        future_traj, 
        traj_score=1, 
        colormap='winter', 
        points_per_step=20, 
        dot_size=25
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_command(self, data):
        cmd = np.array(data['gt_ego_fut_cmd']).argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)

    def _render_pred_title(self):
        self.axes.text(-38, 30, "Pred", fontsize=60)

    def _render_gt_title(self):
        self.axes.text(-38, 30, "GT", fontsize=60)