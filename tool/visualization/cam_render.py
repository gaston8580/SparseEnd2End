import os
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from bev_render import (
    color_mapping, 
    SCORE_THRESH, 
    MAP_SCORE_THRESH,
    CMD_LIST
)


CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

class CamRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
        dazhuo_data = True
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        os.makedirs(self.pred_dir, exist_ok=True)
        self.dazhuo_data = dazhuo_data
        if self.dazhuo_data:
            self.pred_lidar_dir = os.path.join(out_dir, "cam_pred_lidar")
            os.makedirs(self.pred_lidar_dir, exist_ok=True)

    def reset_canvas(self, num_cols):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(2, num_cols, figsize=(160 /3  /3 *num_cols  , 20))
        plt.tight_layout()

    def render(
        self,
        data, 
        result,
        index,
    ):
        origin_path_pre = '/home/ma-user/work/data/ali_odd'
        local_path_pre = '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd'
        img_path = data['cams']['CAM_FRONT_WIDE']['data_path']
        # # origin_path_pre = './data/nuscenes'
        # # local_path_pre = '/home/chengjiafeng/work/data/nuscene/nuscenes'
        # origin_path_pre = './data/nuscenes/samples'
        # local_path_pre = '/home/chengjiafeng/work/data/nuscene/nuscenes_8clips_cam'
        # img_path = data['cams']['CAM_FRONT']['data_path']

        img_path = img_path.replace(origin_path_pre, local_path_pre)
        save_result_path_prefix = img_path.split("/sample")[0]
        save_result_path_suffix = img_path.split("/camera0")[1].split(".jpg")[0]
        # save_result_path_prefix = img_path.split("/CAM_FRONT/")[0]
        # save_result_path_suffix = img_path.split("/CAM_FRONT/")[1].split(".jpg")[0]
        save_result_path_dir = os.path.join(save_result_path_prefix, "samples_results_dz_detect_private_no_aug")
        save_result_path = os.path.join(save_result_path_dir, save_result_path_suffix + ".json")
        result = json.load(open(save_result_path, "r"))


        num_cols = 3
        self.reset_canvas(num_cols)
        self.render_image_data(data, num_cols)
        # self.draw_detection_pred(data, result)
        # self.draw_motion_pred(data, result)
        # self.draw_planning_pred(data, result)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path)

        if self.dazhuo_data:
            num_cols = 1
            self.reset_canvas(num_cols)
            self.render_lidar_image_data(data, num_cols)
            # self.draw_detection_pred(data, result)
            # self.draw_motion_pred(data, result)
            # self.draw_planning_pred(data, result)
            save_lidar_path = os.path.join(self.pred_lidar_dir, str(index).zfill(4) + '.jpg')
            self.save_fig(save_lidar_path)

            return (save_path, save_lidar_path)
        return save_path

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def load_lidar_image(self, lidar_data, cam):
        """Update the axis of the plot with the provided image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(lidar_data, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, num_cols):
        """Render image data for each camera."""
        ax = self.get_axis(index, num_cols)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index, num_cols):
        """Retrieve the corresponding axis based on the index."""
        if num_cols == 1:
            return self.axes[index//num_cols]
        return self.axes[index//num_cols, index % num_cols]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def render_image_data(self, data, num_cols):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            img_path = data['cams'][cam]['data_path']
            img_path = img_path.replace('/home/ma-user/work/data/ali_odd', '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd')
            # img_path = img_path.replace('./data/nuscenes', '/home/chengjiafeng/work/data/nuscene/nuscenes')
            # img_path = img_path.replace('./data/nuscenes/samples', '/home/chengjiafeng/work/data/nuscene/nuscenes_8clips_cam')
            image = self.load_image(img_path, cam)
            self.update_image(image, i, num_cols)

    def render_lidar_image_data(self, data, num_cols):
        """Load and annotate image based on the provided path."""
        cam = 'CAM_FRONT_WIDE'
        img_path = data['cams'][cam]['data_path']
        img_path = img_path.replace('/home/ma-user/work/data/ali_odd', '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd')
        image = self.load_image(img_path, cam)
        self.update_image(image, 0, num_cols)

        # lidar = 'LIDAR'
        # lidar_img_path = data['lidar_path']
        # lidar_img_path = lidar_img_path.replace('/home/ma-user/work/data/ali_odd', '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd').replace('.bin', '.pcd')
        # with open(lidar_img_path, "rb") as f:
        #     pts_bytes = f.read()
        # lidar_points = np.frombuffer(pts_bytes, dtype=np.uint8)  # (nums_pts, )
        # lidar_image = self.load_lidar_image(lidar_points, lidar)
        # self.update_image(lidar_image, 1, num_cols)
        cam = 'CAM_FRONT_WIDE'
        img_path = data['cams'][cam]['data_path']
        img_path = img_path.replace('/home/ma-user/work/data/ali_odd', '/home/chengjiafeng/work/data/nuscene/dazhuo/ali_odd')
        image = self.load_image(img_path, cam)
        self.update_image(image, 1, num_cols)
    
    def draw_detection_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = np.array(result['boxes_3d'])
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = np.array(data['cams'][cam]['cam_intrinsic'])[idx]
            lidar2cam = np.array(data['lidar2cam'])
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(len(result['labels_3d'])):
                score = np.array(result['scores_3d'])[i]
                if score < SCORE_THRESH: 
                    continue
                # color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                color = color_mapping[i % len(color_mapping)]
                
                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3], 
                        view=cam_intrinsic, 
                        normalize=True, 
                        colors=(color, color, color),
                        linewidth=4,
                    )
            
            self.axes[j//3, j % 3].set_xlim(0, imsize[0])
            self.axes[j//3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH: 
                    continue
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                
                traj_score = result['trajs_score'][i].numpy()
                traj = result['trajs_3d'][i].numpy()
                
                mode_idx = traj_score.argmax()
                traj = traj[mode_idx]
                origin = bboxes[i, :2][None]
                traj = np.concatenate([origin, traj], axis=0)
                traj_expand = np.ones((traj.shape[0], 1)) 
                traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
                traj = np.concatenate([traj, traj_expand], axis=1)

                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                traj_points = traj @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

        
    def draw_planning_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "planning" in result):
            return
        # for j, cam in enumerate(CAM_NAMES_NUSC[1]):
        #     idx = CAM_NAMES_NUSC_converter.index(cam)
        #     cam_intrinsic = data['cam_intrinsic'][idx]
        #     lidar2cam = data['lidar2cam']
        #     extrinsic = lidar2cam[idx]
        #     trans = extrinsic[3, :3]
        #     rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        #     imsize = (1600, 900)

        #     plan_trajs = result['planning'][0].cpu().numpy()
        #     plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        #     num_cmd = len(CMD_LIST)
        #     num_mode = plan_trajs.shape[1]
        #     plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        #     plan_trajs = plan_trajs.cumsum(axis=-2)
        #     plan_score = result['planning_score'][0].cpu().numpy()
        #     plan_score = plan_score.reshape(3, -1)

        #     cmd = data['gt_ego_fut_cmd'].argmax()
        #     plan_trajs = plan_trajs[cmd]
        #     plan_score = plan_score[cmd]

        #     mode_idx = plan_score.argmax()
        #     plan_traj = plan_trajs[mode_idx]
        #     traj_expand = np.ones((plan_traj.shape[0], 1)) * -2
        #     # traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
        #     plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        #     traj_points = plan_traj @ extrinsic[:3, :3] + trans
        #     self._render_traj(traj_points, cam_intrinsic, j)

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        # plan_trajs = result['planning'][0].cpu().numpy()
        # plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_trajs = plan_trajs.cumsum(axis=-2)
        # plan_score = result['planning_score'][0].cpu().numpy()
        # plan_score = plan_score.reshape(3, -1)

        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]

        # mode_idx = plan_score.argmax()
        # plan_traj = plan_trajs[mode_idx]
        plan_traj = result["final_planning"]
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj(traj_points, cam_intrinsic, j=1)

    def _render_traj(self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150, points_per_step=10):
        total_steps = (len(traj_points)-1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps-1):
            unit_vec = traj_points[k//points_per_step +
                                    1] - traj_points[k//points_per_step]
            total_xy[k] = (k/points_per_step - k//points_per_step) * \
                unit_vec + traj_points[k//points_per_step]
        in_range_mask = total_xy[:, 2] > 0.1
        traj_points = view_points(
            total_xy.T, cam_intrinsic, normalize=True)[:2, :]
        traj_points = traj_points[:2, in_range_mask]
        self.axes[j // 3, j % 3].scatter(traj_points[0], traj_points[1], color=color, s=s)