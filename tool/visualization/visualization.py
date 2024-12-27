import os
import glob
import argparse
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from utils import prepare_images_for_vconcat
from bev_render import BEVRender
# from bev_render_project import BEVRender
from cam_render import CamRender

plot_choices = dict(
    draw_pred=True,  # True: draw gt and pred; False: only draw gt
    det=True,
    track=True,  # True: draw history tracked boxes
    motion=True,
    map=True,
    planning=True,
)
START = 0
END = 39
INTERVAL = 1


class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, "combine")
        os.makedirs(self.combine_dir, exist_ok=True)

        self.input_path = Path(args.input_path)
        self.dataset = json.load(open(list(self.input_path.glob('*.json'))[0], "r"))
        self.results = json.load(open(list(self.input_path.glob('*.json'))[0], "r"))

        # plot_choices = vars(args.plot_choices)
        plot_choices = plot_choices
        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)
        self.with_infer = args.with_infer

    def add_vis(self, index):
        data = self.dataset[index]
        result = self.results[index]

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        if isinstance(cam_pred_path, tuple):
            self.combine_lidar(bev_gt_path, bev_pred_path, cam_pred_path, index)
        else:
            self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)

    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + ".jpg")
        cv2.imwrite(save_path, merge_image)

    def combine_lidar(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path[0])
        lidar_image = cv2.imread(cam_pred_path[1])
        merge_image = cv2.hconcat([lidar_image, bev_image, bev_gt])
        cam_image, merge_image = prepare_images_for_vconcat(cam_image, merge_image)
        merge_image = cv2.vconcat([cam_image, merge_image])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + ".jpg")
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, "*.jpg"))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(
                img,
                (width // downsample, height // downsample),
                interpolation=cv2.INTER_AREA,
            )
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, "video_nuscene_8clips.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize groundtruth and results")
    parser.add_argument(
        "-i",
        "--input-path",
        default="/home/chengjiafeng/work/data/nuscene/dazhuo",
        type=str,
        help="input dir path of json/pkl files",
    )
    parser.add_argument(
        "-r",
        "--result-path",
        default='/home/chengjiafeng/work/data/nuscene/dazhuo',
        type=str,
        help="prediction result to visualize",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="/home/chengjiafeng/work/data/nuscene/vis_dazhuo_detect",
        type=str,
        help="directory where visualize results will be saved",
    )
    parser.add_argument(
        "--with-infer",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument('--draw-pred', dest='draw_pred', action='store_true', help='Draw gt and pred')
    parser.add_argument('--no-draw-pred', dest='draw_pred', action='store_false', help='Only draw gt')
    parser.set_defaults(draw_pred=True)

    parser.add_argument('--det', action='store_true', help='Enable detection')
    parser.add_argument('--no-det', action='store_false', help='Disable detection')
    parser.set_defaults(det=True)

    parser.add_argument('--track', action='store_true', help='Draw history tracked boxes')
    parser.add_argument('--no-track', action='store_false', help='Do not draw history tracked boxes')
    parser.set_defaults(track=True)

    parser.add_argument('--motion', action='store_true', help='Enable motion')
    parser.add_argument('--no-motion', action='store_false', help='Disable motion')
    parser.set_defaults(motion=True)

    parser.add_argument('--map', action='store_true', help='Enable map')
    parser.add_argument('--no-map', action='store_false', help='Disable map')
    parser.set_defaults(map=True)

    parser.add_argument('--planning', action='store_true', help='Enable planning')
    parser.add_argument('--no-planning', action='store_false', help='Disable planning')
    parser.set_defaults(planning=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END, INTERVAL)):
        if idx > len(visualizer.results):
            break
        visualizer.add_vis(idx)

    visualizer.image2video()


if __name__ == "__main__":
    main()
