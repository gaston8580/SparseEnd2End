#!/bin/bash

PLOT_CHOICES=(
    "--draw-pred" "True"
    "--det" "True"
    "--track" "True"
    "--motion" "True"
    "--map" "True"
    "--planning" "True"
)

input_path=/home/chengjiafeng/work/data/nuscene/dazhuo
result_path=/home/chengjiafeng/work/data/nuscene/dazhuo
out_dir=/home/chengjiafeng/work/data/nuscene/vis_dazhuo_private_detect
python3 tool/visualization/visualization.py -i $input_path -r $result_path -o $out_dir