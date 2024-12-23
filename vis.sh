#!/bin/bash

PLOT_CHOICES=(
    "--draw-pred" "True"
    "--det" "True"
    "--track" "True"
    "--motion" "True"
    "--map" "True"
    "--planning" "True"
)

input_path=/data/sfs_turbo/perception/nuScenes/zdrive/clip_1730513272400
result_path=/data/sfs_turbo/perception/nuScenes/zdrive/samples_results
out_dir=/data/sfs_turbo/perception/nuScenes/zdrive/vis
python3 tool/visualization/visualization.py -i $input_path -r $result_path -o $out_dir