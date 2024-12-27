config_path=dataset/config/sparse4d_temporal_r50_1x4_bs22_256x704_zdrive_det_map_pnp_stage2_3_private_no_aug.py

python3 script/test.py --config ${config_path} --checkpoint /home/chengjiafeng/work/data/nuscene/zdrive_det_map_pnp_stage2_3_private_no_aug_latest.pth --vis