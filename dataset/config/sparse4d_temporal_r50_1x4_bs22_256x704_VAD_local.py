# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
"""
mAP: 0.4645
mATE: 0.5267
mASE: 0.2663
mAOE: 0.4413
mAVE: 0.2286
mAAE: 0.1889
NDS: 0.5623
Eval time: 115.3s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.665	0.375	0.144	0.053	0.185	0.193
truck	0.379	0.527	0.182	0.068	0.179	0.200
bus	0.421	0.698	0.200	0.132	0.405	0.217
trailer	0.184	0.908	0.243	0.514	0.198	0.103
construction_vehicle	0.113	0.812	0.498	1.316	0.113	0.406
pedestrian	0.536	0.531	0.293	0.530	0.297	0.149
motorcycle	0.483	0.470	0.254	0.391	0.320	0.235
bicycle	0.446	0.374	0.262	0.880	0.132	0.008
traffic_cone	0.717	0.245	0.312	nan	nan	nan
barrier	0.645	0.278	0.276	0.089	nan	nan


Per-class results:
		         AMOTA	AMOTP	RECALL	MOTAR	GT	MOTA	MOTP	MT	ML	FAF	TP	FP	FN	IDS	FRAG	TID	LGD
bicycle     0.445	1.196	0.493	0.802	1993	0.394	0.484	45	66	13.7	979	194	1010	4	5	1.48	1.75
bus     	   0.491	1.330	0.577	0.721	2112	0.415	0.777	33	40	21.4	1216	339	893	3	16	1.16	2.48
car     	    0.673	0.810	0.729	0.821	58317	0.596	0.486	2024	984	131.9	42358	7587	15793	166	330	0.84	1.15
motorcy  0.489	1.154	0.556	0.858	1977	0.472	0.533	41	45	11.6	1087	154	878	12	9	2.04	2.31
pedestr    0.531	1.133	0.621	0.789	25423	0.479	0.679	614	522	74.9	15447	3266	9623	353	247	1.51	2.02
trailer 	  0.081	1.607	0.327	0.303	2425	0.099	0.908	24	83	53.9	791	551	1631	3	11	1.36	3.09
truck   	  0.418	1.168	0.550	0.646	9650	0.355	0.612	177	226	49.2	5298	1877	4342	10	61	1.44	2.10

Aggregated results:
AMOTA	0.457
AMOTP	1.196
RECALL	0.561
MOTAR	0.706
GT	14556
MOTA	0.421
MOTP	0.640
MT	2958
ML	1966
FAF	51.0
TP	67176
FP	13968
FN	34170
IDS	541
FRAG	679
TID	1.40
LGD	1.99
Eval time: 1588.5s
"""


log_level = "INFO"
dist_params = dict(backend="nccl")

total_batch_size = 4
num_gpus = 1
batch_size = total_batch_size // num_gpus


# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
with_quality_estimation = True

num_sample = 20
num_single_frame_decoder_map = 1
use_deformable_func = True

map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]
num_map_classes = len(map_class_names)
roi_size = (30, 60)

model = dict(
    type="Sparse4D",
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="ckpt/resnet50-19c8e357.pth"),
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="/home/chengjiafeng/work/data/nuscene/mini/data/kmeans_det_900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=(
            dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                attn_drop=drop_out,
            )
            if temporal
            else None
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        norm_layer=dict(type="LayerNorm", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LayerNorm"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableAttentionAggr",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="GaussianFocalLoss"),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),

    map_head = dict(
            type="Sparse4DMapHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn_map,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="/home/chengjiafeng/work/data/nuscene/mini/data/kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder_map,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)
            )[:],
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                attn_drop=drop_out,
            )
            if temporal_map
            else None,
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                attn_drop=drop_out,
            ),
            norm_layer=dict(type="LayerNorm", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LayerNorm"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableAttentionAggr",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023, # ground height in lidar frame
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=num_map_classes,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(
                    type='LinesL1Loss',
                    loss_weight=10.0,
                    beta=0.01,
                ),
                num_sample=num_sample,
                roi_size=roi_size,
            ),
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            gt_id_key="map_instance_id",
            with_instance_id=False,
            task_prefix='map',
    ),

    
    motion_head=dict(
        type="Sparse4DMotionHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="/home/chengjiafeng/work/data/nuscene/mini/data/kmeans_det_900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=3,
            embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=(
            dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                attn_drop=drop_out,
            )
            if temporal
            else None
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        norm_layer=dict(type="LayerNorm", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LayerNorm"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableAttentionAggr",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            dn_noise_scale=[2.0] * 3 + [0.5] * 7,
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="L1Loss",
            reduction="mean", 
            loss_weight=1.0
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),

    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    use_grid_mask=True,
    use_deformable_func=True,
)

# ================== data ========================
dataset_type = "NuScenes4DDetTrackVADDataset"
data_root = "/home/chengjiafeng/work/data/nuscene/nuscenes/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        load_dim=5,
        use_dim=5,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "lidar2img",
            "image_wh",
            "gt_depth",
            "gt_depth_ori",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "fut_valid_flag",
            'gt_map_labels', 
            'gt_map_pts',
            "ego_his_trajs",
            "ego_fut_trajs",
            "ego_fut_masks",
            "ego_fut_cmd",
            "ego_lcf_feat",
        ],
        meta_keys=[
            "sample_idx",
            "sample_scene",
            "lidar2global",
            "global2lidar",
            "timestamp",
            "track_id",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        load_dim=5,
        use_dim=5,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "lidar2img",
            "image_wh",
            "gt_depth",
            "gt_depth_ori",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "fut_valid_flag",
            'gt_map_labels', 
            'gt_map_pts',
            "ego_his_trajs",
            "ego_fut_trajs",
            "ego_fut_masks",
            "ego_fut_cmd",
            "ego_lcf_feat",
        ],
        meta_keys=[
            "sample_idx",
            "sample_scene",
            "filename",
            "lidar2global",
            "global2lidar",
            "timestamp",
            "track_id",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "lidar2img",
            "image_wh",
            "ori_img",
            "gt_bboxes_3d",
            "ego_his_trajs",
            "ego_fut_trajs",
            "ego_fut_cmd",
            "map_annos",
        ],
        meta_keys=[
            "lidar2global",
            "global2lidar",
            "timestamp",
            "filename",
            "sample_idx",
        ],
    ),
]


data_basic_config = dict(
    type=dataset_type, data_root=data_root, classes=class_names, version="v1.0-mini"
)

input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],  # h,w
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

eval_config = dict(
    **data_basic_config,
    test_mode=True,
)

tracking_threshold = 0.2
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size//2,
    train=dict(
        **data_basic_config,
        ann_file="/home/chengjiafeng/work/data/nuscene/mini/data/nusc_anno_dumpjson/train",
        pipeline=train_pipeline,
        train_mode=True,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file="/home/chengjiafeng/work/data/nuscene/mini/data/nusc_anno_dumpjson/val",
        pipeline=val_pipeline,
        data_aug_conf=data_aug_conf,
        val_mode=True,
        tracking=True,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=class_names,
        version="v1.0-mini",
        ann_file="/home/chengjiafeng/work/data/nuscene/mini/data/nusc_anno_dumpjson/val",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=True,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== runner&hook ========================
seed = 100
# RunnerSetting
workflow = [("train", 1)]
num_epochs = 150
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)


# CheckpointSaverHookSetting
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)

# LoggerHookSetting
log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

# ================== train ========================

optimizer = dict(
    type="AdamW",
    lr=6e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)

# LrUpdaterHookSetting
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# OptimizerStepperHookSetting
# fp16 = dict(loss_scale=32.0)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))

# ================== eval ========================

vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=True,
    with_motion=True,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    pipeline=vis_pipeline,
    eval_mode=eval_mode,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=class_names,
        version="v1.0-mini",
        ann_file="/home/chengjiafeng/work/data/nuscene/mini/data/nusc_anno_dumpjson/val",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=True,
        tracking_threshold=tracking_threshold,
    ),
)

# ================== placehold ========================
work_dir = None
load_from = None
resume_from = None
gpu_ids = None  # only applicable to non-distributed training