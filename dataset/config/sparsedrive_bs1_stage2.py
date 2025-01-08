log_level = "INFO"
dist_params = dict(backend="nccl")

total_batch_size = 1
num_gpus = 1
batch_size = total_batch_size // num_gpus

# ================== model ========================
class_names = [
    "car", 
    "bus", 
    "construction_vehicle",
    "tricycle", 
    "bicycle",
    "motorcycle", 
    "person",
    "traffic_cone", 
    "barrier",
]

num_classes = len(class_names)
input_shape = (704, 256)
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
decouple_attn_motion = True
with_quality_estimation = True

num_sample = 20
num_single_frame_decoder_map = 1
use_deformable_func = True

map_class_names = ['divider']
num_map_classes = len(map_class_names)
roi_size = (60, 60)

# pnp
n = 5
# fut_ts = 12 * n
fut_ts = 6 * n
fut_mode = 6
ego_fut_ts = 6 * n
ego_fut_mode = 6
queue_length = (3 * n) + 1  # history + current

task_config = dict(
    with_det=True,
    with_map=True,
    with_motion_plan=True,
)

model = dict(
    type="SparseDrive",
    task_config=task_config,
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
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
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    det_head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="data/zdrive/anchor/kmeans_det_900_zdrive.npy",
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
            #num_cams=6,
            num_cams=7,
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
            num_dn_groups=0,
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
            loss_weight=0.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
            loss_yawness=dict(type="GaussianFocalLoss"),
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
    map_head=dict(
        type="Sparse4DMapHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn_map,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=100,
            embed_dims=embed_dims,
            anchor="data/zdrive/anchor/kmeans_map_100_zdrive.npy",
            anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
            # num_temp_instances=0 if temporal_map else -1,
            num_temp_instances=33 if temporal_map else -1,
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
            num_cams=7,
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
                ground_height=-1.84023,  # ground height in lidar frame
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
                    cls_cost=dict(type='FocalLossCost', weight=0.0),
                    reg_cost=dict(type='LinesL1Cost', weight=0.0, beta=0.01, permute=True),
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
            loss_weight=0.0,
        ),
        loss_reg=dict(
            type="SparseLineLoss",
            loss_line=dict(
                type='LinesL1Loss',
                loss_weight=0.0,
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
    motion_plan_head=dict(
        type='MotionPlanningHead',
        fut_ts=fut_ts,
        fut_mode=fut_mode,
        ego_fut_ts=ego_fut_ts,
        ego_fut_mode=ego_fut_mode,
        motion_anchor=f'data/zdrive/anchor/kmeans_motion_{fut_mode}_zdrive.npy',
        plan_anchor=f'data/zdrive/anchor/kmeans_plan_{ego_fut_mode}_zdrive.npy',
        embed_dims=embed_dims,
        decouple_attn=decouple_attn_motion,
        instance_queue=dict(
            type="InstanceQueue",
            embed_dims=embed_dims,
            queue_length=queue_length,
            tracking_threshold=0.2,
            feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
        ),
        operation_order=(
            [
                "temp_gnn",
                "gnn",
                "norm",
                "cross_gnn",
                "norm",
                "ffn",                    
                "norm",
            ] * 3 +
            [
                "refine",
            ]
        ),
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        cross_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        norm_layer=dict(type="LayerNorm", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims,
            pre_norm=dict(type="LayerNorm"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 2,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        refine_layer=dict(
            type="MotionPlanningRefinementModule",
            embed_dims=embed_dims,
            fut_ts=fut_ts,
            fut_mode=fut_mode,
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
        ),
        motion_sampler=dict(
            type="MotionTarget",
        ),
        motion_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2
        ),
        motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),
        planning_sampler=dict(
            type="PlanningTarget",
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
        ),
        plan_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5,
        ),
        plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
        plan_loss_status=dict(type='L1Loss', loss_weight=1.0),
        motion_decoder=dict(type="SparseBox3DMotionDecoder"),
        planning_decoder=dict(
            type="HierarchicalPlanningDecoder",
            ego_fut_ts=ego_fut_ts,
            ego_fut_mode=ego_fut_mode,
            use_rescore=True,
        ),
        num_det=50,
        num_map=10,
    ),
)

# ================== data ========================
dataset_type = "NuScenes4DDetTrackVADDataset"
#data_root = "data/nuscenes/"
data_root = "data/zdrive"

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
    #dict(type="BBoxRotation"),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),


    dict(type="ZdriveSparse4DAdaptor"),
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
            "agent_fut_trajs",
            "agent_fut_masks",
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
    dict(type="ZdriveSparse4DAdaptor"),
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
        keys=["img", "timestamp", "lidar2img", "image_wh", "ori_img", "ego_his_trajs", "ego_fut_trajs", "ego_fut_cmd",],
        meta_keys=["lidar2global", "global2lidar", "timestamp", "filename"],
    ),
]


data_basic_config = dict(
    type=dataset_type, data_root=data_root, classes=class_names, version="v1.0-trainval"
)

data_aug_conf_bak = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],  # h,w
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],  # h,w
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (1.0, 1.0),
    "H": 480,
    "W": 640,
    "rand_flip": False,
    "rot3d_range": [0.0, 0.0],
}

tracking_threshold = 0.2
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size//2,
    train=dict(
        **data_basic_config,
        ann_file="data/zdrive/annos_0106",
        pipeline=train_pipeline,
        train_mode=True,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file="data/zdrive/annos_0106",
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
        version="v1.0-trainval",
        ann_file="data/zdrive/annos_0106",
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
num_epochs = 2
#num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_iters_per_epoch = int(398 // (num_gpus * batch_size))
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)


# CheckpointSaverHookSetting
checkpoint_epoch_interval = 10
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
    lr=3e-4,
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
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval*300,
    pipeline=vis_pipeline,
)

# ================== placehold ========================
work_dir = None
#load_from = None
load_from = 'ckpt/stage1.pth'
resume_from = None
gpu_ids = None  # only applicable to non-distributed training
