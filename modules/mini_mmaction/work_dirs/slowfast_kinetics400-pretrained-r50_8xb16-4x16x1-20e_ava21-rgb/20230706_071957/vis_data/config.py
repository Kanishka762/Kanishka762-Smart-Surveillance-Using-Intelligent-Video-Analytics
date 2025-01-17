default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
log_level = 'INFO'
load_from = 'work_dirs/last_checkpoint'
resume = False
url = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth'
model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth'
    ),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(
                1,
                7,
                7,
            ),
            dilations=(
                1,
                1,
                1,
                1,
            ),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(
                0,
                0,
                1,
                1,
            ),
            spatial_strides=(
                1,
                2,
                2,
                1,
            )),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(
                5,
                7,
                7,
            ),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(
                1,
                2,
                2,
                1,
            ))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2304,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))
dataset_type = 'AVADataset'
data_root = '../mmaction2_v1.0/data/ava/rawframes'
anno_root = '../mmaction2_v1.0/data/ava/annotations'
ann_file_train = '../mmaction2_v1.0/data/ava/annotations/ava_custom_train_v2.2.csv'
ann_file_val = '../mmaction2_v1.0/data/ava/annotations/ava_custom_val_v2.2.csv'
exclude_file_train = '../mmaction2_v1.0/data/ava/annotations/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = '../mmaction2_v1.0/data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv'
label_file = '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt'
proposal_file_train = '../mmaction2_v1.0/data/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl'
proposal_file_val = '../mmaction2_v1.0/data/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='RandomRescale', scale_range=(
        256,
        320,
    )),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(
        -1,
        256,
    )),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs'),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AVADataset',
        ann_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_custom_train_v2.2.csv',
        exclude_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_train_excluded_timestamps_v2.2.csv',
        pipeline=[
            dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='RandomRescale', scale_range=(
                256,
                320,
            )),
            dict(type='RandomCrop', size=256),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs'),
        ],
        label_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        proposal_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
        data_prefix=dict(img='../mmaction2_v1.0/data/ava/rawframes')))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AVADataset',
        ann_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_custom_val_v2.2.csv',
        exclude_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv',
        pipeline=[
            dict(
                type='SampleAVAFrames',
                clip_len=32,
                frame_interval=2,
                test_mode=True),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='Resize', scale=(
                -1,
                256,
            )),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs'),
        ],
        label_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        proposal_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        data_prefix=dict(img='../mmaction2_v1.0/data/ava/rawframes'),
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AVADataset',
        ann_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_custom_val_v2.2.csv',
        exclude_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv',
        pipeline=[
            dict(
                type='SampleAVAFrames',
                clip_len=32,
                frame_interval=2,
                test_mode=True),
            dict(type='RawFrameDecode', io_backend='disk'),
            dict(type='Resize', scale=(
                -1,
                256,
            )),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='PackActionInputs'),
        ],
        label_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        proposal_file=
        '../mmaction2_v1.0/data/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        data_prefix=dict(img='../mmaction2_v1.0/data/ava/rawframes'),
        test_mode=True))
val_evaluator = dict(
    type='AVAMetric',
    ann_file='../mmaction2_v1.0/data/ava/annotations/ava_custom_val_v2.2.csv',
    label_file=
    '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
    exclude_file=
    '../mmaction2_v1.0/data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv'
)
test_evaluator = dict(
    type='AVAMetric',
    ann_file='../mmaction2_v1.0/data/ava/annotations/ava_custom_val_v2.2.csv',
    label_file=
    '../mmaction2_v1.0/data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
    exclude_file=
    '../mmaction2_v1.0/data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv'
)
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[
            10,
            15,
        ],
        gamma=0.1),
]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-05),
    clip_grad=dict(max_norm=40, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=128)
launcher = 'none'
work_dir = './work_dirs/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb'
