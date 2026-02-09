
custom_imports = dict(
    imports=['mmpretrain.models'], # This registers the ViT backbone
    allow_failed_imports=False)
# Inherit from the official ViTPose-Base config
_base_ = [
    '../coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py' 
]

#  OVERRIDE GLOBAL SETTINGS 
dataset_metainfo = dict(from_file='configs/_base_/datasets/surg_7kpt.py')
data_root = '/srv/homes/onbo10/thesis_Ons/SurgePoseData/Extracted_left_right/extracted_frames'
work_dir = '/srv/homes/onbo10/thesis_Ons/ViTPose/work_dirs/vitpose_base_surg_experiment_1'

#  MODEL OVERRIDES 
model = dict(
    head=dict(
        out_channels=7,  # Change 17 -> 7
    )
)

#  DATALOADER OVERRIDES 
# We only change the parts that are different from the inherited COCO config
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        metainfo=dataset_metainfo,
        data_root=data_root,
        ann_file='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/train.json',
        data_prefix=dict(img=''),
    )
)
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs')
]
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        _delete_=True, 
        type='CocoDataset',
        metainfo=dataset_metainfo,
        data_root=data_root,
        ann_file='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    )
)


test_dataloader = dict(
    batch_size=1,  
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        metainfo=dataset_metainfo,
        data_root=data_root,
        ann_file='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    )
)

#  EVALUATOR OVERRIDES 
val_evaluator = dict(ann_file='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/val.json')
test_evaluator = dict(ann_file='/srv/homes/onbo10/thesis_Ons/ViTPose/mmpose_data/data/annotations/test.json')

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop', 
    max_epochs=200, 
    val_interval=10  # Validation every 10 epochs to save time
)

# Learning Rate Scheduler

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[170, 190],
        gamma=0.1)
]

#  CHECKPOINT and SAVING 
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,            # Save a checkpoint every 50 epochs
        max_keep_ckpts=3,       # Keep only the last 3 to save disk space
        save_best='coco/AP',    # AUTOMATICALLY save the best performing model
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50) # Print logs every 50 iterations
)


#   PRETRAINED WEIGHTS 
load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth'