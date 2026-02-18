
custom_imports = dict(
    imports=['mmpretrain.models'], # This registers the ViT backbone
    allow_failed_imports=False)
# Inherit from the official ViTPose-Base config
# _base_ = [
#     '../coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py' 
# ]
_base_ =['./bases/vitpose_originals/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py']

#  OVERRIDE GLOBAL SETTINGS 
dataset_metainfo = dict(from_file='configs/_base_/datasets/surg_7kpt.py')
data_root = 'data/SurgPose'
work_dir = 'results/Keypoints_detection/training_results/ViTpose_trainings/Experiment1' 

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
        ann_file='SurgPose_for_Vitpose/annotations/train.json', 
        data_prefix=dict(img='SurgPose_for_HRNet/Extracted_left_right/extracted_frames/'),
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
        ann_file='SurgPose_for_Vitpose/annotations/val.json',
        data_prefix=dict(img='SurgPose_for_HRNet/Extracted_left_right/extracted_frames/'),
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
        ann_file='SurgPose_for_Vitpose/annotations/test.json',
        data_prefix=dict(img='SurgPose_for_HRNet/Extracted_left_right/extracted_frames/'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)

#  EVALUATOR OVERRIDES 
val_evaluator = dict(ann_file='data/SurgPose/SurgPose_for_Vitpose/annotations/val.json')
test_evaluator = dict(ann_file='data/SurgPose/SurgPose_for_Vitpose/annotations/test.json')

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
        save_best='coco/AP',    # save the best performing model
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50) # Print logs every 50 iterations
)


#   PRETRAINED WEIGHTS 
#load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth'
load_from ='weights/original_weights/ViTpose/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth'