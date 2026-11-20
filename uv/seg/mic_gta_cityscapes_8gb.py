"""MiT-B5/HRDA/MIC training for a small GPU; not the paper's training protocol.

256x256 training crops and 128x128 detail crops reduce memory. The student, EMA
teacher, ImageNet feature-distance network, ClassMix, and MIC remain active.
"""
_base_ = ['../../seg/configs/mic/gtaHR2csHR_mic_hrda.py']
crop_size = (256, 256)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    # Explicit common size also aligns the known GTA image/label size mismatches.
    dict(type='Resize', img_scale=(640, 360), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5), dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 256), keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size), dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'), dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
test_pipeline = [dict(type='LoadImageFromFile'), dict(
    type='MultiScaleFlipAug', img_scale=(512, 256), flip=False,
    transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg), dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])])]
model = dict(hr_crop_size=(128, 128), test_cfg=dict(
    mode='slide', batched_slide=False, stride=[128, 128], crop_size=[256, 256]))
data = dict(samples_per_gpu=2, workers_per_gpu=0, train=dict(
    source=dict(pipeline=gta_train_pipeline), target=dict(pipeline=cityscapes_train_pipeline),
    rare_class_sampling=dict(min_crop_ratio=0.5)),
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
seed = 0
name = 'gta2cityscapes_mic_hrda_mitb5_256_s0'
