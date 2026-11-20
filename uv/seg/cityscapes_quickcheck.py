"""Small supervised real-data check; not a MIC benchmark configuration.

Run from the repository root after Cityscapes preprocessing:
  bash uv/run.sh seg tools/train.py ../uv/seg/cityscapes_quickcheck.py \
    --work-dir ../uv/seg/runs/cityscapes-quickcheck
"""
_base_ = ['../../seg/configs/_base_/default_runtime.py']

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNet', depth=18, num_stages=4, out_indices=(3,),
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=False),
    decode_head=dict(
        type='FCNHead', in_channels=512, in_index=0, channels=32,
        num_convs=1, concat_input=False, dropout_ratio=0,
        num_classes=19, norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 256)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(512, 256), flip=False,
         transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'),
                     dict(type='Normalize', **img_norm_cfg),
                     dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])])]
data_root = 'data/cityscapes/'
# The dataset resolves split files relative to data_root, not the working directory.
train_subset = '../../../uv/seg/cityscapes_train_subset.txt'
val_subset = '../../../uv/seg/cityscapes_val_subset.txt'
data = dict(
    samples_per_gpu=2, workers_per_gpu=0,
    train=dict(type='CityscapesDataset', data_root=data_root,
               img_dir='leftImg8bit/train', ann_dir='gtFine/train',
               split=train_subset, pipeline=train_pipeline),
    val=dict(type='CityscapesDataset', data_root=data_root,
             img_dir='leftImg8bit/val', ann_dir='gtFine/val',
             split=val_subset, pipeline=test_pipeline),
    test=dict(type='CityscapesDataset', data_root=data_root,
              img_dir='leftImg8bit/val', ann_dir='gtFine/val',
              split=val_subset, pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9)
optimizer_config = dict()
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2)
checkpoint_config = dict(by_epoch=False, interval=2)
evaluation = dict(interval=2, metric='mIoU')
log_config = dict(interval=1, img_interval=1000)
seed = 0
