"""Two real GTA->Cityscapes MIC updates, followed by validation/checkpointing.

Threshold zero is solely for this execution check: every teacher label supplies
a gradient even before the randomly initialized decoder becomes confident.
"""
_base_ = ['mic_gta_cityscapes_8gb.py']
runner = dict(max_iters=2)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2)
log_config = dict(interval=1, img_interval=1000)
lr_config = dict(warmup=None)
uda = dict(pseudo_threshold=0.0)
data = dict(
    val=dict(split='../../../uv/seg/cityscapes_val_subset.txt'),
    test=dict(split='../../../uv/seg/cityscapes_val_subset.txt'))
