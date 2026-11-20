"""Matched small-GPU control: only masked image consistency is disabled."""
_base_ = ['mic_gta_cityscapes_8gb.py']
uda = dict(mask_mode=None, mask_lambda=0)
name = 'gta2cityscapes_hrda_mitb5_256_s0_no_mic'
