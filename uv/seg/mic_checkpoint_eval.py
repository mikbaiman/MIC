"""Original pretrained MIC(HRDA) evaluation, with outer crops run sequentially.

No resolution/crop-size reduction: batching is changed to reduce peak VRAM.
"""
_base_ = ['../../dataset/checkpoints/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py']
model = dict(test_cfg=dict(batched_slide=False))
data = dict(workers_per_gpu=0)
