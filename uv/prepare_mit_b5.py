"""Strictly validate and convert the pinned Hugging Face F32 safetensors encoder.

The legacy environment needs no new safetensors dependency. No tensor renaming
or shape adaptation is allowed; only the unused ImageNet head may be omitted.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'uv/.cache/matplotlib'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'seg'))


def main():
    import numpy as np
    import torch
    from mmseg.models.backbones.mix_transformer import mit_b5

    torch.set_num_threads(2)
    source = ROOT / 'dataset/mit_b5.huggingface.safetensors'
    with source.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(stream.read(length))
        payload = stream.read()
    state = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        if info['dtype'] != 'F32':
            raise ValueError('Unexpected tensor dtype: ' + info['dtype'])
        start, end = info['data_offsets']
        if not 0 <= start <= end <= len(payload):
            raise ValueError('Invalid tensor offsets: ' + name)
        shape = info['shape']
        count = int(np.prod(shape))
        if end - start != count * 4:
            raise ValueError('Invalid tensor size: ' + name)
        array = np.frombuffer(payload, dtype='<f4', count=count, offset=start).reshape(shape).copy()
        state[name] = torch.from_numpy(array)
    model = mit_b5()
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong = [name for name in expected if name in state and expected[name].shape != state[name].shape]
    if missing or wrong or set(extra) - {'head.weight', 'head.bias'}:
        raise RuntimeError(dict(missing=missing, extra=extra, wrong_shapes=wrong))
    encoder = {name: state[name] for name in expected}
    model.load_state_dict(encoder, strict=True)
    if not all(torch.isfinite(value).all().item() for value in encoder.values()):
        raise RuntimeError('Non-finite pretrained tensor')
    output = ROOT / 'seg/pretrained/mit_b5.pth'
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        saved = torch.load(str(output), map_location='cpu')
        if set(saved) != set(encoder) or any(not torch.equal(saved[k], v) for k, v in encoder.items()):
            raise RuntimeError('Refusing to replace a different pretrained checkpoint')
    else:
        torch.save(encoder, str(output))
    existing = torch.load(str(ROOT / 'dataset/mit_b5.pth'), map_location='cpu')
    same = all(k in existing and torch.equal(existing[k], v) for k, v in encoder.items())
    report = dict(source='https://huggingface.co/smp-hub/mit_b5.imagenet',
                  revision='0851a37f3f718608d9b3cc8ff4cafb81f07374a1',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output=str(output.relative_to(ROOT)), encoder_tensors=len(encoder),
                  parameters=sum(p.numel() for p in model.parameters()),
                  ignored_tensors=extra, strict_load=True,
                  identical_to_supplied_pth=same)
    folder = ROOT / 'uv/seg/runs/mic-setup'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'weights.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
