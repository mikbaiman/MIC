"""Audit a real MIC smoke checkpoint and logged losses; fail on incomplete work."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', default='uv/seg/runs/mic-real-smoke-fd')
    parser.add_argument('--export-student', action='store_true',
                        help='Also write a compact inference checkpoint without teacher/optimizer state.')
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    folder = (ROOT / args.run).resolve()
    checkpoint = torch.load(str(folder / 'iter_2.pth'), map_location='cpu')
    pretrained = torch.load(str(ROOT / 'seg/pretrained/mit_b5.pth'), map_location='cpu')
    state = checkpoint['state_dict']
    assert checkpoint['meta']['iter'] == 2, checkpoint['meta']
    backbone = {k[len('model.backbone.'):]: v for k, v in state.items() if k.startswith('model.backbone.')}
    assert set(backbone) == set(pretrained), 'Backbone tensor names differ'
    changed = sum(not torch.equal(backbone[k], v) for k, v in pretrained.items())
    assert changed > 0, 'Student encoder was not updated'
    teacher = {k[len('ema_model.backbone.'):]: v for k, v in state.items() if k.startswith('ema_model.backbone.')}
    assert set(teacher) == set(backbone), 'EMA teacher missing'
    assert any(not torch.equal(teacher[k], v) for k, v in backbone.items()), 'EMA and student unexpectedly identical'
    reference = {k[len('imnet_model.backbone.'):]: v for k, v in state.items() if k.startswith('imnet_model.backbone.')}
    assert set(reference) == set(pretrained), 'ImageNet feature-distance network missing'
    assert all(torch.equal(reference[k], v) for k, v in pretrained.items()), 'Frozen ImageNet encoder changed'
    assert all(torch.isfinite(v).all().item() for v in state.values()), 'Non-finite model tensor'
    records = []
    for log in sorted(folder.glob('*.log.json')):
        for line in log.read_text().splitlines():
            record = json.loads(line)
            if 'masked.decode.loss_seg' in record:
                records.append(record)
    assert len(records) >= 2, 'Expected two logged MIC training updates'
    records = records[-2:]
    for record in records:
        for key, value in record.items():
            if 'loss' in key:
                assert math.isfinite(value), (key, value)
        for key in ('decode.loss_seg', 'mix.decode.loss_seg', 'masked.decode.loss_seg'):
            assert record[key] > 0, (key, record[key])
        assert 'src.loss_imnet_feat_dist' in record, 'ImageNet feature-distance loss not executed'
    report = dict(checkpoint=str((folder / 'iter_2.pth').relative_to(ROOT)),
                  iterations=2, encoder_tensors_changed=changed,
                  finite_model=True, ema_teacher_verified=True, frozen_imagenet_verified=True,
                  losses=[{k: v for k, v in r.items() if 'loss' in k} for r in records],
                  peak_training_memory_MiB=max(r.get('memory', 0) for r in records))
    if args.export_student:
        student = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        output = folder / 'student_2.pth'
        if output.exists():
            old = torch.load(str(output), map_location='cpu')['state_dict']
            assert set(old) == set(student) and all(torch.equal(old[k], v) for k, v in student.items()), 'Existing export differs'
        else:
            torch.save(dict(state_dict=student, meta=checkpoint['meta']), str(output))
        report['student_checkpoint'] = str(output.relative_to(ROOT))
    (folder / 'verification.json').write_text(json.dumps(report, indent=2))
    print('PASS: MiT-B5/HRDA/MIC updates, EMA teacher, frozen ImageNet network, and finite source/mixed/masked/feature losses.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
