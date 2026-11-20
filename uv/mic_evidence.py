"""Compare published HRDA and MIC checkpoints on one fixed Cityscapes subset.

Runs paired inference and a small MIC backward-pass probe without weight
updates. Writes metrics, raw prediction caches, and two figures.

Baseline: https://github.com/lhoyer/HRDA (published GTA-to-Cityscapes model).
Use --refresh to independently recompute predictions instead of reading caches.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'uv/.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'seg'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--examples', type=int, default=5, help='Number of evenly spaced examples to visualize.')
    parser.add_argument('--out', default='uv/seg/runs/mic-evidence')
    parser.add_argument('--refresh', action='store_true', help='Recompute predictions even when a matching cache exists.')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from mmcv import Config
    from mmseg.apis import init_segmentor, inference_segmentor
    from mmseg.datasets import CityscapesDataset
    from tools.test import update_legacy_cfg

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Run from a terminal with NVIDIA driver access.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    rgb_root = ROOT / 'dataset/leftImg8bit/val'
    gt_root = ROOT / 'dataset/gtFine/val'
    all_images = sorted(rgb_root.rglob('*_leftImg8bit.png'))
    assert len(all_images) == 500, len(all_images)
    assert 1 <= args.images <= len(all_images)
    indices = sorted(np.random.RandomState(args.seed).choice(len(all_images), args.images, replace=False))
    selected = [all_images[i] for i in indices]
    assert 1 <= args.examples <= len(selected)
    example_indices = np.linspace(0, len(selected) - 1, args.examples, dtype=int).tolist()
    names = [str(p.relative_to(rgb_root)) for p in selected]
    (output / 'images.txt').write_text('\n'.join(names) + '\n')
    models = [
        ('HRDA_no_MIC', ROOT / 'dataset/checkpoints/hrda-baseline/gtaHR2csHR_hrda_246ef/gtaHR2csHR_hrda_246ef.json',
         ROOT / 'dataset/checkpoints/hrda-baseline/gtaHR2csHR_hrda_246ef/iter_40000_relevant.pth'),
        ('HRDA_MIC', ROOT / 'dataset/checkpoints/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py',
         ROOT / 'dataset/checkpoints/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth')]
    classes = list(CityscapesDataset.CLASSES)
    palette = np.array(CityscapesDataset.PALETTE, dtype=np.uint8)
    summaries, per_classes, example_predictions, image_scores = [], {}, {}, {}
    for name, config, checkpoint in models:
        print('MODEL', name, flush=True)
        signature = dict(images=names, checkpoint_sha256=digest(checkpoint), resolution=[2048, 1024],
                         crop=[1024, 1024], stride=[512, 512], outer_batched=False)
        cache = output / (name + '.npz')
        metadata = output / (name + '.cache.json')
        saved_signature = json.loads(metadata.read_text())['signature'] if metadata.exists() else {}
        # Older reports fingerprinted the plotting script as well as inference.
        # Plot/gradient-report additions do not invalidate saved label predictions.
        saved_signature.pop('script_sha256', None)
        cached = not args.refresh and cache.exists() and saved_signature == signature
        if cached:
            saved = json.loads(metadata.read_text())
            predictions = list(np.load(str(cache))['predictions'])
            elapsed, peak = saved['seconds'], saved['peak_gpu_MiB']
            training_seed = saved['training_seed']
            print('Using verified prediction cache', flush=True)
        else:
            cfg = update_legacy_cfg(Config.fromfile(str(config)))
            training_seed = cfg.get('seed')
            cfg.model.test_cfg.update(mode='slide', batched_slide=False, crop_size=(1024, 1024), stride=(512, 512))
            cfg.data.test.pipeline[1]['img_scale'] = (2048, 1024)
            model = init_segmentor(cfg, checkpoint=None, device='cpu')
            saved = torch.load(str(checkpoint), map_location='cpu')
            state = saved.get('state_dict', saved)
            state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
            state = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            del saved, state
            model.CLASSES, model.PALETTE = CityscapesDataset.CLASSES, CityscapesDataset.PALETTE
            model.cuda().eval()
            inference_segmentor(model, str(selected[0]))  # Same warmup for each model.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            predictions = []
            begin = time.perf_counter()
            for index, image in enumerate(selected):
                prediction = inference_segmentor(model, str(image))[0]
                assert prediction.shape == (1024, 2048)
                assert prediction.min() >= 0 and prediction.max() < 19
                predictions.append(prediction.astype(np.uint8))
                print('{} {}/{} elapsed {:.1f}s'.format(name, index + 1, len(selected), time.perf_counter() - begin), flush=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            np.savez_compressed(str(cache), predictions=np.stack(predictions))
            metadata.write_text(json.dumps(dict(signature=signature, seconds=elapsed, peak_gpu_MiB=peak,
                                                training_seed=training_seed), indent=2))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        confusion = np.zeros((19, 19), dtype=np.int64)
        image_scores[name] = []
        for path, prediction in zip(selected, predictions):
            label_path = gt_root / str(path.relative_to(rgb_root)).replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
            label = np.array(Image.open(label_path))
            valid = label != 255
            assert np.all(label[valid] < 19)
            encoded = 19 * label[valid].astype(np.int64) + prediction[valid]
            image_confusion = np.bincount(encoded, minlength=19 ** 2).reshape(19, 19)
            confusion += image_confusion
            image_tp = np.diag(image_confusion).astype(float)
            image_union = image_confusion.sum(0) + image_confusion.sum(1) - image_tp
            with np.errstate(divide='ignore', invalid='ignore'):
                image_scores[name].append(float(np.nanmean(image_tp / image_union) * 100))
        true_positive = np.diag(confusion).astype(float)
        ground_truth = confusion.sum(1)
        union = ground_truth + confusion.sum(0) - true_positive
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / union
            recall = true_positive / ground_truth
        metrics = dict(model=name, images=len(selected), training_seed=training_seed,
                       mIoU_percent=float(np.nanmean(iou) * 100),
                       pixel_accuracy_percent=float(true_positive.sum() / confusion.sum() * 100),
                       mean_class_accuracy_percent=float(np.nanmean(recall) * 100),
                       seconds_per_image_including_IO=elapsed / len(selected), peak_gpu_MiB=peak)
        summaries.append(metrics)
        per_classes[name] = [dict(class_name=c, IoU_percent=float(i * 100) if np.isfinite(i) else None,
                                 recall_percent=float(r * 100) if np.isfinite(r) else None,
                                 gt_pixels=int(g)) for c, i, r, g in zip(classes, iou, recall, ground_truth)]
        example_predictions[name] = [predictions[i] for i in example_indices]
        print(json.dumps(metrics, indent=2), flush=True)
        del predictions

    # Direct functional ablation: identical fixed clean-image pseudo-labels and
    # student weights; compare the MIC term's gradient at multiplier one vs zero.
    # This is a 256px execution probe, not an accuracy evaluation or optimizer step.
    print('MIC gradient probe (no weight update)', flush=True)
    from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule
    probe_cfg = Config.fromfile(str(ROOT / 'uv/seg/mic_gta_cityscapes_smoke.py'))
    probe_cfg.uda.max_iters = 2
    student = init_segmentor(probe_cfg, checkpoint=None, device='cpu')
    checkpoint = torch.load(str(models[-1][2]), map_location='cpu')
    state = {k[6:] if k.startswith('model.') else k: v for k, v in checkpoint['state_dict'].items()}
    student.load_state_dict(state, strict=True)
    del checkpoint, state
    student.cuda().eval()
    student.automatic_debug, student.debug = False, False
    mean = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device='cuda').view(1, 3, 1, 1)
    arrays = [np.array(Image.open(p).convert('RGB').resize((256, 256))) for p in selected[:2]]
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().cuda()
    images = (images - mean) / std
    metas = [dict(img_norm_cfg=dict(mean=mean.cpu().numpy().reshape(3),
                                    std=std.cpu().numpy().reshape(3), to_rgb=True)) for _ in arrays]
    with torch.no_grad():
        pseudo_labels = student.encode_decode(images, metas).argmax(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    student.train()
    mic = MaskingConsistencyModule(require_teacher=False, cfg=probe_cfg.uda)
    weights = torch.ones_like(pseudo_labels, dtype=torch.float32)
    losses = mic(student, images, metas, pseudo_labels[:, None], images, metas, None, pseudo_labels, weights)
    loss = sum(value for key, value in losses.items() if 'loss' in key)
    parameters = tuple(student.decode_head.parameters())
    on = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    off = torch.autograd.grad(loss * 0, parameters, allow_unused=True)
    norm = lambda gradients: float(torch.sqrt(sum(g.detach().square().sum() for g in gradients if g is not None)).item())
    gradient_probe = dict(input_images=len(arrays), resolution=[256, 256],
                          masked_consistency_loss=float(loss.item()),
                          decoder_gradient_L2_MIC_on=norm(on), decoder_gradient_L2_MIC_off=norm(off),
                          optimizer_updates=0,
                          note='Functional gradient ablation with fixed clean-image pseudo-labels; confidence filtering disabled for this probe.')
    assert np.isfinite(gradient_probe['masked_consistency_loss']) and gradient_probe['masked_consistency_loss'] > 0
    assert gradient_probe['decoder_gradient_L2_MIC_on'] > 0
    assert gradient_probe['decoder_gradient_L2_MIC_off'] == 0
    del student, parameters, on, off, losses, loss, images
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(gradient_probe, indent=2), flush=True)

    report = dict(subset_seed=args.seed, selected_images=names, models=summaries,
                  delta_mIoU_percentage_points=summaries[1]['mIoU_percent'] - summaries[0]['mIoU_percent'],
                  per_class=per_classes, mic_gradient_probe=gradient_probe,
                  examples=[dict(image=names[i], subset_index=i,
                                 HRDA_mIoU_percent=image_scores['HRDA_no_MIC'][i],
                                 MIC_mIoU_percent=image_scores['HRDA_MIC'][i]) for i in example_indices],
                  limitation='Published independently trained checkpoints: baseline seed 1, MIC seed 2. Fixed subset, not a same-seed causal ablation or full validation benchmark.')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with (output / 'per_class.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class', 'HRDA_IoU_percent', 'MIC_IoU_percent', 'delta_percentage_points'])
        for b, m in zip(per_classes['HRDA_no_MIC'], per_classes['HRDA_MIC']):
            bv, mv = b['IoU_percent'], m['IoU_percent']
            writer.writerow([b['class_name'], bv, mv, mv - bv if bv is not None and mv is not None else None])
    with (output / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', 'HRDA_mIoU_percent', 'MIC_mIoU_percent', 'delta_percentage_points'])
        for name, baseline, mic_score in zip(names, image_scores['HRDA_no_MIC'], image_scores['HRDA_MIC']):
            writer.writerow([name, baseline, mic_score, mic_score - baseline])
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def color(mask):
        result = np.zeros((*mask.shape, 3), dtype=np.uint8)
        valid = mask < 19
        result[valid] = palette[mask[valid]]
        return result
    legend = [Patch(color=p / 255, label=c) for c, p in zip(classes, palette)]
    overview, rows = plt.subplots(args.examples, 4, figsize=(20, 3.3 * args.examples + 1.5), squeeze=False)
    for number, index in enumerate(example_indices):
        original = np.array(Image.open(selected[index]).convert('RGB'))
        label = np.array(Image.open(gt_root / names[index].replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')))
        panels = [original, color(label)]
        titles = ['Real Cityscapes image', 'Ground truth (black = ignored)']
        for key, title in [('HRDA_no_MIC', 'HRDA without MIC'), ('HRDA_MIC', 'HRDA + MIC')]:
            panels.append((0.5 * original + 0.5 * color(example_predictions[key][number])).astype(np.uint8))
            titles.append('{}: {:.2f}% image mIoU'.format(title, image_scores[key][index]))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.2))
        for axis, panel, title in zip(axes.flat, panels, titles):
            axis.imshow(panel); axis.set_title(title); axis.axis('off')
        fig.suptitle('Example {}: {}\nSelected by position, independently of model scores'.format(number + 1, names[index]), fontsize=11)
        fig.legend(handles=legend, loc='lower center', ncol=7, fontsize=8, frameon=False)
        fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.25, wspace=0.08)
        fig.savefig(str(output / 'comparison_{:02d}.png'.format(number + 1)), dpi=160)
        plt.close(fig)
        for axis, panel, title in zip(rows[number], panels, titles):
            axis.imshow(panel); axis.set_title(title, fontsize=10); axis.axis('off')
        rows[number, 0].text(0, -0.06, names[index], transform=rows[number, 0].transAxes, fontsize=8, va='top')
    overview.suptitle('{} evenly spaced examples from the evaluated subset; independent checkpoints\nImage mIoU uses classes with nonempty union; scores are not the aggregate dataset mIoU'.format(args.examples), fontsize=13)
    overview.legend(handles=legend, loc='lower center', ncol=10, fontsize=8, frameon=False)
    overview.subplots_adjust(top=0.93, bottom=0.08, hspace=0.35, wspace=0.06)
    overview.savefig(str(output / 'comparison.png'), dpi=160)
    plt.close(overview)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for axis, field, title in zip(axes,
            ('mIoU_percent', 'pixel_accuracy_percent', 'mean_class_accuracy_percent'),
            ('Mean IoU', 'Pixel accuracy', 'Mean class accuracy')):
        values = [m[field] for m in summaries]
        bars = axis.bar(['HRDA', 'HRDA + MIC'], values, color=['#75899d', '#24795d'], width=0.55)
        axis.set_ylim(0, 100); axis.set_title(title); axis.set_ylabel('%')
        axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)
        for bar, value in zip(bars, values): axis.text(bar.get_x() + bar.get_width()/2, value + 1, '{:.2f}'.format(value), ha='center')
    fig.suptitle('Same {} validation images; independent published checkpoints (different seeds)'.format(args.images))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(output / 'metrics.png'), dpi=160)
    plt.close(fig)
    print('PASS: paired inference comparison, CSV/JSON metrics, and figures saved to', output)
    print('MIC minus HRDA: {:+.2f} mIoU percentage points'.format(report['delta_mIoU_percentage_points']))


if __name__ == '__main__':
    main()
