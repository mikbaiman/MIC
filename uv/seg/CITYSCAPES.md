# Verified Cityscapes data and real-image quick check

The supplied `dataset/gtFine/` and `dataset/leftImg8bit/` folders are the correct
Cityscapes fine annotations and RGB images for segmentation. They match the
supplied `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` archives:

| Split | RGB images | Label-ID masks | Polygon annotations |
| --- | ---: | ---: | ---: |
| train | 2,975 | 2,975 | 2,975 |
| val | 500 | 500 | 500 |
| test | 1,525 | 1,525 | 1,525 |

All 25,000 original extracted image/annotation files passed size and CRC checks
against their archive entries. Every image has a corresponding label-ID mask and
polygon JSON. Sample PNGs passed integrity checks and have 2048 x 1024 resolution.
The machine-readable report is `runs/cityscapes-validation/report.json`.

## Preparation already completed

Relative links expose the existing folders through the repository's expected
layout, without moving or duplicating the downloads:

```text
seg/data/cityscapes/
├── gtFine -> ../../../dataset/gtFine
├── leftImg8bit -> ../../../dataset/leftImg8bit
├── train.txt
├── val.txt
├── test.txt
├── sample_class_stats.json
├── sample_class_stats_dict.json
└── samples_with_class.json
```

The original converter was run successfully:

```bash
bash uv/run.sh seg tools/convert_datasets/cityscapes.py data/cityscapes --nproc 2
```

It generated 5,000 `*_gtFine_labelTrainIds.png` files alongside the original
annotations. Selected train/val masks were checked to use only train IDs 0–18
plus ignored ID 255. The sampled test mask is entirely ignored: the benchmark's
test ground truth is withheld, so use **val** for local mIoU evaluation.

## Real-data test already completed successfully

`cityscapes_quickcheck.py` configures a small ResNet-18/FCN segmentation model,
eight real training images and eight real validation images. The model starts
from random weights and performs two supervised training updates. This verifies
the repository's real dataset loading, augmentation, GPU gradients, validation,
checkpoint serialization/reload, evaluation, and prediction output.

This is a **supervised pipeline sanity check**, not the MIC(HRDA) domain-adaptation
experiment or a model-quality benchmark. No additional dataset/weight downloads
were needed. The observed quick-check mIoU was 0.7%, which is expected to be poor
after two updates from random initialization. Per-class NaNs indicate classes
absent from these few validation images.

Run these commands from the repository root to repeat the check:

```bash
bash uv/run.sh seg tools/train.py ../uv/seg/cityscapes_quickcheck.py \
  --work-dir ../uv/seg/runs/cityscapes-quickcheck --seed 0

bash uv/run.sh seg tools/test.py ../uv/seg/cityscapes_quickcheck.py \
  ../uv/seg/runs/cityscapes-quickcheck/iter_2.pth --eval mIoU \
  --show-dir ../uv/seg/runs/cityscapes-quickcheck/predictions --opacity 0.5
```

These commands replace the quick-check run's outputs if repeated. Outputs:

- Checkpoint: `uv/seg/runs/cityscapes-quickcheck/iter_2.pth`.
- Configuration and logs: `uv/seg/runs/cityscapes-quickcheck/`.
- Eight real-image overlays: `uv/seg/runs/cityscapes-quickcheck/predictions/`.
- Observed training peak memory: approximately 2.3 GB on the RTX 3060 8 GB.

The subset lists are explicitly recorded in `cityscapes_train_subset.txt` and
`cityscapes_val_subset.txt`. The validation subset remains separate from training.

## What is needed for the research experiment?

Update: GTA is now assembled/preprocessed under `dataset/gta/`, strict-compatible
MiT-B5 weights are available under `seg/pretrained/`, and real MiT-B5/HRDA/MIC
training has passed with a small-GPU configuration. See the real-MIC section of
`uv/README.md` for the current repeatable commands and experimental differences.

You do **not** need to download Cityscapes again.

- **Evaluate a pretrained MIC model:** obtain a published MIC(HRDA)
  GTA-to-Cityscapes checkpoint and its matching configuration from
  `seg/README.md`. Your prepared Cityscapes validation images/labels are enough;
  GTA is not required for evaluation. The quick-check checkpoint above is not
  the pretrained research model.
- **Train the documented GTA-to-Cityscapes MIC experiment:** additionally obtain
  the GTA images and labels from
  <https://download.visinf.tu-darmstadt.de/data/from_games/>, place them under
  `seg/data/gta/images/` and `seg/data/gta/labels/`, and preprocess them with
  `tools/convert_datasets/gta.py`. Obtain the SegFormer MiT-B5 ImageNet weights
  (`mit_b5.pth`) using the links in `seg/README.md` and place them under
  `seg/pretrained/mit_b5.pth`. Full HRDA training has not been memory-tested on
  this 8 GB GPU; the small check's memory usage does not establish that it fits.
- **Run the detection adaptation experiment:** Cityscapes also needs its COCO
  annotation conversion, and the target dataset must be Foggy Cityscapes. See
  `det/README.md`; the segmentation check does not establish detection readiness.
