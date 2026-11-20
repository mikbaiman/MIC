# uv environments and compatibility checks for MIC

## Real GTA to Cityscapes MiT-B5 / HRDA / MIC is now verified

The current downloads live under `dataset/`. All 24,966 GTA images and labels
are assembled under `dataset/gta/` (image links preserve the original extracted
parts), and `seg/data/gta` points there. `seg/data/cityscapes/{gtFine,leftImg8bit}`
now point to the corresponding folders under `dataset/`. GTA archive 05 labels
was missing and was downloaded from the official dataset host; all ten label
archives were extracted with CRC checks. The original GTA converter generated
all 24,966 train-ID masks and rare-class sampling statistics.

The suggested public weights at
<https://huggingface.co/smp-hub/mit_b5.imagenet> work. Revision
`0851a37f3f718608d9b3cc8ff4cafb81f07374a1` was downloaded to
`dataset/mit_b5.huggingface.safetensors`. `prepare_mit_b5.py` reads its F32 tensors
without adding dependencies to the legacy environment, verifies all 1,052 encoder
keys/shapes/finiteness, loads them strictly, and writes
`seg/pretrained/mit_b5.pth`. Only the unused ImageNet classifier is omitted. Every
encoder tensor is bit-for-bit identical to the supplied `dataset/mit_b5.pth`.
No backbone tensor names or shapes are changed.

The supplied `dataset/gtaHR2csHR_mic_hrda_650a8.tar.gz` was extracted into
`dataset/checkpoints/`. All 1,126 trained model tensors match its supplied config
strictly. Its original-resolution evaluation on the fixed eight-image validation
subset completed with 68.16% mIoU and 96.52% pixel accuracy. This is a subset
result, not the full 500-image Cityscapes benchmark. Overlays and logs are in
`uv/seg/runs/mic-pretrained-eval/`.

The real training check uses MiT-B5, HRDA, EMA pseudo-labeling, ClassMix, ImageNet
feature distance, and MIC on the actual GTA/Cityscapes data. Two updates and
validation completed, with a peak of 4,235 MiB reported by the training logger.
The saved checkpoint's 1,052 student encoder tensors all changed; its EMA teacher
differs from the student and its frozen ImageNet encoder remains unchanged.
Source, mixed-image, and masked-image losses were positive and finite. Feature
distance was zero for the two crops with no eligible thing-class feature cells;
an empty selection now returns differentiable zero instead of a NaN mean. Two
regression tests cover that case and the unchanged nonempty L2 calculation.

Important configuration differences:

- `seg/configs/mic/gtaHR2csHR_mic_hrda.py` remains the original 1024-pixel
  experiment; it has not been trained at full resolution on this 8 GB card.
- `uv/seg/mic_gta_cityscapes_8gb.py` retains the original MiT-B5, HRDA, EMA,
  feature-distance, ClassMix, and MIC components but uses 256-pixel training
  crops, 128-pixel detail crops, smaller resize scales, a lower rare-class crop
  fraction, sequential outer inference crops, and no extra loader workers.
  It retains the original confidence threshold and 40,000-iteration schedule.
  It is a separate small-GPU experiment and should not be reported as reproducing
  the paper's training protocol.
- `uv/seg/mic_gta_cityscapes_smoke.py` runs two updates, disables LR warmup,
  and temporarily uses confidence threshold zero to ensure a nonzero MIC signal
  before the decoder has learned. It evaluates eight validation images. These
  smoke-only settings are not inherited by the long-training config.
- `uv/seg/hrda_gta_cityscapes_8gb_baseline.py` is a matched long-training control
  with only MIC disabled. Run matched seeds and evaluate the full 500 validation
  images to measure MIC's benefit; two-update scores are not evidence of benefit.

### Commands to independently repeat the real MIC check

Run from the repository root. The first two commands recheck cached downloads;
they do not download anything or change the original image files:

```bash
uv/seg/.venv/bin/python uv/prepare_mic_data.py
uv/seg/.venv/bin/python uv/prepare_mit_b5.py
```

The missing archive and pinned weight download have already been completed. To
obtain them again on a fresh checkout with the other dataset parts present:

```bash
curl -fL --retry 2 -o dataset/mit_b5.huggingface.safetensors \
  https://huggingface.co/smp-hub/mit_b5.imagenet/resolve/0851a37f3f718608d9b3cc8ff4cafb81f07374a1/model.safetensors
curl -fL --retry 2 -o dataset/05_labels.zip \
  https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip
```

If the converted GTA masks/statistics need rebuilding (about 3.5 minutes on the
test machine):

```bash
bash uv/run.sh seg tools/convert_datasets/gta.py data/gta --nproc 8
```

Repeat training into a separate user-verification directory, audit the checkpoint,
export a compact student without optimizer/teacher state, then reload/evaluate:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash uv/run.sh seg tools/train.py \
  ../uv/seg/mic_gta_cityscapes_smoke.py \
  --work-dir ../uv/seg/runs/mic-your-run --seed 0

uv/seg/.venv/bin/python uv/verify_mic_training.py \
  uv/seg/runs/mic-your-run --export-student

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash uv/run.sh seg tools/test.py \
  ../uv/seg/mic_gta_cityscapes_smoke.py \
  ../uv/seg/runs/mic-your-run/student_2.pth --eval mIoU \
  --show-dir ../uv/seg/runs/mic-your-run/predictions
```

Expected: a PASS from the audit, two saved training updates, finite positive
source/mixed/masked losses, an unchanged frozen ImageNet network, and eight
prediction images. Small-crop feature-distance loss may be zero; absent-class
metric NaNs in a small validation subset are separate from non-finite training
losses and do not indicate a training failure.

Evaluate the supplied research checkpoint on the same eight images:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash uv/run.sh seg tools/test.py \
  ../uv/seg/mic_checkpoint_eval.py \
  ../dataset/checkpoints/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth \
  --eval mIoU --show-dir ../uv/seg/runs/mic-pretrained-your-run/predictions \
  --options data.test.split=../../../uv/seg/cityscapes_val_subset.txt
```

For all 500 validation images, omit the `--options data.test.split=...` override.
Consider `--eval-options efficient_test=True` to store prediction arrays on disk
instead of keeping all full-resolution predictions in RAM. Full validation takes
substantially longer than this eight-image check. Sequential outer crop batching
keeps the original inference resolution/crop sizes; it is slower than batching
all outer crops.

The long small-GPU run is ready to launch explicitly (not started automatically):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash uv/run.sh seg tools/train.py \
  ../uv/seg/mic_gta_cityscapes_8gb.py \
  --work-dir ../uv/seg/runs/mic-8gb-s0 --seed 0
```

Its matching no-MIC control uses
`../uv/seg/hrda_gta_cityscapes_8gb_baseline.py` and a separate work directory.

Environment configuration, scripts, locks, environments, and native build
products live under `uv/`. Work is on branch `mikhail`; the original dependency
versions remain pinned. The previous environments are not used or modified.

Targeted source fixes support actual training and evaluation:

- Both classification trainers schedule SAM's inner SGD optimizer, which performs
  the real update, rather than warning about an unobserved outer `step()` call.
- Both classification trainers save a first best checkpoint even at 0% accuracy.
- Classification logging restores stdout/stderr without closing borrowed terminal
  streams, avoiding interpreter shutdown failure (exit code 120).
- Detection skips training-only domain losses in evaluation and EMA pseudo-labeling.
- Detection's instance domain head processes all four feature-pyramid levels.
- The uv runners default to Matplotlib's noninteractive Agg backend, allowing
  segmentation debug images to be saved without a Tk/Tcl desktop installation.

Regression tests and full CLI fixture checks accompany these changes. Research
models, dataset definitions, experiment settings, and pinned versions are retained.

## Original versions

The tasks have incompatible requirements and therefore use separate environments.
This uses uv's pip interface rather than combining the tasks into one project.

| Profile | Python | PyTorch | torchvision | Other original pins |
| --- | --- | --- | --- | --- |
| `cls` | 3.8.5 | 1.9.1+cu111 | 0.10.1+cu111 | timm 0.5.4, Kornia 0.5.8, W&B 0.12.2 |
| `seg` | 3.8.5 | 1.7.1+cu110 | 0.8.2+cu110 | mmcv-full 1.3.7, timm 0.3.2, NumPy 1.19.2 |
| `det` | 3.9.7 | 1.12.0+cu113 | 0.13.0+cu113 | timm 0.6.11, Kornia 0.5.8, NumPy 1.22.3 |

Classification and segmentation `requirements.in` files include the original
requirements files directly. Classification selects the official CUDA 11.1 builds
of its original torch/torchvision versions; segmentation retains its explicitly
specified CUDA 11.0 builds. Detection follows `det/INSTALL.md`, with applicable
additional version pins from `det/environment.yml`; it does not install unrelated
Conda/notebook packages. COCO API is resolved from PyPI because INSTALL.md's Git
installation does not fix a release. Its resolved release is recorded in the lock.

Dependencies omitted or left unpinned upstream are fixed by the lock files.
Classification additionally declares its imported analysis libraries and keeps
NumPy below 1.24 because the original code uses `np.float`. The historical uv
Python 3.8.5 build uses LibreSSL, so classification/segmentation require urllib3
below 2. Setuptools remains available for legacy runtime imports. Isolated build
constraints keep PyYAML 5.4.1 compatible with Cython below 3 without patching that
dependency. Segmentation uses OpenMMLab's official matching prebuilt MMCV wheel.

Target: Linux x86-64, NVIDIA GPU. Setup was tested with uv 0.13.7. The locks are
for these Python/platform combinations, not universal Windows/macOS environments.
Python distributions, package caches, environments, build products, and smoke
outputs are ignored by `uv/.gitignore`. All three installations used approximately
12 GB of shared package cache on the test machine, excluding datasets/checkpoints.

## Set up or restore an environment

Run from the repository root:

```bash
bash uv/setup.sh cls
bash uv/setup.sh seg
bash uv/setup.sh det
```

Only run the command for the task you want. These download the exact Python
version with uv and sync the committed package versions into `uv/PROFILE/.venv`.
They never install into the old environments. These commands are repeatable.
No dataset or model weights are downloaded by setup.

The environments are already installed in this checkout. To activate one manually:

```bash
source uv/seg/.venv/bin/activate
```

The runner below is more convenient because it also selects the original expected
working directory, puts the selected Python on PATH for child processes, and
exposes detection's isolated native build. Classification scripts run inside
`cls/examples`; segmentation inside `seg`; detection inside `det`. Relative
dataset/config/checkpoint paths follow those directories.

```bash
bash uv/run.sh cls cdan_mcc_sdat_masking.py --help
bash uv/run.sh seg tools/train.py --help
bash uv/run.sh det tools/train_net.py --help
```

Detection also needs its original C++/CUDA extension. A CPU build is already
available in this checkout. To rebuild it without touching `det/`:

```bash
bash uv/build-det.sh cpu
```

For a GPU build, install a CUDA 11.3 development toolkit and a compatible host
compiler, expose `nvcc` on PATH and set `CUDA_HOME` if necessary, then run:

```bash
bash uv/build-det.sh cuda
```

No CUDA toolkit or system package was installed by this setup. The NVIDIA driver's
`nvidia-smi` CUDA value is the driver capability, not the installed toolkit version.
The PyTorch wheels include their CUDA runtimes; those suffice for the tested
classification and segmentation operations. Building detection's CUDA extension
requires the additional compiler/toolkit. The CUDA extension build is unverified.

## First test: no dataset needed

From the repository root:

```bash
bash uv/check.sh
```

To test your GPU and the actual training/checkpoint/evaluation CLIs as well:

```bash
bash uv/check.sh --cuda --integration
```

The integration checks generate tiny image/label fixtures under ignored
`uv/cls/runs/integration-*` and `uv/seg/runs/integration-*` directories. They train
small ResNet-18 models for two iterations, save/reload checkpoints, and run the
real evaluation entry points. Classification checks both baseline and MIC;
segmentation checks DACS self-training, EMA pseudo-labels, ClassMix, MIC, runner
hooks, and mIoU evaluation. No public datasets or pretrained weights are required.
Synthetic scores are plumbing checks, not accuracy benchmarks.
The deliberately tiny segmentation fixture contains only three ground-truth
class IDs within the original 19-class Cityscapes output layout. Per-class NaNs
for absent classes and low aggregate accuracy are expected for these untrained
two-iteration models; successful execution is indicated by the final PASS lines.

You can also run individual model checks:

```bash
uv/cls/.venv/bin/python uv/smoke.py cls --device cpu
uv/seg/.venv/bin/python uv/smoke.py seg --device cpu
uv/det/.venv/bin/python uv/smoke.py det --device cpu
```

On a terminal with NVIDIA driver access:

```bash
uv/cls/.venv/bin/python uv/smoke.py cls --device cuda
uv/seg/.venv/bin/python uv/smoke.py seg --device cuda
```

The scripts use tiny random tensors and untrained models. They download neither
images nor pretrained weights and do not measure model accuracy.

Verified on the RTX 3060 8 GB, driver 580.178.04:

| Check | Result |
| --- | --- |
| Dependency consistency in all three environments | PASS |
| All three original training entry points import and display `--help` | PASS |
| Original classification MIC `train()`, teacher, losses, two-pass SAM, inference | PASS, CPU and CUDA |
| Original segmentation MIC config and block masking, small ResNet/FCN segmentor forward/backward, compiled MMCV NMS | PASS, CPU and CUDA |
| Original detection C++ extension build | PASS, CPU |
| Default detector inference and EMA pseudo-labeling (without a bypass flag) | PASS, CPU |
| Original detection metric-logger and sampler tests | PASS, 10 tests |
| New scheduler/logger, domain-head, and feature-distance regression tests | PASS, 6 tests |
| Baseline and MIC classification training, checkpoint reload, evaluation CLIs | PASS, CUDA, generated images |
| Segmentation DACS+MIC training, validation, checkpoint reload, mIoU CLI | PASS, CUDA, generated images |
| Detection CUDA extension / GPU detector training | Not tested: no `nvcc` installed |
| Full HRDA/MiT-B5 MIC training or dataset evaluation | Not tested |

The detector previously raised `IndexError` because it sent filtered evaluation
boxes and pre-filtering features into training-only domain heads. Evaluation and
teacher pseudo-labeling now skip those losses; the smoke test uses the default
public forward call and checks teacher outputs as well. Regression tests also
verify outputs and gradients at all four pyramid levels. Classification tests
verify the actual SAM updates and learning-rate sequence and reject recurrence of
the scheduler-order warning. WILDS' outdated-version notice is informational; its
documented version is intentionally retained.

GPU access is unavailable inside this app's restricted sandbox, but the same
classification/segmentation tests passed when executed with host driver access.
If CUDA is unavailable, first check `nvidia-smi` from your normal terminal.

An 8 GB GPU is enough for the verified small checks. The original HRDA experiment
uses 1024-pixel crops, MiT-B5, and teacher models; full training memory requirements
were not established. Do not infer full-training feasibility from the smoke tests.

## Next: real data

### Classification

Use OfficeHome or VisDA-2017. The original loaders attempt automatic downloads
from URLs embedded in the source. If those old links fail, download the datasets
manually as described in `cls/README.md`, including the expected image-list files.
No datasets have been downloaded here.

A small real-data pipeline check (not meaningful accuracy training) is:

```bash
bash uv/run.sh cls cdan_mcc_sdat_masking.py data/office-home \
  -d OfficeHome -s Ar -t Cl -a resnet18 --scratch \
  -b 2 -j 0 --epochs 1 --iters-per-epoch 1 --print-freq 1 \
  --seed 0 --alpha 0.9 --pseudo_label_weight prob \
  --mask_block_size 32 --mask_ratio 0.5 \
  --log ../../uv/cls/runs/officehome-smoke
```

This attempts dataset downloads into `cls/examples/data/office-home`. `--scratch`
avoids pretrained-weight downloads, W&B is off unless `--log_results` is supplied,
and results go under `uv/cls/runs`. It still validates on the target dataset; one
training iteration does not mean only one validation batch. The original loaders
check all OfficeHome domain archives, not just the two selected domains.

### Segmentation: the original MIC(HRDA) experiment

The locally supplied Cityscapes archives have now been verified and preprocessed.
A real-image GPU quick check (training, validation, checkpoint reload, and eight
prediction overlays) completed successfully. See
[the Cityscapes quick-check guide](seg/CITYSCAPES.md) for the recorded verification,
repeatable commands, and the distinction from the full MIC research experiment.

For training GTA to Cityscapes, manually obtain:

- Cityscapes `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from
  <https://www.cityscapes-dataset.com/downloads/>; an account may be required.
  Extract under `seg/data/cityscapes/`.
- GTA images and labels from <https://download.visinf.tu-darmstadt.de/data/from_games/>.
  Extract under `seg/data/gta/images/` and `seg/data/gta/labels/`.
- MiT-B5 ImageNet weights (`mit_b5.pth`) from the SegFormer links in
  `seg/README.md` / `seg/tools/download_checkpoints.sh`; place under
  `seg/pretrained/mit_b5.pth`.

Preprocess only the datasets actually downloaded:

```bash
bash uv/run.sh seg tools/convert_datasets/cityscapes.py data/cityscapes --nproc 2
bash uv/run.sh seg tools/convert_datasets/gta.py data/gta --nproc 2
```

Then the unchanged documented training command is:

```bash
bash uv/run.sh seg run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py
```

This starts the full original experiment and writes generated configs, logs, and
checkpoints under `seg/`; it is not an 8 GB memory-tested quick check. Do not run it
just to check installation. Synthia, ACDC, and Dark Zurich are optional alternative
experiments and are not required for GTA to Cityscapes.

For evaluation of an already trained GTA-to-Cityscapes checkpoint, GTA and the
ImageNet initialization weights are not needed: obtain the checkpoint and its
matching config from `seg/README.md`, plus Cityscapes validation images/labels,
preprocess Cityscapes, and run:

```bash
bash uv/run.sh seg tools/test.py /absolute/path/to/matching_config.py \
  /absolute/path/to/checkpoint.pth --eval mIoU
```

This evaluation was not run here. A random-image inference smoke test does not
replace evaluation with a trained checkpoint and labels.

### Detection

For the documented Cityscapes to Foggy Cityscapes experiment, obtain both datasets
from <https://www.cityscapes-dataset.com/> and arrange them under `det/datasets/`
as specified in `det/README.md`. Convert annotations with the two scripts in
`det/tools/cityscapes/`. The uv runner exposes the original polygon helper through
the cityscapesScripts package search path, replacing INSTALL.md's manual copy step
without modifying either source tree.

Before full detection training, install/build the CUDA extension. CPU inference
and regression checks are not a claim that the complete GPU detection experiment
has been validated on real data.

## Maintaining the locks

Normal setup uses the existing locks. To deliberately re-resolve unspecified
transitive packages without editing the original requirements:

```bash
bash uv/lock.sh cls
bash uv/lock.sh seg
bash uv/lock.sh det
```

Re-run the corresponding setup and smoke tests after relocking. See uv's pip
interface documentation at <https://docs.astral.sh/uv/pip/>.
