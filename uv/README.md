# uv environments and compatibility checks for MIC

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
| New scheduler/logger and domain-head regression tests | PASS, 4 tests |
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
