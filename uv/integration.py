"""Run the real training/evaluation CLIs against tiny generated image datasets."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def run(profile, script, *arguments):
    subprocess.run([sys.executable, "-B", str(ROOT / "uv/run.py"), profile,
                    script, *map(str, arguments)], check=True)


def image(path, index, labels=False):
    import numpy as np
    from PIL import Image

    y, x = np.indices((64, 64))
    pixels = ((x // 16 + y // 16 + index) % 3).astype("uint8")
    if not labels:
        pixels = np.stack([pixels * 80, (x * 4).astype("uint8"),
                           (y * 4).astype("uint8")], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def classification(directory, device):
    data = directory / "data"
    (data / "image_list").mkdir(parents=True)
    for split, list_name in (("train", "train.txt"), ("validation", "validation.txt")):
        lines = []
        for index in range(2):
            path = data / split / (str(index) + ".png")
            image(path, index)
            lines.append("{} 0\n".format(path.relative_to(data)))
        (data / "image_list" / list_name).write_text("".join(lines))
    for script in ("cdan_mcc_sdat.py", "cdan_mcc_sdat_masking.py"):
        output = directory / Path(script).stem
        common = [data, "-d", "VisDA2017", "-s", "Synthetic", "-t", "Real",
                  "-a", "resnet18", "--scratch", "--bottleneck-dim", "16",
                  "-b", "2", "-j", "0", "--seed", "0",
                  "--gpu", "0" if device == "cuda" else ""]
        run("cls", script, *common, "--epochs", 1, "--iters-per-epoch", 2,
            "--print-freq", 1, "--log", output)
        checkpoint = output / "checkpoints/best.pth"
        if not checkpoint.is_file():
            raise RuntimeError("Training did not produce {}".format(checkpoint))
        run("cls", "eval.py", *common, "--phase", "test", "--weight_path", checkpoint)
    print("PASS: baseline and MIC CLIs trained, validated, saved/reloaded checkpoints, and evaluated.")


def segmentation(directory, device):
    if device != "cuda":
        raise ValueError("The original segmentation CLI requires CUDA; use the CPU tensor smoke test instead.")
    import mmcv
    sys.path.insert(0, str(ROOT / "seg"))
    from mmseg.core.evaluation.class_names import cityscapes_classes, cityscapes_palette

    for domain in ("source", "target"):
        for index in range(2):
            image(directory / domain / "images" / (str(index) + ".jpg"), index)
            image(directory / domain / "labels" / (str(index) + ".png"), index, labels=True)
    norm = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations"),
                      dict(type="RandomFlip", prob=0.0),
                      dict(type="Normalize", **norm), dict(type="DefaultFormatBundle"),
                      dict(type="Collect", keys=["img", "gt_semantic_seg"])]
    test_pipeline = [dict(type="LoadImageFromFile"), dict(
        type="MultiScaleFlipAug", img_scale=(64, 64), flip=False,
        transforms=[dict(type="Resize", keep_ratio=True), dict(type="RandomFlip"),
                    dict(type="Normalize", **norm), dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"])])]

    def dataset(domain, pipeline):
        return dict(type="CustomDataset", data_root=str(directory / domain),
                    img_dir="images", ann_dir="labels", pipeline=pipeline,
                    classes=cityscapes_classes(), palette=cityscapes_palette())

    config = dict(
        _base_=[str(ROOT / "seg/configs/_base_/default_runtime.py"),
                str(ROOT / "seg/configs/_base_/uda/dacs.py")],
        model=dict(type="EncoderDecoder", pretrained=None,
                   backbone=dict(type="ResNet", depth=18, num_stages=4, out_indices=(3,),
                                 norm_cfg=dict(type="BN", requires_grad=True), norm_eval=False),
                   decode_head=dict(type="FCNHead", in_channels=512, in_index=0,
                                    channels=16, num_convs=1, concat_input=False,
                                    dropout_ratio=0, num_classes=19,
                                    norm_cfg=dict(type="BN", requires_grad=True),
                                    align_corners=False,
                                    loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)),
                   train_cfg=dict(), test_cfg=dict(mode="whole")),
        uda=dict(mask_mode="separatetrgaug", mask_lambda=1,
                 mask_generator=dict(type="block", mask_ratio=0.7, mask_block_size=16, _delete_=True),
                 pseudo_threshold=0.0),
        data=dict(samples_per_gpu=2, workers_per_gpu=0,
                  train=dict(type="UDADataset", source=dataset("source", train_pipeline),
                             target=dataset("target", train_pipeline)),
                  val=dataset("target", test_pipeline), test=dataset("target", test_pipeline)),
        optimizer=dict(type="SGD", lr=0.001, momentum=0.9), optimizer_config=None,
        lr_config=dict(policy="poly", power=1.0, min_lr=0.0, by_epoch=False),
        runner=dict(type="IterBasedRunner", max_iters=2),
        checkpoint_config=dict(by_epoch=False, interval=2),
        evaluation=dict(interval=2, metric="mIoU"),
        log_config=dict(interval=1, img_interval=1000), seed=0,
    )
    filename = directory / "config.py"
    filename.write_text(mmcv.Config(config).pretty_text)
    output = directory / "work"
    run("seg", "tools/train.py", filename, "--work-dir", output, "--seed", 0)
    checkpoint = output / "iter_2.pth"
    if not checkpoint.is_file():
        raise RuntimeError("Training did not produce {}".format(checkpoint))
    run("seg", "tools/test.py", filename, checkpoint, "--eval", "mIoU")
    print("PASS: DACS+MIC CLI trained, validated, saved/reloaded a checkpoint, and evaluated mIoU.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("cls", "seg"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(ROOT / "uv/.cache/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    parent = ROOT / "uv" / args.profile / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="integration-", dir=str(parent)))
    print("Synthetic inputs and outputs:", directory, flush=True)
    globals()["classification" if args.profile == "cls" else "segmentation"](directory, args.device)


if __name__ == "__main__":
    main()
