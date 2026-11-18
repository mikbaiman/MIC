"""Exercise original MIC code on synthetic tensors without downloads or edits."""
import argparse
import importlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "uv/.cache/matplotlib"))
os.environ.setdefault("WANDB_MODE", "disabled")


def classification(torch, device):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset

    sys.path[:0] = [str(ROOT / "cls/examples"), str(ROOT / "cls")]
    entry = importlib.import_module("cdan_mcc_sdat_masking")
    from common.vision.models import resnet18

    model = entry.ImageClassifier(resnet18(pretrained=False), 3, bottleneck_dim=16).to(device)
    discriminator = entry.DomainDiscriminator(16 * 3, 16).to(device)
    domain_loss = entry.ConditionalDomainAdversarialLoss(discriminator).to(device)
    teacher = entry.EMATeacher(model, alpha=0.9, pseudo_label_weight="prob").to(device)
    mask = entry.Masking(16, 0.5, 0, 0, False, (0, 0, 0), (1, 1, 1))
    optimizer = entry.SAM(model.get_parameters(), torch.optim.SGD, rho=0.02, lr=0.001)
    adversarial_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.001)
    adversarial_scheduler = torch.optim.lr_scheduler.LambdaLR(adversarial_optimizer, lambda _: 1)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    source = entry.ForeverDataIterator(DataLoader(TensorDataset(images, labels), batch_size=2))
    target = entry.ForeverDataIterator(DataLoader(TensorDataset(images.clone(), labels), batch_size=2))
    args = SimpleNamespace(iters_per_epoch=1, trade_off=1.0, log_results=False, print_freq=1)
    entry.device = device  # The original CLI sets this global before calling train().
    before = model.head.weight.detach().clone()
    entry.train(source, target, model, teacher, domain_loss,
                entry.MinimumClassConfusionLoss(2.0), mask, optimizer,
                adversarial_optimizer, scheduler, adversarial_scheduler, 0, args)
    assert not torch.equal(before, model.head.weight.detach())
    model.eval()
    with torch.no_grad():
        output = model(images.to(device))
    assert output.shape == (2, 3) and torch.isfinite(output).all()
    print("PASS: original MIC classification train() completed a two-pass SAM update and inference.")


def segmentation(torch, device):
    sys.path.insert(0, str(ROOT / "seg"))
    import mmcv
    from mmcv.ops import nms
    from mmseg.models import build_segmentor
    from mmseg.models.utils.masking_transforms import BlockMaskGenerator

    config = mmcv.Config.fromfile(str(ROOT / "seg/configs/mic/gtaHR2csHR_mic_hrda.py"))
    assert config.uda.mask_generator.mask_ratio == 0.7
    model = build_segmentor(dict(
        type="EncoderDecoder",
        backbone=dict(type="ResNet", depth=18, num_stages=4, out_indices=(3,),
                      norm_cfg=dict(type="BN", requires_grad=True), norm_eval=False),
        decode_head=dict(type="FCNHead", in_channels=512, in_index=0, channels=16,
                         num_convs=1, concat_input=False, dropout_ratio=0,
                         num_classes=3, norm_cfg=dict(type="BN", requires_grad=True),
                         align_corners=False,
                         loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)),
        train_cfg=dict(log_config=dict(img_interval=1000)), test_cfg=dict(mode="whole"),
    )).to(device)
    model.automatic_debug = False
    images = torch.randn(2, 3, 64, 64, device=device)
    labels = torch.randint(0, 3, (2, 1, 64, 64), device=device)
    masked = BlockMaskGenerator(0.7, 16).mask_image(images)
    losses = model.forward_train(masked, [{}, {}], labels)
    loss = sum(value for name, value in losses.items() if "loss" in name)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        output = model.encode_decode(images, [{}, {}])
    assert output.shape == (2, 3, 64, 64)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    _, keep = nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    assert keep.numel() == 1
    print("PASS: original MIC config/masking, segmentor forward/backward, and compiled MMCV NMS.")


def detection(torch, device):
    sys.path.insert(0, str(ROOT / "det"))
    import maskrcnn_benchmark

    # Prefer our isolated build over any pre-existing extension in det/.
    extension_dir = ROOT / "uv/det/build/lib/maskrcnn_benchmark"
    maskrcnn_benchmark.__path__.insert(0, str(extension_dir))
    from maskrcnn_benchmark import _C
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.modeling.masking import Masking

    config = cfg.clone()
    config.merge_from_file(str(ROOT / "det/configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml"))
    config.MODEL.DEVICE = str(device)
    config.MODEL.WEIGHT = ""  # Synthetic test: no pretrained download.
    config.MODEL.RPN.PRE_NMS_TOP_N_TEST = 20
    config.MODEL.RPN.POST_NMS_TOP_N_TEST = 10
    config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 10
    model = build_detection_model(config).to(device).eval()
    images = torch.randn(1, 3, 64, 64, device=device)
    images = Masking(16, 0.5, 0, 0, False, (0, 0, 0), (1, 1, 1))(images)
    with torch.no_grad():
        results = model(images, with_DA_ON=False)
    assert len(results) == 1 and results[0].bbox.shape[1] == 4
    print("PASS: original detector config, MIC masking, native extension, and synthetic detector inference.")
    print("Native extension:", _C.__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("cls", "seg", "det"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    expected = (3, 9, 7) if args.profile == "det" else (3, 8, 5)
    if sys.version_info[:3] != expected:
        raise RuntimeError("Use the profile's exact Python {} (got {}).".format(expected, sys.version_info[:3]))
    import torch

    expected_torch = {"cls": "1.9.1+cu111", "seg": "1.7.1+cu110", "det": "1.12.0+cu113"}[args.profile]
    assert torch.__version__ == expected_torch, (torch.__version__, expected_torch)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "runtime CUDA:", torch.version.cuda)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check driver access outside the sandbox.")
        print("GPU:", torch.cuda.get_device_name(0))
    globals()[{"cls": "classification", "seg": "segmentation", "det": "detection"}[args.profile]](torch, device)
    if args.device == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
