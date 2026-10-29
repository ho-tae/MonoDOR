"""Import checks; --cuda additionally executes small CUDA forward/backward ops."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "VMamba")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    expected = {"torch": "2.1.0", "torchvision": "0.16.0", "numpy": "1.26.4",
                "mmcv": "2.1.0", "mmengine": "0.10.7", "mmsegmentation": "1.2.2",
                "mmdet": "3.3.0", "spconv-cu118": "2.3.8", "timm": "0.4.12"}
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Expected Python 3.10")
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        print(f"{name}: {actual}", flush=True)
        if actual.split("+")[0] != version:
            raise RuntimeError(f"Expected {name}=={version}")
    for module in ("mmcv.ops", "torch_scatter", "selective_scan_cuda_oflex",
                   "pcdet.ops.iou3d_nms.iou3d_nms_cuda", "pcdet.datasets", "pcdet.models"):
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)
    import torch
    if torch.version.cuda != "11.8":
        raise RuntimeError(f"Expected CUDA 11.8 PyTorch build, got {torch.version.cuda}")
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check the NVIDIA driver and container GPU access")
        from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
        from mmcv.ops import nms
        from torch_scatter import scatter_add
        from models.csms6s import selective_scan_fn
        boxes = torch.tensor([[0, 0, 0, 2, 2, 2, 0]], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(boxes_iou3d_gpu(boxes, boxes), torch.ones(1, 1, device="cuda"))
        nms(torch.tensor([[0., 0., 2., 2.]], device="cuda"), torch.ones(1, device="cuda"), 0.5)
        out = scatter_add(torch.ones(2, device="cuda"), torch.tensor([0, 0], device="cuda"))
        torch.testing.assert_close(out, torch.tensor([2.], device="cuda"))
        u = torch.randn(1, 8, 16, device="cuda", requires_grad=True)
        delta = torch.rand_like(u)
        a = -torch.ones(8, 1, device="cuda")
        b = torch.randn(1, 1, 1, 16, device="cuda")
        c = torch.randn_like(b)
        scan = selective_scan_fn(u, delta, a, b, c, backend="oflex")
        scan.sum().backward()
        assert torch.isfinite(scan).all() and torch.isfinite(u.grad).all()
        torch.cuda.synchronize()
        print(f"CUDA operator checks OK: {torch.cuda.get_device_name(0)}")
    print("Environment checks passed. This is not a full training/evaluation run.")


if __name__ == "__main__":
    main()
