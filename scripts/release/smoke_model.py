"""Run one validation frame through the model; this does not measure AP."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "VMamba")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kitti")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "checkpoints/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth")
    args = parser.parse_args()
    data_root, pretrained = args.data_root.resolve(), args.pretrained.resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    os.chdir(ROOT / "tools")
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    cfg = cfg_from_yaml_file(str(ROOT / "configs/release/vmamba_kitti.yaml"), EasyDict())
    cfg.DATA_CONFIG.DATA_PATH = str(data_root)
    cfg.MODEL.VFE.FFN.DDN.ARGS.pretrained_path = str(pretrained)
    dataset = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                           root_path=data_root, logger=common_utils.create_logger())
    batch = dataset.collate_batch([dataset[0]])
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda().eval()
    load_data_to_gpu(batch)
    with torch.no_grad():
        predictions, _ = model(batch)
    torch.cuda.synchronize()
    assert len(predictions) == 1
    for key in ("pred_boxes", "pred_scores"):
        assert torch.isfinite(predictions[0][key]).all(), key
    print("One-frame model forward passed. Detector weights are untrained; no AP claim.")


if __name__ == "__main__":
    main()
