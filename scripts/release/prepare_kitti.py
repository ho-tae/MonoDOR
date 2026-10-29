"""Validate KITTI files; optionally generate train/val infos for this release."""
import argparse
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "VMamba")]


def validate(root, require_infos=True):
    splits = {}
    missing = []
    for split in ("train", "val"):
        split_file = root / "ImageSets" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        ids = split_file.read_text().split()
        if not ids or len(ids) != len(set(ids)) or any(not x.isdigit() for x in ids):
            raise ValueError(f"Invalid or duplicate frame IDs: {split_file}")
        splits[split] = ids
        for frame in ids:
            for folder, suffix in (("image_2", ".png"), ("calib", ".txt"),
                                   ("label_2", ".txt"), ("velodyne", ".bin"),
                                   ("depth_2", ".png"), ("lidar_depth", ".png")):
                file = root / "training" / folder / (frame + suffix)
                if not file.is_file():
                    missing.append(str(file))
        if require_infos and not (root / f"kitti_infos_{split}.pkl").is_file():
            missing.append(str(root / f"kitti_infos_{split}.pkl"))
        print(f"{split}: {len(ids)} frames")
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Train and validation splits overlap")
    if missing:
        raise FileNotFoundError(f"{len(missing)} required files missing; first 10:\n" + "\n".join(missing[:10]))
    print("Required files present. Depth generation provenance/encoding is not certified by this check.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kitti")
    parser.add_argument("--create-infos", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.data_root.resolve()
    validate(root, require_infos=not args.create_infos)
    if not args.create_infos:
        return
    targets = [root / f"kitti_infos_{split}.pkl" for split in ("train", "val")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Info files already exist. Back them up or choose a fresh dataset directory.")
    import yaml
    from easydict import EasyDict
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    config = EasyDict(yaml.safe_load((ROOT / "configs/release/vmamba_kitti.yaml").read_text()))
    dataset = KittiDataset(config.DATA_CONFIG, config.CLASS_NAMES, root_path=root, training=False)
    for split, target in zip(("train", "val"), targets):
        dataset.set_split(split)
        infos = dataset.get_infos(num_workers=args.workers, has_label=True, count_inside_pts=True)
        with target.open("xb") as handle:
            pickle.dump(infos, handle)
        print(f"Created {target}")
    # This model uses image-affine augmentation, not ground-truth database sampling.


if __name__ == "__main__":
    main()
