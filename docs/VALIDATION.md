# Release validation record

Prepared on 2026-10-29. These checks describe the working tree used to prepare the release, including pre-existing experimental edits. They do not certify all historical paper experiments.

| Check | Status | Scope |
|---|---|---|
| Core version checks | Passed | Python 3.10, PyTorch 2.1.0/cu118, pinned OpenMMLab stack |
| Dataset/model and native-extension imports | Passed | Current development environment, also with GPUs hidden |
| CUDA operator execution | Passed | 3D IoU, MMCV NMS, scatter-add, selective-scan forward/backward |
| Full model forward | Passed | One real KITTI validation frame, ADE20K initialization, finite box/score outputs |
| GPU-free selective-scan setup | Passed | `setup.py --name` with `CUDA_VISIBLE_DEVICES=''`, target architecture 8.0; not a fresh compilation |
| Release syntax/configuration/link checks | Passed | Public scripts, YAML, recursive VMamba config inheritance and local Markdown links |
| Source export | Passed | Public source and file hashes, no papers/weights/data/revision material |
| Fresh Docker build | Not run | Docker is unavailable on this host |
| Fresh native installation | Not run | Existing shared environment was not replaced |
| Full dataset completeness | Incomplete | 3,712/3,769 split parsed; 2,874 expected annotation files missing from the current local dataset |
| Full training / benchmark AP rerun | Not run | No new 80-epoch run or full validation result generated |

The current machine exposes NVIDIA A100-PCIE-40GB devices with driver 535.230.02 and CUDA toolkit 11.8. The one-frame forward uses existing prepared metadata and does not require reading all annotation text files, so it can pass while the complete dataset check fails.

During initialization, the upstream segmentation checkpoint's 150-class head differs from the detector's 151-channel depth head. The existing loader reports these mismatches and filters incompatible head tensors. The one-frame check uses an untrained detector head and cannot establish detection accuracy.

The shared environment also has unrelated dependency warnings/conflicts. It is an observation source, not a clean package lock. The release requirements select compatible direct pins, including OpenCV/NumPy; a clean container build and `pip check` are still required to validate the complete installation recipe.

## Commands

```bash
python scripts/release/check_source.py
python scripts/release/check_environment.py --cuda
CUDA_VISIBLE_DEVICES='' python scripts/release/check_environment.py
python scripts/release/prepare_kitti.py --data-root /absolute/path/to/kitti
python scripts/release/smoke_model.py \
  --data-root /absolute/path/to/kitti \
  --pretrained /absolute/path/to/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
python scripts/release/export_source.py
python release-dist/source/scripts/release/check_source.py
```

Use the environment's actual paths. The dataset command is expected to fail until missing assets are restored. The portable configuration and all public setup commands use repository-relative or user-supplied paths.
