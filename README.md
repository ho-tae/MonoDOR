# MonoDOR

**Depth- and Occupancy-Aware Representation Learning for Monocular 3D Object Detection**

Code for **Monocular 3D Detection under Occlusion via Depth and Occupancy-Aware Representation Learning**

Heetae Han and Sang Min Yoon · **Accepted to IEEE Transactions on Intelligent Vehicles (T-IV)**

MonoDOR is the code-release name for the VMamba-based detector with depth-aware enhancement (DAE) and occupancy-aware enhancement (OAE).

**Publication status:** accepted to **IEEE Transactions on Intelligent Vehicles (T-IV)**. The publisher page is forthcoming. This repository provides the code and environment setup; the paper PDF, DOI and final citation will be added separately when available.

[한국어 안내](docs/README_ko.md) · [Reproduction guide](docs/REPRODUCING.md) · [Results](docs/RESULTS.md) · [Validation status](docs/VALIDATION.md)

## Overview

The implementation builds on OpenPCDet/CaDDN and VMamba. Image features are lifted into a depth-indexed frustum, transformed into a voxel representation, refined using predicted occupancy, and passed to a BEV detection head.

```text
RGB image → VMamba / UPerNet → depth-guided frustum features
                                      ↓ calibration
                              voxel features + OAE
                                      ↓
                            BEV backbone → 3D boxes
```

The model uses **RGB images and calibration at inference**. Training uses LiDAR-derived depth and occupancy supervision together with detection annotations. The current KITTI loader also reads point/depth files during validation; the supplied evaluation command therefore expects the prepared dataset, even though measured depth and points are not model inference inputs.

This release provides one configuration for the **current combined model**. Historical ablations involved source edits and are not selectable by changing a DAE/OAE YAML flag.

## Results at a glance

Reported KITTI **test-set** results from the available manuscript, AP R40 (%), IoU = 0.5. These are paper-reported values, not fresh measurements of this source snapshot.

| Class | Metric | Easy | Moderate | Hard |
|---|---|---:|---:|---:|
| Pedestrian | AP3D | 17.32 | 11.22 | 9.56 |
| Pedestrian | APBEV | 18.91 | 12.50 | 11.77 |
| Cyclist | AP3D | 9.33 | 5.55 | 5.25 |
| Cyclist | APBEV | 10.44 | 6.95 | 6.77 |

The [results note](docs/RESULTS.md) includes the compact Car ablation and efficiency tables, with their measurement and provenance limits. No paper PDF or trained detector weights are bundled.

## Environment

| Component | Release target |
|---|---|
| Platform | Linux x86-64, NVIDIA GPU |
| Python | 3.10 |
| PyTorch / torchvision | 2.1.0 / 0.16.0, CUDA 11.8 builds |
| CUDA development toolkit | 11.8 |
| NumPy | 1.26.4 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| MMSegmentation / MMDetection | 1.2.2 / 3.3.0 |
| spconv | spconv-cu118 2.3.8 |
| VMamba CUDA kernel | bundled selective_scan, oflex backend |

Core versions come from the available development environment. The [observed environment](requirements/environment-observed.json) records that evidence; the [release dependency file](requirements/release-cu118.txt) is a curated set of direct pins, not a complete historical lockfile. Full Docker build verification remains pending; see [validation status](docs/VALIDATION.md).

## Quick start with Docker

Install Docker with the Compose plugin and NVIDIA Container Toolkit on the host. The Compose file uses `gpus: all` (Compose 2.30.0+). Docker builds need CUDA development tools inside the image, but do not need a visible GPU.

```bash
git clone https://github.com/ho-tae/MonoDOR.git
cd MonoDOR
cp docker/.env.example docker/.env
```

Edit `docker/.env` with absolute host paths for `KITTI_ROOT`, `CHECKPOINT_ROOT`, and `OUTPUT_ROOT`. Create those directories first. Put the ADE20K VMamba/UPerNet initialization checkpoint in `CHECKPOINT_ROOT` under this exact filename:

```text
upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
```

Obtain it from the upstream VMamba segmentation model resources described in the [reproduction guide](docs/REPRODUCING.md). No checkpoint is downloaded automatically.

```bash
# A100 target; change TORCH_CUDA_ARCH_LIST in docker/.env for another GPU.
docker compose --env-file docker/.env -f docker/compose.yaml build
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  python scripts/release/check_environment.py --cuda
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  python scripts/release/prepare_kitti.py
```

KITTI must contain the standard train/val split, metadata pickles, RGB images, calibration, annotations, point clouds, `depth_2`, and `lidar_depth`. [Dataset preparation](docs/REPRODUCING.md#2-prepare-kitti) explains the layout and the remaining depth-preprocessing dependency.

Train the current combined model:

```bash
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  bash scripts/release/run.sh train --extra_tag release_run --fix_random_seed
```

Evaluate a trained detector checkpoint (separate from the initialization checkpoint):

```bash
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  bash scripts/release/run.sh eval \
  --ckpt ../output/configs/release/vmamba_kitti/release_run/ckpt/checkpoint_epoch_80.pth \
  --extra_tag release_run --eval_tag val_epoch80 --save_to_file
```

Data and initialization weights are mounted read-only; outputs are persisted on the host. Source code is copied into the image, so rebuild after changing it.

## Native installation

Use a dedicated Python 3.10 environment with CUDA 11.8 `nvcc` and a C++ compiler. Install the system packages listed in [docker/release.Dockerfile](docker/release.Dockerfile), then:

```bash
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 bash scripts/release/install.sh
python scripts/release/check_environment.py --cuda
mkdir -p data checkpoints
ln -s /absolute/path/to/kitti data/kitti
# Place the initialization checkpoint in checkpoints/.
python scripts/release/prepare_kitti.py
bash scripts/release/run.sh train --extra_tag release_run --fix_random_seed
```

`8.0` targets the A100 used for local checks. Set the architecture for your GPU before compiling both OpenPCDet and selective scan. The launcher works from any directory and sets the working directory expected by the configuration. Relative arguments such as `--ckpt` are resolved from `tools/`.

## Training configuration

The public entry point is [configs/release/vmamba_kitti.yaml](configs/release/vmamba_kitti.yaml).

| Setting | Value |
|---|---|
| Classes | Car, Pedestrian, Cyclist |
| Train / validation split | 3,712 / 3,769 frames |
| Initialization | VMamba-B / UPerNet trained on ADE20K |
| Epochs / batch size | 80 / 2 on one GPU |
| Optimizer / configured LR | `adam_onecycle` / 0.001 |
| Image-affine target resolution | 1280 × 384 |
| Depth discretization | 150 LID bins, 2.0–46.8 m, plus one invalid category |
| Voxel size | 0.16 × 0.16 × 0.16 m |
| Spatial range | x: 2–46.8, y: −30.08–30.08, z: −3–1 m |

The 80-epoch setting is supported by the available recent run records. It does not establish that every historical paper result came from this exact snapshot. Checkpoints and a complete result-to-source mapping are not yet included.

## Repository map

| Path | Purpose |
|---|---|
| `pcdet/` | Detector, data pipeline, losses and CUDA operators |
| `VMamba/` | Bundled VMamba implementation and segmentation configuration |
| `configs/release/` | Portable configuration for this release |
| `scripts/release/` | Installation, checks, training/evaluation launcher and source export |
| `docker/` | Release image and GPU Compose service |
| `requirements/` | Direct dependency pins and observed environment metadata |
| `docs/` | Setup, results, validation and release instructions |

## Acknowledgements and license

This repository derives from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), the CaDDN implementation in OpenPCDet, [VMamba](https://github.com/MzeroMiko/VMamba), and the [OpenMMLab](https://github.com/open-mmlab) ecosystem. Preserve their authorship, citations and bundled license notices when redistributing. The repository's existing root [Apache-2.0 license](LICENSE) is retained; bundled third-party code remains subject to its own notices.

## Citation and release status

The T-IV acceptance has been confirmed by the authors. A final publisher link, bibliographic metadata and BibTeX will be added after online publication. No placeholder DOI, publication date, or detector download is presented as an available artifact.

For publishing the prepared source snapshot without local research material, use the [release instructions](docs/RELEASE.md).
