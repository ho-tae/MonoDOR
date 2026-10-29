# Reproducing the environment and running the code

This guide describes the current combined model. It separates environment setup, data availability, operator/model checks, and full benchmark reproduction. Successful installation alone does not establish the reported AP values.

## 1. Install the environment

Use the Docker commands in the [README](../README.md) or a clean Python 3.10 environment. The native installer requires Linux x86-64, a C++ compiler, CUDA 11.8 development tools and the system libraries listed in [the Dockerfile](../docker/release.Dockerfile).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 bash scripts/release/install.sh
python scripts/release/check_environment.py --cuda
```

The installer builds both OpenPCDet CUDA extensions and the bundled VMamba selective-scan extension. It installs the PyTorch 2.1/CUDA 11.8 MMCV and torch-scatter wheels explicitly. Do not replace MMCV with `mmcv-lite`, or install a different PyTorch/CUDA pair without rebuilding the extensions.

The default CUDA architecture targets A100. For another GPU, obtain its compute capability and set `TORCH_CUDA_ARCH_LIST` before compilation. Selective scan now uses the supplied architecture list during GPU-free Docker builds. Support for other architectures has not been tested in this release.

The direct pins intentionally avoid copying every package in the shared development environment. That environment contains unrelated tools and dependency conflicts, including an OpenCV-headless build requiring NumPy 2. The release uses matching OpenCV 4.7 packages with NumPy 1.26.4. OS packages, transitive Python dependencies and the Docker base tag are not an immutable lock; capture the final resolved environment for each run.

Dependency references: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [MMCV CUDA 11.8/PyTorch 2.1 wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html), [MMSegmentation 1.2.2 compatibility checks](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py), [Docker Compose GPU support](https://docs.docker.com/reference/compose-file/services/#gpus).

## 2. Prepare KITTI

Obtain the KITTI 3D object detection data under its dataset terms. The release does not redistribute it. The loader expects:

```text
kitti/
├── ImageSets/
│   ├── train.txt                 # 3,712 frame IDs
│   └── val.txt                   # 3,769 frame IDs
├── kitti_infos_train.pkl
├── kitti_infos_val.pkl
└── training/
    ├── image_2/                  # left color image, PNG
    ├── calib/                    # per-frame calibration, TXT
    ├── label_2/                  # detection annotations, TXT
    ├── velodyne/                 # LiDAR point cloud, BIN
    ├── depth_2/                  # depth-supervision PNG
    └── lidar_depth/              # occupancy-supervision depth PNG
```

The train/val frame-ID lists used by the available environment are bundled at `configs/release/splits/`; copy `train.txt` and `val.txt` into the dataset's `ImageSets/` directory if needed.

Both depth folders are required by this configuration. The loader decodes PNG values as depth in metres by dividing by 256. `depth_2` is the depth-head supervision input. `lidar_depth` supplies observations that are back-projected and used to build occupancy targets. They must align with the corresponding image/calibration.

**The exact historical preprocessing pipeline for these depth assets is not included.** Supply the original prepared assets to reproduce the training inputs. A newly projected sparse map or a different depth-completion method should not be assumed equivalent. This release does not silently generate substitute depth labels or claim that the two folders are interchangeable.

For native execution, point `data/kitti` to the prepared dataset:

```bash
mkdir -p data
ln -s /absolute/path/to/kitti data/kitti
python scripts/release/prepare_kitti.py
```

The validator checks all frame IDs, train/val overlap and required files. It does not certify PNG encoding or pickle contents. If metadata is absent, generate it on a writable dataset directory:

```bash
python scripts/release/prepare_kitti.py --data-root /absolute/path/to/kitti --create-infos
```

This generates train/val metadata only, without requiring KITTI test images or building an unused ground-truth sampling database. It refuses to overwrite existing metadata. With Docker, run preparation in a one-off container with a writable dataset mount before using the read-only Compose mount:

```bash
docker run --rm --gpus all --shm-size=16g \
  --mount type=bind,src=/absolute/path/to/kitti,dst=/dataset \
  monodor:cu118 \
  python scripts/release/prepare_kitti.py --data-root /dataset --create-infos
```

The model's forward path uses RGB and calibration at inference, but the current dataset loader reads points and depth during evaluation too. Keep these files for the supplied validation command.

## 3. Supply initialization weights

The expected initialization is a **VMamba-B UPerNet checkpoint trained on ADE20K**, not a trained 3D detector:

```text
checkpoints/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
```

Consult the bundled [VMamba segmentation guide](../VMamba/segmentation/readme.md) and [upstream VMamba repository](https://github.com/MzeroMiko/VMamba) for model resources and their terms. Verify that the checkpoint matches the base configuration, and record its SHA-256 checksum. The setup scripts do not guess a download URL or publish a local checkpoint.

The original segmentation checkpoint has 150 semantic output channels. The detector's configuration uses 151 channels: 150 valid depth bins plus an invalid category. Expected segmentation-head shape mismatch messages occur during initialization, and the DDN wrapper filters those head tensors when loading compatible parameters. This is distinct from the ADE20K class-count setting. A detector checkpoint must match the detector architecture.

## 4. Check the installation and one model forward

```bash
python scripts/release/check_environment.py --cuda
python scripts/release/smoke_model.py
```

The first command checks pinned core versions, imports, 3D IoU, MMCV NMS, torch-scatter and selective-scan forward/backward. The second reads one validation frame and runs the model using initialization weights. It verifies finite box/score outputs, but the detection head is untrained; this is not an AP evaluation.

For an existing external dataset/checkpoint location:

```bash
python scripts/release/smoke_model.py \
  --data-root /absolute/path/to/kitti \
  --pretrained /absolute/path/to/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
```

## 5. Train, resume and evaluate

Run from the repository root. The launcher changes into `tools/` internally and injects the public configuration.

```bash
# 80 epochs, batch size 2, fixed code-defined seed (666).
bash scripts/release/run.sh train --extra_tag release_run --fix_random_seed

# Resume using the same experiment tag; specify the actual checkpoint.
bash scripts/release/run.sh train --extra_tag release_run --fix_random_seed \
  --ckpt ../output/configs/release/vmamba_kitti/release_run/ckpt/checkpoint_epoch_40.pth

# Evaluate on the configured validation split.
bash scripts/release/run.sh eval \
  --ckpt ../output/configs/release/vmamba_kitti/release_run/ckpt/checkpoint_epoch_80.pth \
  --extra_tag release_run --eval_tag val_epoch80 --save_to_file
```

Use a new `--extra_tag` for an independent run: the underlying training code can discover existing checkpoints in an existing experiment directory. The fixed seed does not guarantee bitwise deterministic GPU kernels. AMP is off by default; `--use_amp`, smaller images, voxel changes or a different batch size create a different experimental configuration.

Outputs go under `output/configs/release/vmamba_kitti/<extra_tag>/`. Evaluation logs report AP for each class and metric. Keep AP11/AP40 and BEV/3D results distinct; the summary tables use AP R40.

Paths passed to `--ckpt` and `--set` are interpreted from `tools/`; absolute paths avoid ambiguity. For example, the dataset location can be overridden with:

```bash
bash scripts/release/run.sh train --extra_tag external_data --fix_random_seed \
  --set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti
```

`--set` consumes the remaining arguments, so place it last. Use `train --help` or `eval --help` for the underlying arguments. This release documents single-GPU execution; it does not claim a validated distributed recipe.

## 6. Record experiment provenance

Preserve the resolved YAML, command, seed, checkpoint hashes, logs and exact source snapshot. The repository contains historical experiments with source edits that are not represented by YAML alone.

```bash
mkdir -p output/provenance
python -m pip freeze > output/provenance/pip-freeze.txt
nvidia-smi > output/provenance/nvidia-smi.txt
nvcc --version > output/provenance/nvcc.txt
git rev-parse HEAD > output/provenance/base-commit.txt
git diff --binary > output/provenance/worktree.patch
sha256sum checkpoints/*.pth > output/provenance/checkpoints.sha256
python scripts/release/export_source.py --output release-dist/run-source
```

`worktree.patch` alone omits untracked files. Keep the exported source and its file hash manifest too. Docker users should also record the built image ID/digest. Do not describe the base Git commit alone as the identity of a modified experiment.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named selective_scan_cuda_oflex` | Build the bundled kernel with the same Python/PyTorch/CUDA stack. |
| Missing `mmcv._ext` or undefined CUDA symbols | Use the pinned MMCV wheel; remove conflicting MMCV variants in a clean environment. |
| `no kernel image is available` | Rebuild with the architecture of the target GPU. |
| Compilation process killed | Reduce `MAX_JOBS`; provide more host RAM. |
| Missing `depth_2` or `lidar_depth` | Supply the original prepared depth assets; these are required, not optional. |
| CUDA out of memory | Check GPU availability and other jobs. A modified batch size is a different training setup. |
| Checkpoint not found | Supply the named ADE20K initialization file; evaluation additionally needs trained detector weights. |
| DataLoader shared-memory errors | Use the supplied 16 GB shared-memory setting or reduce worker count. |
| Compose rejects `gpus` | Use Compose 2.30.0+ or the documented `docker run --gpus all` form. |
