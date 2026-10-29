# Docker environment

Use `release.Dockerfile` and `compose.yaml` for the VMamba detector. The older `Dockerfile` and `cu116.Dockerfile` are historical OpenPCDet recipes and do not define this release.

From the repository root:

```bash
cp docker/.env.example docker/.env
# Edit the absolute data/checkpoint/output paths, then:
docker compose --env-file docker/.env -f docker/compose.yaml build
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research bash
```

The image uses PyTorch 2.1.0 / CUDA 11.8 development tools. `TORCH_CUDA_ARCH_LIST=8.0` targets A100; set a different value before building for a different GPU. `MAX_JOBS=2` limits native build memory. The selective-scan setup supports a GPU-free build when the target architecture is explicit.

The service exposes GPUs, allocates 16 GB shared memory, mounts KITTI and checkpoints read-only, and mounts output read-write. It does not require privileged mode, host networking or an X server. Host data preparation must finish before mounting the dataset read-only. Compose 2.30.0+ is needed for the `gpus` field.

Build without Compose (no data paths needed):

```bash
docker build -f docker/release.Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST=8.0 \
  --build-arg MAX_JOBS=2 -t monodor:cu118 .
```

Run an operator check without mounting a dataset:

```bash
docker run --rm --gpus all monodor:cu118 \
  python scripts/release/check_environment.py --cuda
```

The build context excludes datasets, weights, outputs, paper PDFs and private revision documents. Do not bind-mount the entire host source tree over the image's source directory: that can hide the native extensions built inside the image. Rebuild after source changes.

The base image is pinned by version tag, not immutable digest; direct Python dependencies are pinned, while some transitive dependencies and OS packages are resolved at build time. Save the resulting image digest and `pip freeze` alongside each experiment. See [reproduction](../docs/REPRODUCING.md) and [validation](../docs/VALIDATION.md).
