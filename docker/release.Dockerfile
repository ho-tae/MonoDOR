FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ARG TORCH_CUDA_ARCH_LIST=8.0
ARG MAX_JOBS=2
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git ninja-build libglib2.0-0 libgl1 libsm6 libxext6 libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/monodor
COPY . .
RUN bash scripts/release/install.sh
RUN python scripts/release/check_environment.py
CMD ["bash"]
