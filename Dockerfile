FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG PIP_TRUSTED_HOSTS="--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.nvidia.com"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install ${PIP_TRUSTED_HOSTS} --no-deps iopaint==1.6.0 && \
    grep -Ev '^(iopaint==|onnxruntime|torch([[:punct:]]|$)|torchvision([[:punct:]]|$))' requirements.txt > /tmp/requirements-docker.txt && \
    python -m pip install ${PIP_TRUSTED_HOSTS} --index-url ${TORCH_INDEX_URL} 'torch>=2.8,<2.11' torchvision && \
    python -m pip install ${PIP_TRUSTED_HOSTS} 'onnxruntime-gpu>=1.20.0' && \
    python -m pip install ${PIP_TRUSTED_HOSTS} -r /tmp/requirements-docker.txt

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
