#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/models}"
mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/lama_fp32.onnx" ] || [ ! -f "$MODELS_DIR/migan_pipeline_v2.onnx" ]; then
    echo "[entrypoint] models missing, downloading..."
    python download_model.py
fi

exec python app.py
