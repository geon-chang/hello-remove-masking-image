"""Download MI-GAN and LaMa ONNX weights into ./models."""
from __future__ import annotations

import os

from huggingface_hub import hf_hub_download

MODELS = [
    {
        "name": "MI-GAN",
        "repo_id": "andraniksargsyan/migan",
        "filename": "migan_pipeline_v2.onnx",
        "size_mb": 29.5,
        "license": "MIT",
    },
    {
        "name": "LaMa",
        "repo_id": "Carve/LaMa-ONNX",
        "filename": "lama_fp32.onnx",
        "size_mb": 208.0,
        "license": "Apache 2.0",
    },
]


def main() -> None:
    os.makedirs("models", exist_ok=True)
    total = 0.0
    for i, m in enumerate(MODELS, start=1):
        print(f"[{i}/{len(MODELS)}] {m['name']} ({m['license']})")
        print(f"  from: https://huggingface.co/{m['repo_id']}")
        print(f"  file: {m['filename']} (~{m['size_mb']} MB)")
        path = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir="models",
        )
        total += m["size_mb"]
        print(f"  -> {path}\n")
    print(f"Done. Approx total: {total:.1f} MB")


if __name__ == "__main__":
    main()
