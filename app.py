"""Gradio demo: compare LaMa vs MI-GAN object removal side-by-side."""
from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from lama_model import LaMaModel
from migan_model import MIGANModel

MIGAN_PATH = os.getenv("MIGAN_MODEL_PATH", "models/migan_pipeline_v2.onnx")
LAMA_PATH = os.getenv("LAMA_MODEL_PATH", "models/lama_fp32.onnx")


def pick_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


for name, path in [("MI-GAN", MIGAN_PATH), ("LaMa", LAMA_PATH)]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} model not found at {path}. Run `python download_model.py` first."
        )

providers = pick_providers()
print(f"Loading LaMa  from {LAMA_PATH}")
LAMA = LaMaModel(LAMA_PATH, providers=providers)
print(f"Loading MI-GAN from {MIGAN_PATH}")
MIGAN = MIGANModel(MIGAN_PATH, providers=providers)
print("Active providers:", LAMA.sess.get_providers())


def _extract_mask(layers: list, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers or []:
        if layer is None:
            continue
        arr = np.asarray(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            painted = arr[..., 3] > 0
        elif arr.ndim == 3:
            painted = np.asarray(Image.fromarray(arr).convert("L")) > 0
        else:
            painted = arr > 0
        mask[painted] = 255
    return mask


def process(editor_value, migan_invert: bool):
    if editor_value is None:
        raise gr.Error("이미지를 업로드하고 브러시로 지울 영역을 칠해주세요.")

    bg = editor_value.get("background")
    layers = editor_value.get("layers") or []
    if bg is None:
        raise gr.Error("배경 이미지가 없습니다. 먼저 이미지를 업로드하세요.")
    if not layers or all(lyr is None for lyr in layers):
        raise gr.Error("브러시로 제거할 영역을 칠해주세요.")

    image_rgb = np.array(Image.fromarray(bg).convert("RGB"))
    mask = _extract_mask(layers, image_rgb.shape[:2])
    if mask.max() == 0:
        raise gr.Error("칠한 영역이 감지되지 않았습니다.")

    t0 = time.perf_counter()
    lama_out = LAMA(image_rgb, mask)
    lama_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    migan_out = MIGAN(image_rgb, mask, invert_mask=migan_invert)
    migan_ms = (time.perf_counter() - t1) * 1000

    return (
        Image.fromarray(lama_out),
        f"LaMa   inference: {lama_ms:.1f} ms  (providers: {LAMA.sess.get_providers()[0]})",
        Image.fromarray(migan_out),
        f"MI-GAN inference: {migan_ms:.1f} ms  (providers: {MIGAN.sess.get_providers()[0]})",
        Image.fromarray(mask),
    )


with gr.Blocks(title="LaMa vs MI-GAN Eraser Demo") as demo:
    gr.Markdown(
        """
        # LaMa vs MI-GAN — Object Removal Comparison
        브러시로 지우고 싶은 영역을 칠한 뒤 **Run** 버튼을 눌러 두 모델 결과를 비교하세요.

        - **LaMa**: [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) (Apache 2.0) · 512×512 고정
        - **MI-GAN**: [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) (MIT) · 임의 해상도
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="이미지 + 마스크 (브러시로 제거 영역 칠하기)",
                type="numpy",
                brush=gr.Brush(default_size=30, colors=["#ffffff"]),
                eraser=gr.Eraser(default_size=30),
                sources=["upload", "clipboard"],
                layers=True,
                transforms=[],
            )
            migan_invert = gr.Checkbox(
                value=True,
                label="MI-GAN: invert mask (debug)",
                info="ON: 칠한 영역을 '제거 대상'으로 처리(권장). OFF로 결과가 더 자연스러우면 파이프라인이 내부에서 이미 반전 처리한다는 뜻 — 보고해주세요.",
            )
            run = gr.Button("Run", variant="primary")
            mask_preview = gr.Image(label="추출된 마스크 (reference)", type="pil")

        with gr.Column(scale=1):
            lama_img = gr.Image(label="LaMa 결과", type="pil")
            lama_time = gr.Markdown()
            migan_img = gr.Image(label="MI-GAN 결과", type="pil")
            migan_time = gr.Markdown()

    run.click(
        process,
        inputs=[editor, migan_invert],
        outputs=[lama_img, lama_time, migan_img, migan_time, mask_preview],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
