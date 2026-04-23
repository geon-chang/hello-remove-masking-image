"""Gradio demo with two tabs:

1. Comparison — AS-IS (LaMa, MI-GAN) vs TO-BE (PowerPaint v2-1 object-remove),
   all three run in parallel on a single mask with per-model inference time.
2. PowerPaint Playground — run PowerPaint v2-1 with any of its five tasks
   (object-remove, text-guided, shape-guided, context-aware, outpainting)
   with custom prompt / steps / guidance / seed.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from lama_model import LaMaModel
from migan_model import MIGANModel

MIGAN_PATH = os.getenv("MIGAN_MODEL_PATH", "models/migan_pipeline_v2.onnx")
LAMA_PATH = os.getenv("LAMA_MODEL_PATH", "models/lama_fp32.onnx")
ENABLE_POWERPAINT = os.getenv("ENABLE_POWERPAINT", "1") != "0"

# Generic-only quality negatives (no specific object/feature names).
# Object-specific terms ("eye", "person", "chair" 등)을 넣으면 해당 도메인에서는 잘
# 작동하지만 다른 이미지에서 원치 않는 회피 패턴을 유발. 대신 low-CFG(=3) + dil=0 +
# 일반 품질 negatives 조합이 /tmp/pp_test 반복 실험에서 가장 자연스럽게 제거됨.
OBJECT_REMOVE_NEGATIVE = (
    "blurry, low quality, bad quality, worst quality, artifacts, "
    "noisy, distorted, deformed, jpeg artifacts, watermark, text"
)


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
print("ONNX providers:", LAMA.sess.get_providers())


# --- PowerPaint lazy loader ---------------------------------------------------
# Heavy (torch + diffusers + ~5GB weights); load on first click, not at import.

_PP_MODEL = None
_PP_LOCK = threading.Lock()
_PP_ERROR: Optional[str] = None


def get_powerpaint():
    global _PP_MODEL, _PP_ERROR
    if not ENABLE_POWERPAINT:
        raise gr.Error("PowerPaint is disabled (ENABLE_POWERPAINT=0).")
    if _PP_MODEL is not None:
        return _PP_MODEL
    if _PP_ERROR is not None:
        raise gr.Error(f"PowerPaint failed to load: {_PP_ERROR}")
    with _PP_LOCK:
        if _PP_MODEL is not None:
            return _PP_MODEL
        try:
            from powerpaint_model import PowerPaintModel

            print("Loading PowerPaint v2-1 (first call — may download weights)...")
            t0 = time.perf_counter()
            _PP_MODEL = PowerPaintModel()
            print(f"PowerPaint loaded in {time.perf_counter() - t0:.1f}s on {_PP_MODEL.device}")
            return _PP_MODEL
        except Exception as e:  # noqa: BLE001
            _PP_ERROR = str(e)
            raise gr.Error(f"PowerPaint load failed: {e}") from e


# --- shared helpers -----------------------------------------------------------

def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """브러시 경계를 px만큼 확장. 타이트한 마스크가 객체 그림자/테두리를 남기는 문제 완화."""
    if px <= 0:
        return mask
    import cv2
    k = max(1, int(px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


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


def _unpack_editor(editor_value) -> tuple[np.ndarray, np.ndarray]:
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
    return image_rgb, mask


# --- Tab 1: Comparison --------------------------------------------------------

def run_comparison(
    editor_value,
    migan_invert: bool,
    pp_steps: int,
    pp_cfg: float,
    pp_seed: int,
    pp_dilate: int,
    pp_negative: str,
):
    image_rgb, mask = _unpack_editor(editor_value)
    pp_mask = _dilate_mask(mask, int(pp_dilate))
    negative = pp_negative.strip() if pp_negative else ""
    if not negative:
        negative = OBJECT_REMOVE_NEGATIVE

    t0 = time.perf_counter()
    lama_out = LAMA(image_rgb, mask)
    lama_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    migan_out = MIGAN(image_rgb, mask, invert_mask=migan_invert)
    migan_ms = (time.perf_counter() - t1) * 1000

    pp_img = None
    pp_label = "PowerPaint 비활성화"
    if ENABLE_POWERPAINT:
        try:
            from powerpaint_model import PowerPaintParams

            pp = get_powerpaint()
            t2 = time.perf_counter()
            pp_out = pp(
                image_rgb,
                pp_mask,
                PowerPaintParams(
                    task="object-remove",
                    negative_prompt=negative,
                    steps=int(pp_steps),
                    guidance_scale=float(pp_cfg),
                    seed=int(pp_seed),
                ),
            )
            pp_ms = (time.perf_counter() - t2) * 1000
            pp_img = Image.fromarray(pp_out)
            pp_label = (
                f"PowerPaint v2-1 (object-remove) · {pp_ms:.0f} ms · device={pp.device} · "
                f"steps={int(pp_steps)} · cfg={pp_cfg} · dilate={int(pp_dilate)}px"
            )
        except gr.Error:
            raise
        except Exception as e:  # noqa: BLE001
            pp_label = f"PowerPaint 에러: {e}"

    return (
        Image.fromarray(lama_out),
        f"LaMa inference: {lama_ms:.1f} ms  (provider: {LAMA.sess.get_providers()[0]})",
        Image.fromarray(migan_out),
        f"MI-GAN inference: {migan_ms:.1f} ms  (provider: {MIGAN.sess.get_providers()[0]})",
        pp_img,
        pp_label,
        Image.fromarray(pp_mask if ENABLE_POWERPAINT else mask),
    )


# --- Tab 2: Outpainting -------------------------------------------------------

def run_outpaint(
    image,
    top: int,
    right: int,
    bottom: int,
    left: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
):
    if image is None:
        raise gr.Error("확장할 이미지를 먼저 업로드하세요.")
    image_rgb = np.array(Image.fromarray(image).convert("RGB"))
    top, right, bottom, left = int(top), int(right), int(bottom), int(left)
    if top + right + bottom + left == 0:
        raise gr.Error("확장할 방향(top/right/bottom/left) 중 최소 1개 이상은 0보다 크게 설정하세요.")

    h, w = image_rgb.shape[:2]
    new_h = h + top + bottom
    new_w = w + left + right
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    canvas[top:top + h, left:left + w] = image_rgb
    mask = np.full((new_h, new_w), 255, dtype=np.uint8)
    mask[top:top + h, left:left + w] = 0

    from powerpaint_model import PowerPaintParams

    pp = get_powerpaint()
    t0 = time.perf_counter()
    out = pp(
        canvas,
        mask,
        PowerPaintParams(
            task="outpainting",
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            steps=int(steps),
            guidance_scale=float(cfg),
            seed=int(seed),
        ),
    )
    ms = (time.perf_counter() - t0) * 1000
    return (
        Image.fromarray(out),
        (
            f"outpaint · {ms:.0f} ms · device={pp.device} · "
            f"원본 {w}×{h} → 결과 {new_w}×{new_h}  "
            f"(T{top}/R{right}/B{bottom}/L{left}) · steps={int(steps)} · cfg={cfg} · seed={int(seed)}"
        ),
        Image.fromarray(mask),
    )


# --- UI -----------------------------------------------------------------------

with gr.Blocks(title="Eraser Model Playground") as demo:
    gr.Markdown(
        """
        # Object-Removal Model Playground

        브러시로 칠한 영역을 **3개 모델**로 비교/실험하는 데모입니다.

        | 구분 | 모델 | 특징 | 라이선스 |
        |---|---|---|---|
        | **AS-IS** | LaMa (ONNX) | 빠르고 가벼움 · 512×512 고정 · CPU OK | Apache 2.0 |
        | **AS-IS** | MI-GAN (ONNX) | 경량 GAN · 임의 해상도 · CPU OK | MIT |
        | **TO-BE** | **PowerPaint v2-1** | SD 1.5 + BrushNet 기반 diffusion · 고품질 · GPU 권장 | CreativeML OpenRAIL-M |

        **탭 구성**
        - **Tab 1 — Outpainting**: 이미지를 상/하/좌/우로 자유롭게 확장 (PowerPaint v2-1 outpainting task)
        - **Tab 2 — Comparison**: 동일 이미지·마스크를 LaMa / MI-GAN / PowerPaint 3개 모델로 동시에 돌려 비교

        ### PowerPaint v2-1 이란?
        Stable Diffusion 1.5 위에 **BrushNet**(마스크 조건부 ControlNet)과 **학습된 task token**을 얹어
        한 모델로 5가지 인페인팅 태스크(object-remove / text-guided / shape-guided / context-aware / outpainting)를
        수행할 수 있게 한 모델입니다. 기존 LaMa·MI-GAN 계열과 달리 **텍스트 프롬프트**를 받을 수 있고,
        배경 재구성 품질이 현저히 향상됩니다. 단 가중치가 **~5 GB**이고 GPU(cuda/mps) 사용이 사실상 필수입니다.

        > 💡 PowerPaint는 **첫 호출 시** HF Hub에서 가중치를 다운로드하고 GPU(cuda > mps > cpu)가 자동 선택됩니다.
        > 로딩에 30초–수 분 걸릴 수 있습니다.
        """
    )

    with gr.Tab("Outpainting (이미지 확장)"):
        gr.Markdown(
            """
            ### PowerPaint v2-1 outpainting — 이미지를 원하는 방향으로 확장
            이미지만 업로드하고 확장할 방향(위/오른쪽/아래/왼쪽)의 픽셀 값을 지정하면,
            PowerPaint가 해당 영역을 자연스럽게 생성합니다. 마스크는 자동 생성됩니다.

            - **Prompt 선택**: 비워두면 주변 맥락을 그대로 이어서 생성. 원하는 느낌이 있으면 간단히 기술 (예: `city skyline at sunset`).
            - **권장 확장량**: 한 방향당 128–256 px. 더 큰 값은 품질이 흐트러질 수 있음.
            - **Steps 30–45 / CFG 5–7.5** 권장.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_img_op = gr.Image(
                    label="확장할 이미지",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                with gr.Row():
                    top_px = gr.Slider(0, 512, value=128, step=16, label="Top (↑) 확장 px")
                    bottom_px = gr.Slider(0, 512, value=128, step=16, label="Bottom (↓) 확장 px")
                with gr.Row():
                    left_px = gr.Slider(0, 512, value=128, step=16, label="Left (←) 확장 px")
                    right_px = gr.Slider(0, 512, value=128, step=16, label="Right (→) 확장 px")
                prompt_op = gr.Textbox(
                    label="Prompt (선택)",
                    value="",
                    info="비워두면 주변 맥락으로 자동 확장.",
                )
                negative_prompt_op = gr.Textbox(
                    label="Negative prompt (선택)",
                    value="",
                )
                with gr.Row():
                    steps_op = gr.Slider(10, 50, value=30, step=1, label="Steps")
                    cfg_op = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="CFG")
                    seed_op = gr.Number(value=42, precision=0, label="Seed")
                run_op = gr.Button("Outpaint", variant="primary")
                mask_preview_op = gr.Image(label="자동 생성된 outpainting 마스크", type="pil")

            with gr.Column(scale=1):
                out_img_op = gr.Image(label="확장된 결과", type="pil")
                out_info_op = gr.Markdown()

        run_op.click(
            run_outpaint,
            inputs=[
                input_img_op,
                top_px, right_px, bottom_px, left_px,
                prompt_op, negative_prompt_op,
                steps_op, cfg_op, seed_op,
            ],
            outputs=[out_img_op, out_info_op, mask_preview_op],
        )

    with gr.Tab("Comparison (AS-IS vs TO-BE)"):
        gr.Markdown(
            """
            ### 동일 이미지·동일 마스크를 3개 모델에 동시에 통과시킵니다
            - **LaMa / MI-GAN** — 기존 경량 인페인팅 (ms 단위, CPU 가능)
            - **PowerPaint v2-1 (object-remove)** — diffusion 기반 고품질 제거 (초 단위, GPU 권장)

            오른쪽 세로로 결과 3종이 나란히 표시되며 각 모델의 추론 시간(ms)도 함께 출력됩니다.
            PowerPaint 파라미터(Steps/CFG/Seed)는 하단 아코디언에서 조절할 수 있습니다.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                editor_cmp = gr.ImageEditor(
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
                    info="ON 권장 (UI의 흰 영역 = 제거 대상).",
                )
                with gr.Accordion("PowerPaint 파라미터 (object-remove 품질 튜닝)", open=True):
                    pp_dilate = gr.Slider(
                        0, 30, value=0, step=1,
                        label="Mask dilation (px)",
                        info="기본 0. 마스크 경계에 객체 잔상이 남을 때만 2–6 정도로 높이세요. 너무 크면 주변 피처까지 변형됨.",
                    )
                    pp_steps = gr.Slider(10, 50, value=30, step=1, label="Steps", info="적을수록 빠름. 30–45 권장.")
                    pp_cfg = gr.Slider(1.0, 12.0, value=3.0, step=0.5, label="Guidance scale", info="object-remove는 2–4 권장. 낮을수록 unconditional 배경에 가깝게 채움.")
                    pp_seed = gr.Number(value=42, precision=0, label="Seed")
                    pp_negative = gr.Textbox(
                        value="",
                        label="Negative prompt (비우면 generic quality negs 적용)",
                        info="비워두면 'blurry, low quality, artifacts, …' 같은 도메인 중립 negative가 적용됩니다. 특정 객체명은 가급적 넣지 마세요.",
                        lines=2,
                    )
                run_cmp = gr.Button("Run all 3 models", variant="primary")
                mask_preview_cmp = gr.Image(label="추출된 마스크", type="pil")

            with gr.Column(scale=1):
                lama_img = gr.Image(label="LaMa 결과 (AS-IS)", type="pil")
                lama_time = gr.Markdown()
                migan_img = gr.Image(label="MI-GAN 결과 (AS-IS)", type="pil")
                migan_time = gr.Markdown()
                pp_img_cmp = gr.Image(label="PowerPaint v2-1 결과 (TO-BE)", type="pil")
                pp_time_cmp = gr.Markdown()

        run_cmp.click(
            run_comparison,
            inputs=[editor_cmp, migan_invert, pp_steps, pp_cfg, pp_seed, pp_dilate, pp_negative],
            outputs=[lama_img, lama_time, migan_img, migan_time, pp_img_cmp, pp_time_cmp, mask_preview_cmp],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
