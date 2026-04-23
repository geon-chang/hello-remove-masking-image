"""PowerPaint v2-1 wrapper.

Bypasses iopaint's ModelManager (which only registers locally-cached models)
and instantiates the PowerPaintV2 class directly with a manually-built
ModelInfo. SD 1.5 base weights come from the community mirror
`stable-diffusion-v1-5/stable-diffusion-v1-5` (runwayml repo was deleted in
July 2024). PowerPaint-specific components (BrushNet + task tokenizer) are
downloaded from `Sanster/PowerPaint_v2`.

Device auto-pick: CUDA > MPS > CPU. Weights auto-download on first call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


SD15_BASE = os.getenv(
    "POWERPAINT_SD15_BASE",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PowerPaintParams:
    task: str = "object-remove"     # object-remove | text-guided | shape-guided | context-aware | outpainting
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    fitting_degree: float = 1.0     # 1.0 = strict mask shape, lower = looser
    seed: int = 42


class PowerPaintModel:
    """Thin wrapper around iopaint's PowerPaintV2 class (bypassing ModelManager)."""

    _VALID_TASKS = {
        "object-remove",
        "text-guided",
        "shape-guided",
        "context-aware",
        "outpainting",
    }

    def __init__(self, device: Optional[torch.device] = None) -> None:
        # Lazy imports so module import stays cheap.
        from iopaint.model.power_paint import power_paint_v2 as _pp_mod
        from iopaint.model.power_paint.power_paint_v2 import PowerPaintV2
        from iopaint.model.power_paint.powerpaint_tokenizer import add_task_to_prompt
        from iopaint.schema import (
            ModelInfo, ModelType, InpaintRequest, PowerPaintTask, HDStrategy,
        )

        self._InpaintRequest = InpaintRequest
        self._PowerPaintTask = PowerPaintTask
        self._HDStrategy = HDStrategy
        self._current_prompt = ""
        self._current_negative = ""

        # iopaint's task_to_prompt() hardcodes prompt="" / negative_prompt=""
        # (powerpaint_tokenizer.py:39-42), so any user-supplied negative prompt
        # is silently dropped. Patch it to read from the wrapper's per-call
        # state so object-remove can actually reject "object / person / duplicate".
        wrapper = self

        def _task_to_prompt_with_user_prompts(task):
            promptA, promptB, negA, negB = add_task_to_prompt(
                wrapper._current_prompt, wrapper._current_negative, task
            )
            return promptA.strip(), promptB.strip(), negA.strip(), negB.strip()

        _pp_mod.task_to_prompt = _task_to_prompt_with_user_prompts

        self.device = device or pick_device()
        # PowerPaintV2 requires model_type == DIFFUSERS_SD (regular SD 1.5,
        # not inpaint variant).
        model_info = ModelInfo(
            name=SD15_BASE,
            path=SD15_BASE,
            model_type=ModelType.DIFFUSERS_SD,
        )
        self.model = PowerPaintV2(
            self.device,
            model_info=model_info,
            enable_controlnet=False,
            controlnet_method=None,
            enable_brushnet=False,
            brushnet_method=None,
            no_half=(self.device.type != "cuda"),   # fp16 only on CUDA
            low_mem=False,
            cpu_offload=False,
            sd_cpu_textencoder=False,
            disable_nsfw=True,
            local_files_only=False,
            callback=None,
        )

    def __call__(
        self,
        image_rgb: np.ndarray,
        mask_gray: np.ndarray,
        params: Optional[PowerPaintParams] = None,
    ) -> np.ndarray:
        """Inpaint.

        Args:
            image_rgb: HxWx3 uint8 RGB.
            mask_gray: HxW uint8; 255 marks the region to repaint.
            params: PowerPaint params (task, prompt, ...). Defaults to
                object-remove with empty prompt.

        Returns:
            HxWx3 uint8 RGB.
        """
        import cv2

        p = params or PowerPaintParams()
        if p.task not in self._VALID_TASKS:
            raise ValueError(
                f"Unknown task {p.task!r}. Valid: {sorted(self._VALID_TASKS)}"
            )

        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if mask_gray.dtype != np.uint8:
            mask_gray = mask_gray.astype(np.uint8)
        if mask_gray.ndim == 2:
            mask_hw1 = mask_gray[:, :, None]
        else:
            mask_hw1 = mask_gray

        # Publish per-call prompts so the monkey-patched task_to_prompt picks
        # them up (iopaint's built-in path ignores config.prompt/negative_prompt
        # for PowerPaint v2).
        self._current_prompt = p.prompt or ""
        self._current_negative = p.negative_prompt or ""

        config = self._InpaintRequest(
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            sd_steps=p.steps,
            sd_guidance_scale=p.guidance_scale,
            sd_seed=p.seed,
            sd_keep_unmasked_area=True,
            sd_match_histograms=True,
            sd_mask_blur=11,
            # Original keeps full resolution (no crop/downsample) so unmasked
            # regions don't suffer the VAE round-trip blur that CROP/RESIZE can
            # introduce at borders.
            hd_strategy=self._HDStrategy.ORIGINAL,
            enable_powerpaint_v2=True,
            powerpaint_task=self._PowerPaintTask(p.task),
            fitting_degree=p.fitting_degree,
        )

        # PowerPaintV2.forward may return float (0-255); cast to uint8 before cvtColor.
        out_bgr = self.model(image_rgb, mask_hw1, config)
        if out_bgr.dtype != np.uint8:
            out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        # Paste-back: guarantees unmasked region == original pixels.
        # 5x5 Gaussian is a seam-hiding anti-alias — keeps fade within ~2 px so
        # small masks (eyes, blemishes) don't bleed into adjacent features.
        feather = cv2.GaussianBlur(mask_gray, (5, 5), 0).astype(np.float32) / 255.0
        feather = feather[:, :, None]
        blended = out_rgb.astype(np.float32) * feather + image_rgb.astype(np.float32) * (1.0 - feather)
        return np.clip(blended, 0, 255).astype(np.uint8)
