"""MI-GAN ONNX pipeline_v2 wrapper.

The ONNX pipeline accepts arbitrary-resolution uint8 inputs and returns a full
original-resolution inpainted image. Preprocessing (crop around mask, resize,
normalize) and postprocessing (paste back with blended mask) are embedded in
the graph, so this wrapper only handles I/O plumbing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnxruntime as ort


class MIGANModel:
    def __init__(self, model_path: str, providers: Optional[list] = None) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def __call__(
        self,
        image_rgb: np.ndarray,
        mask_gray: np.ndarray,
        invert_mask: bool = True,
    ) -> np.ndarray:
        """Run inpainting.

        Args:
            image_rgb: HxWx3 uint8, RGB.
            mask_gray: HxW uint8; by UI convention 255 marks the region to
                remove.
            invert_mask: If True (default), invert so the model receives
                MI-GAN's 255=known / 0=masked convention.

        Returns:
            HxWx3 uint8 RGB result at the original resolution.
        """
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if mask_gray.dtype != np.uint8:
            mask_gray = mask_gray.astype(np.uint8)
        if invert_mask:
            mask_gray = 255 - mask_gray

        img_nchw = np.transpose(image_rgb, (2, 0, 1))[None]  # (1,3,H,W)
        msk_nchw = mask_gray[None, None]                      # (1,1,H,W)

        feeds: dict[str, np.ndarray] = {}
        for name in self.input_names:
            lname = name.lower()
            if "image" in lname:
                feeds[name] = img_nchw
            elif "mask" in lname:
                feeds[name] = msk_nchw
        if len(feeds) != 2:
            feeds = {self.input_names[0]: img_nchw, self.input_names[1]: msk_nchw}

        out = self.sess.run(None, feeds)[0]
        if out.ndim == 4:
            out = out[0]
        return np.transpose(out, (1, 2, 0)).astype(np.uint8)
