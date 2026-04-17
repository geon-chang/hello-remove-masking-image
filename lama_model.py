"""LaMa ONNX wrapper (Carve/LaMa-ONNX lama_fp32.onnx, fixed 512x512)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = (512, 512)


class LaMaModel:
    def __init__(self, model_path: str, providers: Optional[list] = None) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_meta = self.sess.get_inputs()
        self.input_names = [i.name for i in self.input_meta]

    def __call__(self, image_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
        """Run inpainting.

        Args:
            image_rgb: HxWx3 uint8 RGB.
            mask_gray: HxW uint8, 255 marks the region to remove.

        Returns:
            HxWx3 uint8 RGB at the original resolution.
        """
        orig_h, orig_w = image_rgb.shape[:2]
        img = cv2.resize(image_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask_gray, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)

        img_f = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # (1,3,H,W)
        msk_f = (msk.astype(np.float32) / 255.0)[None, None]                # (1,1,H,W)
        msk_f = (msk_f > 0.5).astype(np.float32)

        feeds = self._feeds(img_f, msk_f)
        out = self.sess.run(None, feeds)[0]
        if out.ndim == 4:
            out = out[0]
        out = np.transpose(out, (1, 2, 0))
        if out.max() <= 1.0 + 1e-4:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    def _feeds(self, img: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        if len(self.input_names) == 2:
            feeds: dict[str, np.ndarray] = {}
            for name in self.input_names:
                lname = name.lower()
                if "image" in lname:
                    feeds[name] = img
                elif "mask" in lname:
                    feeds[name] = mask
            if len(feeds) != 2:
                feeds = {self.input_names[0]: img, self.input_names[1]: mask}
            return feeds
        if len(self.input_names) == 1:
            x = np.concatenate([img, mask], axis=1).astype(np.float32)
            return {self.input_names[0]: x}
        raise RuntimeError(f"Unsupported input count: {len(self.input_names)}")
