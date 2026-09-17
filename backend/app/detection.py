"""YOLOv8 object detection wrapper.

Wraps an Ultralytics YOLOv8 model behind a small, typed interface so the rest
of the pipeline never touches the ``ultralytics`` API directly. This keeps
the detector swappable (e.g. for a shelf-specific fine-tuned checkpoint)
without changing any downstream code.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from ultralytics import YOLO

from app.config import get_settings
from app.schemas import BoundingBox, Detection


class ShelfDetector:
    """Thin, typed wrapper around a YOLOv8 model for shelf-image inference."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self._model = YOLO(model_path)
        self._device = device

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        """Run inference on a BGR ``numpy`` image and return detections.

        Args:
            image: Image array as loaded by OpenCV (``H x W x 3``, BGR).
            confidence_threshold: Minimum detection confidence to keep.
            iou_threshold: IoU threshold used for non-max suppression.
        """
        results = self._model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(box.conf[0].item()),
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                    )
                )
        return detections


@lru_cache
def get_detector() -> ShelfDetector:
    """Return a process-wide singleton detector so model weights are loaded
    into memory only once."""
    settings = get_settings()
    return ShelfDetector(model_path=settings.yolo_model_path, device=settings.device)
