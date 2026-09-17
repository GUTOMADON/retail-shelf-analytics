"""YOLOv8 object detection wrapper and detection post-processing.

Wraps an Ultralytics YOLOv8 model behind a small, typed interface so the rest
of the pipeline never touches the ``ultralytics`` API directly. This keeps
the detector swappable (e.g. for a shelf-specific fine-tuned checkpoint)
without changing any downstream code.

The stock COCO-pretrained checkpoint returns all 80 COCO classes, most of
which cannot appear as a shelf product (person, refrigerator, car, ...).
Three cleanup passes run on the raw output, in order:

1. `filter_by_class` keeps only classes that plausibly are a retail product.
2. `suppress_overlapping_boxes` is a class-agnostic greedy NMS on IoU, since
   Ultralytics only runs NMS within each class.
3. `suppress_contained_boxes` removes a box that fully contains several
   other boxes (a coarse false-positive "umbrella"), which a plain IoU NMS
   pass does not catch, since the umbrella box may have low IoU with each
   individual child box while still fully enclosing all of them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.config import get_settings
from app.schemas import BoundingBox, Detection

if TYPE_CHECKING:
    # Imported for type checkers only. Ultralytics (and therefore torch) is
    # loaded lazily inside ShelfDetector so that the pure post-processing
    # helpers in this module can be imported and unit-tested without pulling
    # in the model runtime.
    from ultralytics import YOLO

RETAIL_CLASS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "bottle",
        "wine glass",
        "cup",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "book",
        "vase",
        "teddy bear",
        "handbag",
        "suitcase",
        "backpack",
    }
)
"""COCO classes that can plausibly be a packaged or loose retail product.
Structural and unrelated classes (refrigerator, person, chair, car, ...) are
dropped. This is a coarse filter, not a retail taxonomy: a fine-tuned
retail-specific detector would replace this list entirely rather than
constrain a generic one."""


def _box_area(box: BoundingBox) -> float:
    return max(0.0, box.width) * max(0.0, box.height)


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def filter_by_class(detections: list[Detection], allowlist: frozenset[str]) -> list[Detection]:
    """Keep only detections whose class is in the allowlist."""
    return [d for d in detections if d.class_name in allowlist]


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = _intersection_area(a, b)
    if intersection == 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def suppress_overlapping_boxes(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Class-agnostic greedy NMS on IoU, keeping the higher-confidence box.

    Ultralytics' own NMS runs per class, so two boxes of different classes
    that genuinely describe the same physical object (for example a bottle
    detected once as "bottle" and once as "vase") both survive. This is a
    standard greedy NMS pass, run again across all kept classes together, as
    a second line of defense behind `suppress_contained_boxes` below, which
    only catches containment rather than general high-overlap duplicates.
    """
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(_iou(candidate.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


def suppress_contained_boxes(detections: list[Detection], containment_threshold: float) -> list[Detection]:
    """Remove structural false-positive "umbrella" boxes.

    A box that contains two or more other, mutually non-overlapping boxes is
    almost certainly a coarse false positive spanning several real objects
    (for example a misclassified "refrigerator" box swallowing several real
    "bottle" boxes underneath it), so it is dropped and its children are
    kept. A box that contains exactly one other box is resolved by keeping
    whichever of the two has higher confidence, since in that case there is
    no independent evidence for which one is the umbrella.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: _box_area(d.bbox), reverse=True)
    dropped: set[int] = set()

    for outer in ordered:
        if id(outer) in dropped:
            continue
        contained = [
            inner
            for inner in ordered
            if inner is not outer
            and id(inner) not in dropped
            and _box_area(inner.bbox) > 0
            and _intersection_area(outer.bbox, inner.bbox) / _box_area(inner.bbox) >= containment_threshold
        ]
        if len(contained) >= 2:
            dropped.add(id(outer))
        elif len(contained) == 1:
            inner = contained[0]
            dropped.add(id(inner) if outer.confidence >= inner.confidence else id(outer))

    return [d for d in ordered if id(d) not in dropped]


class ShelfDetector:
    """Thin, typed wrapper around a YOLOv8 model for shelf-image inference."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        # Import ultralytics here rather than at module load. Loading it (and
        # torch) is only needed to actually run inference; keeping it out of
        # the module top level lets the pure helpers above be imported and
        # tested with no model runtime installed.
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._device = device

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        """Run inference on a BGR ``numpy`` image and return raw detections,
        before class filtering or containment suppression."""
        results = self._model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
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