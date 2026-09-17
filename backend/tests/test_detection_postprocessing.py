"""Unit tests for detection filtering and deduplication.

Both functions under test are pure geometry/set operations with no
dependency on the YOLO model itself, so they are tested directly with
synthetic Detection objects.
"""

from app.detection import filter_by_class, suppress_contained_boxes
from app.schemas import BoundingBox, Detection


def _det(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, name: str = "bottle") -> Detection:
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=confidence,
        class_id=0,
        class_name=name,
    )


def test_filter_by_class_drops_non_allowlisted_classes():
    detections = [
        _det(0, 0, 10, 10, name="bottle"),
        _det(0, 0, 10, 10, name="refrigerator"),
        _det(0, 0, 10, 10, name="person"),
    ]
    kept = filter_by_class(detections, frozenset({"bottle"}))
    assert len(kept) == 1
    assert kept[0].class_name == "bottle"


def test_filter_by_class_keeps_all_when_all_allowlisted():
    detections = [_det(0, 0, 10, 10, name="bottle"), _det(20, 0, 30, 10, name="cup")]
    kept = filter_by_class(detections, frozenset({"bottle", "cup"}))
    assert len(kept) == 2


def test_suppress_contained_boxes_drops_box_engulfed_by_larger_one():
    """Reproduces the reported bug where a large mis-classified box (e.g.
    "refrigerator") swallows real product boxes underneath it."""
    big_false_positive = _det(0, 0, 200, 200, name="refrigerator", confidence=0.68)
    real_products = [
        _det(10, 10, 50, 50, name="bottle", confidence=0.8),
        _det(60, 10, 100, 50, name="bottle", confidence=0.75),
    ]

    kept = suppress_contained_boxes([big_false_positive] + real_products, containment_threshold=0.85)

    kept_names = {d.class_name for d in kept}
    assert "refrigerator" not in kept_names
    assert len(kept) == 2


def test_suppress_contained_boxes_keeps_non_overlapping_boxes():
    detections = [_det(0, 0, 10, 10), _det(100, 100, 110, 110), _det(200, 200, 210, 210)]
    kept = suppress_contained_boxes(detections, containment_threshold=0.85)
    assert len(kept) == 3


def test_suppress_contained_boxes_drops_near_duplicate():
    a = _det(0, 0, 50, 50, confidence=0.7)
    near_duplicate = _det(2, 2, 52, 52, confidence=0.6)
    kept = suppress_contained_boxes([a, near_duplicate], containment_threshold=0.85)
    assert len(kept) == 1


def test_suppress_contained_boxes_empty_input():
    assert suppress_contained_boxes([], containment_threshold=0.85) == []
