"""Unit tests for detection filtering and deduplication.

Both functions under test are pure geometry/set operations with no
dependency on the YOLO model itself, so they are tested directly with
synthetic Detection objects.
"""

from app.detection import (
    deduplicate_same_class_boxes,
    filter_by_class,
    suppress_contained_boxes,
    suppress_overlapping_boxes,
)
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


def test_suppress_overlapping_boxes_drops_lower_confidence_duplicate():
    """Two different classes describing the same physical object: Ultralytics
    only runs NMS within a class, so this is the case it cannot catch."""
    high_conf = _det(10, 10, 60, 90, name="bottle", confidence=0.8)
    low_conf_duplicate = _det(12, 11, 61, 88, name="vase", confidence=0.4)

    kept = suppress_overlapping_boxes([low_conf_duplicate, high_conf], iou_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].class_name == "bottle"


def test_suppress_overlapping_boxes_keeps_adjacent_non_duplicate_boxes():
    """Two bottles standing side by side, touching but not the same object,
    must both survive."""
    left = _det(0, 0, 40, 100)
    right = _det(39, 0, 79, 100)

    kept = suppress_overlapping_boxes([left, right], iou_threshold=0.5)

    assert len(kept) == 2


def test_suppress_overlapping_boxes_empty_input():
    assert suppress_overlapping_boxes([], iou_threshold=0.5) == []


def test_deduplicate_same_class_boxes_collapses_overlap_above_floor():
    """Two same-class boxes at IoU ~0.35, below Ultralytics' own per-class
    NMS threshold but clearly the same physical object, collapse to one."""
    high_conf = _det(0, 0, 10, 10, name="bottle", confidence=0.8)
    low_conf = _det(4.8, 0, 14.8, 10, name="bottle", confidence=0.5)

    kept = deduplicate_same_class_boxes([high_conf, low_conf], iou_threshold=0.3)

    assert len(kept) == 1
    assert kept[0].confidence == 0.8


def test_deduplicate_same_class_boxes_keeps_different_classes_at_same_iou():
    """The same overlap geometry across two different classes is left alone;
    that case belongs to suppress_overlapping_boxes, not this pass."""
    bottle = _det(0, 0, 10, 10, name="bottle", confidence=0.8)
    cup = _det(4.8, 0, 14.8, 10, name="cup", confidence=0.5)

    kept = deduplicate_same_class_boxes([bottle, cup], iou_threshold=0.3)

    assert len(kept) == 2


def test_deduplicate_same_class_boxes_keeps_both_below_floor():
    a = _det(0, 0, 10, 10, name="bottle", confidence=0.8)
    b = _det(8, 0, 18, 10, name="bottle", confidence=0.5)

    kept = deduplicate_same_class_boxes([a, b], iou_threshold=0.3)

    assert len(kept) == 2
