"""Unit tests for the shelf-region grouping and gap-detection logic.

These tests use synthetic detections instead of running YOLO, since the
post-processing logic in ``app.shelf_analysis`` is pure geometry and should
be verifiable without a model or GPU.
"""

from app.schemas import BoundingBox, Detection, RegionStatus
from app.shelf_analysis import group_into_shelf_regions, summarize_compliance


def _det(x1: float, y1: float, x2: float, y2: float, name: str = "product") -> Detection:
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
        class_id=0,
        class_name=name,
    )


def test_dynamic_clustering_splits_two_rows():
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100, 150)]
    bottom_row = [_det(x, 200, x + 40, 250) for x in (0, 50, 100, 150)]

    regions = group_into_shelf_regions(
        detections=top_row + bottom_row,
        image_width=400,
        image_height=300,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
    )

    assert len(regions) == 2
    assert regions[0].y_start < regions[1].y_start
    assert regions[0].facing_count == 4
    assert regions[1].facing_count == 4


def test_gap_detection_flags_understocked_row():
    # Four facings tightly packed, then a wide gap before the last facing.
    row = [
        _det(0, 0, 40, 50),
        _det(40, 0, 80, 50),
        _det(80, 0, 120, 50),
        _det(300, 0, 340, 50),  # far away -> gap
    ]

    regions = group_into_shelf_regions(
        detections=row,
        image_width=400,
        image_height=100,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.status == RegionStatus.UNDERSTOCKED
    assert len(region.gaps) == 1
    assert region.gaps[0].estimated_missing_facings >= 4


def test_fixed_bands_detect_fully_empty_shelf():
    # Only populate the top third of a 300px-tall image split into 3 bands.
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100)]

    regions = group_into_shelf_regions(
        detections=top_row,
        image_width=400,
        image_height=300,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
        expected_shelf_count=3,
    )

    assert len(regions) == 3
    assert regions[0].facing_count == 3
    assert regions[1].status == RegionStatus.EMPTY
    assert regions[2].status == RegionStatus.EMPTY


def test_summarize_compliance_counts_statuses():
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100)]

    regions = group_into_shelf_regions(
        detections=top_row,
        image_width=400,
        image_height=300,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
        expected_shelf_count=3,
    )
    summary = summarize_compliance(regions)

    assert summary.total_regions == 3
    assert summary.empty_regions == 2
    assert summary.total_facings == 3


def test_no_detections_returns_no_dynamic_regions():
    regions = group_into_shelf_regions(
        detections=[],
        image_width=400,
        image_height=300,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
    )
    assert regions == []
    summary = summarize_compliance(regions)
    assert summary.total_regions == 0
    assert summary.overall_occupancy_ratio == 0.0
