"""Unit tests for the shelf-region grouping and gap-detection logic.

These tests use synthetic detections instead of running YOLO, since the
post-processing logic in ``app.shelf_analysis`` is pure geometry and should
be verifiable without a model or GPU.
"""

from app.schemas import BoundingBox, Detection, RegionStatus
from app.shelf_analysis import describe_shelf_count_mismatch, group_into_shelf_regions, summarize_compliance

DBSCAN_EPS_FACTOR = 0.25
GAP_WIDTH_FACTOR = 1.35
OCCUPANCY_THRESHOLD = 0.75
MIN_DETECTIONS_FOR_CONFIDENCE = 2


def _det(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, name: str = "bottle") -> Detection:
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=confidence,
        class_id=0,
        class_name=name,
    )


def _group(detections: list[Detection]):
    return group_into_shelf_regions(
        detections=detections,
        dbscan_eps_factor=DBSCAN_EPS_FACTOR,
        gap_width_factor=GAP_WIDTH_FACTOR,
        understocked_occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_detections_for_confidence=MIN_DETECTIONS_FOR_CONFIDENCE,
    )


def test_clustering_splits_two_rows():
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100, 150)]
    bottom_row = [_det(x, 200, x + 40, 250) for x in (0, 50, 100, 150)]

    regions = _group(top_row + bottom_row)

    assert len(regions) == 2
    assert regions[0].y_start < regions[1].y_start
    assert regions[0].facing_count == 4
    assert regions[1].facing_count == 4


def test_clustering_never_invents_an_empty_row():
    """A fixed equal-height band could land on a structural gap between two
    physical shelves and report it as EMPTY even when both shelves are
    full (see docs/AUDIT.md). Density-based clustering cannot do this,
    since a region can only be built from detections that exist."""
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100, 150)]
    bottom_row = [_det(x, 500, x + 40, 550) for x in (0, 50, 100, 150)]

    regions = _group(top_row + bottom_row)

    assert len(regions) == 2
    assert all(r.status != RegionStatus.EMPTY for r in regions)
    assert all(r.facing_count > 0 for r in regions)


def test_gap_detection_flags_understocked_row():
    row = [
        _det(0, 0, 40, 50),
        _det(40, 0, 80, 50),
        _det(80, 0, 120, 50),
        _det(300, 0, 340, 50),
    ]

    regions = _group(row)

    assert len(regions) == 1
    region = regions[0]
    assert region.status == RegionStatus.UNDERSTOCKED
    assert len(region.gaps) == 1
    assert region.gaps[0].estimated_missing_facings >= 4


def test_region_with_too_few_detections_is_unknown():
    lone_detection = [_det(0, 0, 40, 50)]

    regions = _group(lone_detection)

    assert len(regions) == 1
    assert regions[0].status == RegionStatus.UNKNOWN
    assert regions[0].gaps == []


def test_low_confidence_detections_still_grouped_but_status_reflects_evidence():
    row = [_det(x, 0, x + 40, 50, confidence=0.15) for x in (0, 50, 100)]

    regions = _group(row)

    assert len(regions) == 1
    assert regions[0].avg_confidence == 0.15
    assert regions[0].status in {RegionStatus.OK, RegionStatus.UNDERSTOCKED}


def test_summarize_compliance_counts_statuses():
    top_row = [_det(x, 10, x + 40, 60) for x in (0, 50, 100)]
    lone = [_det(0, 500, 40, 550)]

    regions = _group(top_row + lone)
    summary = summarize_compliance(regions)

    assert summary.total_regions == 2
    assert summary.unknown_regions == 1
    assert summary.total_facings == 4


def test_no_detections_returns_no_regions():
    regions = _group([])
    assert regions == []
    summary = summarize_compliance(regions)
    assert summary.total_regions == 0
    assert summary.overall_occupancy_ratio == 0.0


def test_shelf_count_mismatch_reports_missing_rows():
    regions = _group([_det(x, 10, x + 40, 60) for x in (0, 50, 100)])
    note = describe_shelf_count_mismatch(regions, expected_shelf_count=3)
    assert note is not None
    assert "1" in note


def test_shelf_count_match_returns_no_note():
    regions = _group([_det(x, 10, x + 40, 60) for x in (0, 50, 100)])
    note = describe_shelf_count_mismatch(regions, expected_shelf_count=1)
    assert note is None


def test_shelf_count_hint_not_given_returns_no_note():
    regions = _group([_det(0, 10, 40, 60)])
    assert describe_shelf_count_mismatch(regions, expected_shelf_count=None) is None


def test_zero_width_boxes_do_not_crash_gap_estimation():
    """Regression test: degenerate (zero-width) boxes used to raise
    ZeroDivisionError when computing the average facing width for gap
    detection. Malformed detections should degrade gracefully instead."""
    row = [_det(5, 5, 5, 50), _det(500, 5, 500, 50)]
    regions = _group(row)
    assert len(regions) == 1
    assert regions[0].gaps == []


def test_single_pixel_image_extent_detection():
    """A detection at the extreme edge of a tiny image should not crash
    clustering or region building."""
    regions = _group([_det(0, 0, 1, 1)])
    assert len(regions) == 1
    assert regions[0].status == RegionStatus.UNKNOWN
