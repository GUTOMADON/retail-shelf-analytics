"""Post-processing: turn raw YOLO detections into shelf-level insight.

The detector only knows "there is a product at (x1, y1, x2, y2)". Everything
a merchandiser actually cares about, facings per shelf, occupancy, stock
gaps, is computed here from the geometry of the detections.

Shelf rows are found with density-based clustering (DBSCAN) over each
detection's vertical center, following the row-grouping approach used by
Alijanloo/Retail-Shelf-Monitoring (see docs/PRIOR_ART.md). This replaced an
earlier design that also supported dividing the image into a fixed number of
equal-height horizontal bands: on a shelf photographed at an angle, physical
shelves are not evenly spaced in pixel space, and an equal-height cut could
land on the structural gap between two shelves and report it as EMPTY, even
though the shelf itself was full (see docs/AUDIT.md for the reproduction).

A necessary consequence of building rows only from detections: a region can
never be reported as EMPTY by this module, since there is no detection to
build a region from in a truly empty area. A completely out-of-stock shelf
cannot be told apart from "outside the photographed shelving unit" using
geometry alone. `describe_shelf_count_mismatch` surfaces that ambiguity
honestly instead of guessing.
"""

from __future__ import annotations

import statistics

import numpy as np
from sklearn.cluster import DBSCAN

from app.schemas import ComplianceSummary, Detection, RegionStatus, ShelfRegion, StockGap


def _scale_normalized_distance_matrix(detections: list[Detection]) -> np.ndarray:
    """Pairwise vertical distance, in units of the pair's own average box
    height, rather than raw pixels.

    A single global pixel threshold (for example, "the median box height
    across the whole image") is wrong under perspective: a shelf photographed
    at an angle has large boxes near the camera and small boxes far from it,
    so one scalar cannot separate rows correctly everywhere in the frame at
    once. Normalizing each pairwise distance by that pair's own box height
    makes the clustering scale-invariant across the image instead.
    """
    n = len(detections)
    centers = np.array([d.bbox.center_y for d in detections])
    heights = np.array([max(d.bbox.height, 1.0) for d in detections])

    center_diff = np.abs(centers[:, None] - centers[None, :])
    avg_height = (heights[:, None] + heights[None, :]) / 2.0
    return center_diff / avg_height


def _cluster_rows(detections: list[Detection], eps_factor: float) -> list[list[Detection]]:
    """Group detections into rows via scale-normalized density clustering."""
    if not detections:
        return []
    if len(detections) == 1:
        return [detections]

    distance_matrix = _scale_normalized_distance_matrix(detections)
    labels = DBSCAN(eps=eps_factor, min_samples=1, metric="precomputed").fit_predict(distance_matrix)

    rows_by_label: dict[int, list[Detection]] = {}
    for label, detection in zip(labels, detections):
        rows_by_label.setdefault(int(label), []).append(detection)
    return list(rows_by_label.values())


def _find_gaps(row: list[Detection], gap_width_factor: float) -> tuple[list[StockGap], float]:
    """Return the stock gaps within a row and the average facing width."""
    ordered = sorted(row, key=lambda d: d.bbox.center_x)
    avg_width = statistics.mean(d.bbox.width for d in ordered)
    if avg_width <= 0:
        # Degenerate (zero-width) boxes carry no usable spacing information.
        return [], 0.0
    threshold = avg_width * gap_width_factor

    gaps: list[StockGap] = []
    for current, nxt in zip(ordered, ordered[1:]):
        gap_width = nxt.bbox.x1 - current.bbox.x2
        if gap_width > threshold:
            gaps.append(
                StockGap(
                    x_start=current.bbox.x2,
                    x_end=nxt.bbox.x1,
                    width_px=gap_width,
                    estimated_missing_facings=max(1, round(gap_width / avg_width)),
                )
            )
    return gaps, avg_width


def _build_region(
    region_id: int,
    row: list[Detection],
    gap_width_factor: float,
    occupancy_threshold: float,
    min_detections_for_confidence: int,
) -> ShelfRegion:
    avg_confidence = statistics.mean(d.confidence for d in row)
    y_start = min(d.bbox.y1 for d in row)
    y_end = max(d.bbox.y2 for d in row)

    if len(row) < min_detections_for_confidence:
        return ShelfRegion(
            region_id=region_id,
            y_start=y_start,
            y_end=y_end,
            facing_count=len(row),
            detections=row,
            gaps=[],
            occupancy_ratio=0.0,
            avg_confidence=round(avg_confidence, 4),
            status=RegionStatus.UNKNOWN,
        )

    gaps, _ = _find_gaps(row, gap_width_factor)
    filled_width = sum(d.bbox.width for d in row)
    gap_width_total = sum(g.width_px for g in gaps)
    occupancy_ratio = filled_width / (filled_width + gap_width_total) if (filled_width + gap_width_total) > 0 else 0.0

    status = RegionStatus.UNDERSTOCKED if (occupancy_ratio < occupancy_threshold or gaps) else RegionStatus.OK

    return ShelfRegion(
        region_id=region_id,
        y_start=y_start,
        y_end=y_end,
        facing_count=len(row),
        detections=row,
        gaps=gaps,
        occupancy_ratio=round(occupancy_ratio, 4),
        avg_confidence=round(avg_confidence, 4),
        status=status,
    )


def group_into_shelf_regions(
    detections: list[Detection],
    dbscan_eps_factor: float,
    gap_width_factor: float,
    understocked_occupancy_threshold: float,
    min_detections_for_confidence: int,
) -> list[ShelfRegion]:
    """Cluster detections into shelf regions and compute occupancy per region."""
    rows = _cluster_rows(detections, dbscan_eps_factor)
    regions = [
        _build_region(
            region_id=i,
            row=row,
            gap_width_factor=gap_width_factor,
            occupancy_threshold=understocked_occupancy_threshold,
            min_detections_for_confidence=min_detections_for_confidence,
        )
        for i, row in enumerate(rows)
    ]
    regions.sort(key=lambda r: r.y_start)
    for new_id, region in enumerate(regions):
        region.region_id = new_id
    return regions


def describe_shelf_count_mismatch(regions: list[ShelfRegion], expected_shelf_count: int | None) -> str | None:
    """Explain a mismatch between the requested and the detected row count.

    Rows are only ever built from actual detections, so if fewer rows are
    found than expected, this cannot distinguish "a shelf is fully out of
    stock" from "two shelves were close enough in this photo to merge into
    one detected row". Both are reported as a note instead of guessing.
    """
    if expected_shelf_count is None:
        return None
    found = len(regions)
    if found == expected_shelf_count:
        return None
    if found < expected_shelf_count:
        missing = expected_shelf_count - found
        return (
            f"Expected {expected_shelf_count} shelves but only {found} product row(s) were detected. "
            f"The remaining {missing} may be fully out of stock (no products for the detector to anchor "
            "a row on), or two physical shelves may be close enough in this photo to have been grouped "
            "into one detected row. This cannot be told apart from geometry alone; manual review is "
            "recommended for the missing shelves."
        )
    extra = found - expected_shelf_count
    return (
        f"Expected {expected_shelf_count} shelves but {found} product rows were detected. "
        f"{extra} extra row(s) may mean a single physical shelf was split into two clusters, for example "
        "if it holds two visually distinct product groups with a wide gap between them."
    )


def summarize_compliance(regions: list[ShelfRegion]) -> ComplianceSummary:
    total_regions = len(regions)
    ok = sum(1 for r in regions if r.status == RegionStatus.OK)
    understocked = sum(1 for r in regions if r.status == RegionStatus.UNDERSTOCKED)
    empty = sum(1 for r in regions if r.status == RegionStatus.EMPTY)
    unknown = sum(1 for r in regions if r.status == RegionStatus.UNKNOWN)
    total_facings = sum(r.facing_count for r in regions)
    total_missing = sum(sum(g.estimated_missing_facings for g in r.gaps) for r in regions)

    scored_regions = [r for r in regions if r.status != RegionStatus.UNKNOWN]
    if not scored_regions:
        overall_occupancy = 0.0
    else:
        overall_occupancy = sum(r.occupancy_ratio for r in scored_regions) / len(scored_regions)

    return ComplianceSummary(
        total_regions=total_regions,
        ok_regions=ok,
        understocked_regions=understocked,
        empty_regions=empty,
        unknown_regions=unknown,
        overall_occupancy_ratio=round(overall_occupancy, 4),
        total_facings=total_facings,
        total_estimated_missing_facings=total_missing,
    )
