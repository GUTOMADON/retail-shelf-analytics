"""Post-processing: turn raw YOLO detections into shelf-level insight.

The detector only knows "there is a product at (x1, y1, x2, y2)". Everything
a merchandiser actually cares about — facings per shelf, empty slots,
under-stocked sections — is computed here from the geometry of the
detections.

Two grouping strategies are supported:

- **Fixed bands** (``expected_shelf_count`` is given): the image is split
  into that many equal-height horizontal bands. This is the only way to
  catch a shelf that is *completely* empty, since a band with zero
  detections is still a region.
- **Dynamic clustering** (default): detections are clustered by vertical
  position using a gap threshold. This adapts to shelves photographed at a
  slight angle or with uneven spacing, but cannot flag a shelf that produced
  no detections at all, since no data point exists to anchor a region.
"""

from __future__ import annotations

import statistics

from app.schemas import ComplianceSummary, Detection, RegionStatus, ShelfRegion, StockGap


def _median_box_height(detections: list[Detection]) -> float:
    heights = [d.bbox.height for d in detections]
    return statistics.median(heights) if heights else 1.0


def _cluster_rows_dynamic(detections: list[Detection], row_gap_factor: float) -> list[list[Detection]]:
    ordered = sorted(detections, key=lambda d: d.bbox.center_y)
    if not ordered:
        return []

    threshold = _median_box_height(ordered) * row_gap_factor
    rows: list[list[Detection]] = [[ordered[0]]]
    for current, nxt in zip(ordered, ordered[1:]):
        gap = nxt.bbox.center_y - current.bbox.center_y
        if gap > threshold:
            rows.append([])
        rows[-1].append(nxt)
    return rows


def _cluster_rows_fixed_bands(
    detections: list[Detection], image_height: int, expected_shelf_count: int
) -> list[list[Detection]]:
    band_height = image_height / expected_shelf_count
    rows: list[list[Detection]] = [[] for _ in range(expected_shelf_count)]
    for det in detections:
        band_index = int(det.bbox.center_y // band_height)
        band_index = min(max(band_index, 0), expected_shelf_count - 1)
        rows[band_index].append(det)
    return rows


def _find_gaps(row: list[Detection], gap_width_factor: float) -> tuple[list[StockGap], float]:
    """Return the stock gaps within a row and the average facing width."""
    ordered = sorted(row, key=lambda d: d.bbox.center_x)
    avg_width = statistics.mean(d.bbox.width for d in ordered)
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


def _build_region(region_id: int, row: list[Detection], gap_width_factor: float, occupancy_threshold: float, y_bounds: tuple[float, float] | None = None) -> ShelfRegion:
    if not row:
        y_start, y_end = y_bounds if y_bounds else (0.0, 0.0)
        return ShelfRegion(
            region_id=region_id,
            y_start=y_start,
            y_end=y_end,
            facing_count=0,
            detections=[],
            gaps=[],
            occupancy_ratio=0.0,
            status=RegionStatus.EMPTY,
        )

    gaps, avg_width = _find_gaps(row, gap_width_factor)
    filled_width = sum(d.bbox.width for d in row)
    gap_width_total = sum(g.width_px for g in gaps)
    occupancy_ratio = filled_width / (filled_width + gap_width_total) if (filled_width + gap_width_total) > 0 else 0.0

    y_start = y_bounds[0] if y_bounds else min(d.bbox.y1 for d in row)
    y_end = y_bounds[1] if y_bounds else max(d.bbox.y2 for d in row)

    status = RegionStatus.OK
    if occupancy_ratio < occupancy_threshold or gaps:
        status = RegionStatus.UNDERSTOCKED

    return ShelfRegion(
        region_id=region_id,
        y_start=y_start,
        y_end=y_end,
        facing_count=len(row),
        detections=row,
        gaps=gaps,
        occupancy_ratio=round(occupancy_ratio, 4),
        status=status,
    )


def group_into_shelf_regions(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    row_gap_factor: float,
    gap_width_factor: float,
    understocked_occupancy_threshold: float,
    expected_shelf_count: int | None = None,
) -> list[ShelfRegion]:
    """Group detections into shelf regions and compute occupancy per region."""
    if expected_shelf_count and expected_shelf_count > 0:
        rows = _cluster_rows_fixed_bands(detections, image_height, expected_shelf_count)
        band_height = image_height / expected_shelf_count
        regions = [
            _build_region(
                region_id=i,
                row=row,
                gap_width_factor=gap_width_factor,
                occupancy_threshold=understocked_occupancy_threshold,
                y_bounds=(i * band_height, (i + 1) * band_height),
            )
            for i, row in enumerate(rows)
        ]
    else:
        rows = _cluster_rows_dynamic(detections, row_gap_factor)
        regions = [
            _build_region(
                region_id=i,
                row=row,
                gap_width_factor=gap_width_factor,
                occupancy_threshold=understocked_occupancy_threshold,
            )
            for i, row in enumerate(rows)
        ]

    return sorted(regions, key=lambda r: r.y_start)


def summarize_compliance(regions: list[ShelfRegion]) -> ComplianceSummary:
    total_regions = len(regions)
    ok = sum(1 for r in regions if r.status == RegionStatus.OK)
    understocked = sum(1 for r in regions if r.status == RegionStatus.UNDERSTOCKED)
    empty = sum(1 for r in regions if r.status == RegionStatus.EMPTY)
    total_facings = sum(r.facing_count for r in regions)
    total_missing = sum(sum(g.estimated_missing_facings for g in r.gaps) for r in regions)

    if total_regions == 0:
        overall_occupancy = 0.0
    else:
        overall_occupancy = sum(r.occupancy_ratio for r in regions) / total_regions

    return ComplianceSummary(
        total_regions=total_regions,
        ok_regions=ok,
        understocked_regions=understocked,
        empty_regions=empty,
        overall_occupancy_ratio=round(overall_occupancy, 4),
        total_facings=total_facings,
        total_estimated_missing_facings=total_missing,
    )
