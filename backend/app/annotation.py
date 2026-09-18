"""Render an annotated shelf image from an analysis result.

Two render modes are supported:

- Clean view (default): a colored left-edge tag per shelf, plain product
  boxes with no per-box label, and a slim gap marker sized to the actual
  gap height. Meant to be readable as a standalone result image, and to be
  the one used in documentation and demos.
- Debug view: adds per-box class and confidence labels and a full-width
  status banner per shelf. Meant for development and troubleshooting, not
  for a first-look demo image. Labels get a dark backdrop for contrast and
  an anti-collision pass that nudges or drops one that would overlap a
  label already drawn in the same region, since a dense shelf row can pack
  boxes closer together than their labels' text width.

When a ComplianceSummary is passed in, a compact executive summary card
(occupancy, facing count, gap count, status breakdown) is drawn in the
bottom-right corner in both views, so a single exported image is readable
on its own without the surrounding dashboard.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import ComplianceSummary, RegionStatus, ShelfRegion, StockGap

_STATUS_COLORS: dict[RegionStatus, tuple[int, int, int]] = {
    RegionStatus.OK: (34, 197, 94),
    RegionStatus.UNDERSTOCKED: (245, 158, 11),
    RegionStatus.EMPTY: (239, 68, 68),
    RegionStatus.UNKNOWN: (148, 163, 184),
}
_GAP_COLOR = (239, 68, 68)
_BOX_COLOR = (59, 130, 246)
_MARGIN_WIDTH = 14
_LABEL_BACKDROP_COLOR = (0, 0, 0, 190)

_LABEL_NUDGE_ATTEMPTS = 4
"""Vertical steps tried before a colliding debug label is dropped instead of
drawn on top of another one. A dense shelf row rarely stacks more than a
handful of boxes at the same x-position, so a small, fixed budget is enough
without scanning the whole image for free space."""


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _place_label(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    placed: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Find a vertical slot for a label that does not overlap one already
    drawn in this region.

    Boxes on a dense shelf can sit close enough together that their
    confidence labels collide even though the boxes themselves do not,
    since a label is drawn just above its box rather than inside it. The
    label is nudged straight down, one label-height at a time; if every
    attempt still collides, the caller skips it rather than render
    unreadable overlapping text.
    """
    for step in range(_LABEL_NUDGE_ATTEMPTS + 1):
        y = origin_y + step * height
        candidate = (origin_x, y, origin_x + width, y + height)
        if not any(_rects_overlap(candidate, other) for other in placed):
            return candidate
    return None


def _draw_region_margin(draw: ImageDraw.ImageDraw, region: ShelfRegion, font: ImageFont.ImageFont) -> None:
    """Compact status indicator confined to the left margin, so it never
    overlaps the products themselves regardless of how the region's
    boundaries align with the photo."""
    color = _STATUS_COLORS[region.status]
    draw.rectangle([0, region.y_start, _MARGIN_WIDTH, region.y_end], fill=(*color, 220))

    label = str(region.region_id + 1)
    text_width = draw.textlength(label, font=font)
    text_x = (_MARGIN_WIDTH - text_width) / 2
    text_y = region.y_start + 3
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)


def _draw_gap_marker(draw: ImageDraw.ImageDraw, region: ShelfRegion, gap: StockGap, font: ImageFont.ImageFont, debug: bool) -> None:
    """Slim marker sized to the row's own height, not the full region."""
    row_height = region.y_end - region.y_start
    marker_height = min(row_height, max(24.0, row_height * 0.35))
    center_y = (region.y_start + region.y_end) / 2
    top = center_y - marker_height / 2
    bottom = center_y + marker_height / 2

    draw.rectangle([gap.x_start, top, gap.x_end, bottom], outline=_GAP_COLOR, width=2, fill=(*_GAP_COLOR, 45))
    label = f"~{gap.estimated_missing_facings} missing" if debug else f"~{gap.estimated_missing_facings}"
    draw.text((gap.x_start + 3, top + 2), label, fill=_GAP_COLOR, font=font)


def _draw_summary_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    compliance: ComplianceSummary,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    """Compact executive summary card in the bottom-right corner: occupancy,
    facings, gaps, and a status breakdown. Meant to make a single exported
    Clean view image readable on its own, without the surrounding dashboard.
    """
    lines = [
        f"{compliance.overall_occupancy_ratio * 100:.0f}% occupancy",
        f"{compliance.total_facings} facings across {compliance.total_regions} shelves",
        f"{compliance.total_estimated_missing_facings} est. missing facings",
    ]
    status_counts = [
        (_STATUS_COLORS[RegionStatus.OK], f"{compliance.ok_regions} OK"),
        (_STATUS_COLORS[RegionStatus.UNDERSTOCKED], f"{compliance.understocked_regions} under-stocked"),
        (_STATUS_COLORS[RegionStatus.EMPTY], f"{compliance.empty_regions} empty"),
        (_STATUS_COLORS[RegionStatus.UNKNOWN], f"{compliance.unknown_regions} unknown"),
    ]
    status_counts = [(color, text) for color, text in status_counts if not text.startswith("0 ")]

    padding = 14
    line_height = body_font.size + 6
    title = "SHELF ANALYSIS"
    title_height = title_font.size + 10
    panel_height = padding * 2 + title_height + len(lines) * line_height + len(status_counts) * line_height
    text_widths = [draw.textlength(t, font=body_font) for t in lines + [s[1] for s in status_counts]]
    text_widths.append(draw.textlength(title, font=title_font))
    panel_width = max(text_widths) + padding * 2 + 16

    x2, y2 = image.width - 16, image.height - 16
    x1, y1 = x2 - panel_width, y2 - panel_height

    draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=(15, 23, 42, 215))
    cursor_y = y1 + padding
    draw.text((x1 + padding, cursor_y), title, fill=(148, 197, 255, 255), font=title_font)
    cursor_y += title_height

    for line in lines:
        draw.text((x1 + padding, cursor_y), line, fill=(255, 255, 255, 255), font=body_font)
        cursor_y += line_height

    for color, text in status_counts:
        dot_r = 4
        dot_cy = cursor_y + body_font.size / 2
        draw.ellipse([x1 + padding, dot_cy - dot_r, x1 + padding + dot_r * 2, dot_cy + dot_r], fill=(*color, 255))
        draw.text((x1 + padding + dot_r * 2 + 6, cursor_y), text, fill=(226, 232, 240, 255), font=body_font)
        cursor_y += line_height


def draw_annotations(
    image_rgb: np.ndarray,
    regions: list[ShelfRegion],
    debug: bool = False,
    compliance: ComplianceSummary | None = None,
) -> bytes:
    """Draw detections, shelf indicators, and gap markers; return PNG bytes."""
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    margin_font = _load_font(max(11, image.width // 110))
    box_font = _load_font(max(11, image.width // 100))
    banner_font = _load_font(max(14, image.width // 70))

    for region in regions:
        _draw_region_margin(draw, region, margin_font)

        if debug:
            color = _STATUS_COLORS[region.status]
            draw.line([(0, region.y_start), (image.width, region.y_start)], fill=(*color, 100), width=1)
            banner = f"Shelf {region.region_id + 1} | {region.facing_count} facings | {region.status.value.upper()} | conf {region.avg_confidence:.2f}"
            draw.rectangle(
                [_MARGIN_WIDTH + 4, region.y_start + 2, _MARGIN_WIDTH + 8 + draw.textlength(banner, font=banner_font), region.y_start + 22],
                fill=(0, 0, 0, 150),
            )
            draw.text((_MARGIN_WIDTH + 8, region.y_start + 4), banner, fill=(255, 255, 255, 255), font=banner_font)

        placed_label_boxes: list[tuple[float, float, float, float]] = []
        for det in region.detections:
            b = det.bbox
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=_BOX_COLOR, width=2)
            if debug:
                tag = f"{det.class_name} {det.confidence:.2f}"
                text_width = draw.textlength(tag, font=box_font)
                text_height = box_font.size + 4
                label_rect = _place_label(b.x1 + 2, max(0, b.y1 - 14), text_width, text_height, placed_label_boxes)
                if label_rect is not None:
                    x1, y1, x2, y2 = label_rect
                    draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], fill=_LABEL_BACKDROP_COLOR)
                    draw.text((x1, y1), tag, fill=_BOX_COLOR, font=box_font)
                    placed_label_boxes.append(label_rect)

        for gap in region.gaps:
            _draw_gap_marker(draw, region, gap, box_font, debug)

    if compliance is not None:
        summary_title_font = _load_font(max(13, image.width // 90))
        summary_body_font = _load_font(max(12, image.width // 100))
        _draw_summary_panel(image, draw, compliance, summary_title_font, summary_body_font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
