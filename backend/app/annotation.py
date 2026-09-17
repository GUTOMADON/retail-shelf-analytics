"""Render an annotated shelf image from an analysis result.

Two render modes are supported:

- Clean view (default): a colored left-edge tag per shelf, plain product
  boxes with no per-box label, and a slim gap marker sized to the actual
  gap height. Meant to be readable as a standalone result image, and to be
  the one used in documentation and demos.
- Debug view: adds per-box class and confidence labels and a full-width
  status banner per shelf. Meant for development and troubleshooting, not
  for a first-look demo image, since labels can overlap on dense shelves.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import RegionStatus, ShelfRegion, StockGap

_STATUS_COLORS: dict[RegionStatus, tuple[int, int, int]] = {
    RegionStatus.OK: (34, 197, 94),
    RegionStatus.UNDERSTOCKED: (245, 158, 11),
    RegionStatus.EMPTY: (239, 68, 68),
    RegionStatus.UNKNOWN: (148, 163, 184),
}
_GAP_COLOR = (239, 68, 68)
_BOX_COLOR = (59, 130, 246)
_MARGIN_WIDTH = 14


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


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


def draw_annotations(image_rgb: np.ndarray, regions: list[ShelfRegion], debug: bool = False) -> bytes:
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

        for det in region.detections:
            b = det.bbox
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=_BOX_COLOR, width=2)
            if debug:
                tag = f"{det.class_name} {det.confidence:.2f}"
                draw.text((b.x1 + 2, max(0, b.y1 - 14)), tag, fill=_BOX_COLOR, font=box_font)

        for gap in region.gaps:
            _draw_gap_marker(draw, region, gap, box_font, debug)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
