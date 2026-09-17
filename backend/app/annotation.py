"""Render an annotated shelf image from an analysis result.

Draws, on top of the original photo: one bounding box per detected facing,
a dashed separator between shelf regions, and a highlighted rectangle over
every detected stock gap. The result is what the frontend shows the user.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import RegionStatus, ShelfRegion

_STATUS_COLORS: dict[RegionStatus, tuple[int, int, int]] = {
    RegionStatus.OK: (34, 197, 94),          # green
    RegionStatus.UNDERSTOCKED: (245, 158, 11),  # amber
    RegionStatus.EMPTY: (239, 68, 68),        # red
}
_GAP_COLOR = (239, 68, 68)
_BOX_COLOR = (59, 130, 246)  # blue


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_annotations(image_rgb: np.ndarray, regions: list[ShelfRegion]) -> bytes:
    """Draw detections, region bands, and gaps; return a PNG byte string."""
    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(max(12, image.width // 90))
    label_font = _load_font(max(14, image.width // 70))

    for region in regions:
        color = _STATUS_COLORS[region.status]

        # Region band on the left margin as a status indicator.
        draw.rectangle(
            [0, region.y_start, 10, region.y_end],
            fill=(*color, 200),
        )
        draw.line(
            [(0, region.y_start), (image.width, region.y_start)],
            fill=(*color, 120),
            width=2,
        )

        label = f"Shelf {region.region_id + 1} · {region.facing_count} facings · {region.status.value.upper()}"
        draw.rectangle(
            [12, region.y_start + 4, 12 + draw.textlength(label, font=label_font) + 12, region.y_start + 28],
            fill=(0, 0, 0, 160),
        )
        draw.text((18, region.y_start + 6), label, fill=(255, 255, 255, 255), font=label_font)

        for det in region.detections:
            b = det.bbox
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=_BOX_COLOR, width=3)
            tag = f"{det.class_name} {det.confidence:.2f}"
            draw.rectangle(
                [b.x1, max(0, b.y1 - 18), b.x1 + draw.textlength(tag, font=font) + 6, b.y1],
                fill=(*_BOX_COLOR, 200),
            )
            draw.text((b.x1 + 3, max(0, b.y1 - 17)), tag, fill=(255, 255, 255, 255), font=font)

        for gap in region.gaps:
            draw.rectangle(
                [gap.x_start, region.y_start + 4, gap.x_end, region.y_end - 4],
                outline=_GAP_COLOR,
                width=3,
                fill=(*_GAP_COLOR, 60),
            )
            gap_label = f"gap · ~{gap.estimated_missing_facings} missing"
            draw.text(
                (gap.x_start + 4, region.y_start + 32),
                gap_label,
                fill=_GAP_COLOR,
                font=font,
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
