"""Pydantic models shared across the detection and analysis pipeline.

These models define the exact JSON contract returned by the API, which the
frontend types in ``frontend/src/types.ts`` mirror.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


class Detection(BaseModel):
    """A single product facing found by the object detector."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


class RegionStatus(str, Enum):
    OK = "ok"
    UNDERSTOCKED = "understocked"
    EMPTY = "empty"


class StockGap(BaseModel):
    """A horizontal gap within a shelf region that likely indicates missing
    stock between two detected facings (or between a facing and the shelf
    edge)."""

    x_start: float
    x_end: float
    width_px: float
    estimated_missing_facings: int


class ShelfRegion(BaseModel):
    """A horizontal band of the image corresponding to one physical shelf."""

    region_id: int
    y_start: float
    y_end: float
    facing_count: int
    detections: list[Detection]
    gaps: list[StockGap]
    occupancy_ratio: float = Field(ge=0.0, le=1.0)
    status: RegionStatus


class ComplianceSummary(BaseModel):
    total_regions: int
    ok_regions: int
    understocked_regions: int
    empty_regions: int
    overall_occupancy_ratio: float = Field(ge=0.0, le=1.0)
    total_facings: int
    total_estimated_missing_facings: int


class AnalysisConfig(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    row_gap_factor: float
    gap_width_factor: float
    understocked_occupancy_threshold: float


class AnalysisReport(BaseModel):
    """The full result of analyzing one shelf photo."""

    image_width: int
    image_height: int
    config: AnalysisConfig
    shelf_regions: list[ShelfRegion]
    compliance: ComplianceSummary
    annotated_image_base64: str
    """PNG image, base64-encoded, with bounding boxes, region separators and
    gap highlights drawn over the original photo."""
