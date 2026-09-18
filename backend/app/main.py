"""FastAPI application exposing the shelf-analysis pipeline.

A single endpoint, ``POST /api/analyze``, accepts a shelf photo plus optional
tuning parameters and returns a JSON report together with a base64-encoded
annotated image, ready for the frontend to render.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.annotation import draw_annotations
from app.config import get_settings
from app.detection import (
    RETAIL_CLASS_ALLOWLIST,
    deduplicate_same_class_boxes,
    detect_tiled,
    filter_by_class,
    get_detector,
    suppress_contained_boxes,
    suppress_overlapping_boxes,
)
from app.schemas import AnalysisConfig, AnalysisReport
from app.shelf_analysis import describe_shelf_count_mismatch, group_into_shelf_regions, summarize_compliance

settings = get_settings()

app = FastAPI(
    title="Retail Shelf Analytics API",
    description="Detects products on a shelf photo, groups them into shelf regions, and flags stock gaps.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisReport)
async def analyze_shelf(
    file: UploadFile = File(..., description="Shelf photo (JPEG or PNG)."),
    confidence: float = Query(default=settings.default_confidence, ge=0.0, le=1.0),
    iou: float = Query(default=settings.default_iou, ge=0.0, le=1.0),
    expected_shelf_count: int | None = Query(
        default=None,
        ge=1,
        le=20,
        description="Optional hint for how many shelves the photo shows. Rows are always found "
        "from the detections themselves; if the count found does not match this hint, "
        "the mismatch is reported in `shelf_count_note` instead of being silently resolved.",
    ),
    debug: bool = Query(
        default=False,
        description="Render the annotated image with per-box confidence labels and a full-width "
        "status banner per shelf, for troubleshooting. Off by default for a readable result image.",
    ),
) -> AnalysisReport:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, or WEBP images are supported.")

    raw_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_size_mb} MB limit.")

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    height, width = image_bgr.shape[:2]
    start_time = time.perf_counter()

    detector = get_detector()
    if settings.tiled_inference:
        raw_detections = detect_tiled(detector, image_bgr, confidence_threshold=confidence, iou_threshold=iou)
    else:
        raw_detections = detector.detect(image_bgr, confidence_threshold=confidence, iou_threshold=iou)
    detections = filter_by_class(raw_detections, RETAIL_CLASS_ALLOWLIST)
    detections = suppress_overlapping_boxes(detections, settings.cross_class_iou_threshold)
    detections = suppress_contained_boxes(detections, settings.containment_suppression_threshold)
    if settings.same_class_dedup_iou is not None:
        detections = deduplicate_same_class_boxes(detections, settings.same_class_dedup_iou)

    regions = group_into_shelf_regions(
        detections=detections,
        dbscan_eps_factor=settings.dbscan_eps_factor,
        gap_width_factor=settings.gap_width_factor,
        understocked_occupancy_threshold=settings.understocked_occupancy_threshold,
        min_detections_for_confidence=settings.min_detections_for_confidence,
    )
    compliance = summarize_compliance(regions)
    shelf_count_note = describe_shelf_count_mismatch(regions, expected_shelf_count)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_png = draw_annotations(image_rgb, regions, debug=debug, compliance=compliance)
    annotated_base64 = base64.b64encode(annotated_png).decode("ascii")

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return AnalysisReport(
        image_width=width,
        image_height=height,
        config=AnalysisConfig(
            confidence_threshold=confidence,
            iou_threshold=iou,
            dbscan_eps_factor=settings.dbscan_eps_factor,
            gap_width_factor=settings.gap_width_factor,
            understocked_occupancy_threshold=settings.understocked_occupancy_threshold,
            min_detections_for_confidence=settings.min_detections_for_confidence,
        ),
        shelf_regions=regions,
        compliance=compliance,
        shelf_count_note=shelf_count_note,
        processing_time_ms=round(processing_time_ms, 1),
        annotated_image_base64=annotated_base64,
    )
