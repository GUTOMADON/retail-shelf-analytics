"""One-off script to generate the annotated sample images used in the README.

Not part of the shipped application. Run manually with:
    python sample_data/generate_samples.py
"""

import io
import json
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.annotation import draw_annotations
from app.config import get_settings
from app.detection import (
    RETAIL_CLASS_ALLOWLIST,
    ShelfDetector,
    filter_by_class,
    suppress_contained_boxes,
    suppress_overlapping_boxes,
)
from app.shelf_analysis import describe_shelf_count_mismatch, group_into_shelf_regions, summarize_compliance

SAMPLES = [
    {"file": "shelf_soda_bottles.jpg", "confidence": 0.15, "expected_shelf_count": 2},
    {"file": "shelf_sauce_aisle.jpg", "confidence": 0.12, "expected_shelf_count": 6},
]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

settings = get_settings()
detector = ShelfDetector(model_path="yolov8n.pt", device="cpu")


def run_sample(file_name: str, confidence: float, expected_shelf_count: int, debug: bool, output_stem: str) -> None:
    image_path = Path(__file__).parent / file_name
    image_bgr = cv2.imread(str(image_path))
    height, width = image_bgr.shape[:2]

    start = time.perf_counter()
    raw_detections = detector.detect(image_bgr, confidence_threshold=confidence, iou_threshold=0.45)
    detections = filter_by_class(raw_detections, RETAIL_CLASS_ALLOWLIST)
    detections = suppress_overlapping_boxes(detections, settings.cross_class_iou_threshold)
    detections = suppress_contained_boxes(detections, settings.containment_suppression_threshold)

    regions = group_into_shelf_regions(
        detections=detections,
        dbscan_eps_factor=settings.dbscan_eps_factor,
        gap_width_factor=settings.gap_width_factor,
        understocked_occupancy_threshold=settings.understocked_occupancy_threshold,
        min_detections_for_confidence=settings.min_detections_for_confidence,
    )
    compliance = summarize_compliance(regions)
    note = describe_shelf_count_mismatch(regions, expected_shelf_count)
    processing_time_ms = (time.perf_counter() - start) * 1000

    print(f"{file_name} (debug={debug}): {len(raw_detections)} raw -> {len(detections)} kept detections")
    print(f"  regions: {[(r.facing_count, r.status.value) for r in regions]}")
    print(f"  compliance: {compliance.model_dump()}")
    if note:
        print(f"  note: {note}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_png = draw_annotations(image_rgb, regions, debug=debug, compliance=compliance)
    annotated_image = Image.open(io.BytesIO(annotated_png)).convert("RGB")
    annotated_image.save(OUTPUT_DIR / f"{output_stem}.jpg", "JPEG", quality=90, optimize=True)

    report = {
        "image_width": width,
        "image_height": height,
        "processing_time_ms": round(processing_time_ms, 1),
        "shelf_regions": [r.model_dump() for r in regions],
        "compliance": compliance.model_dump(),
        "shelf_count_note": note,
    }
    (OUTPUT_DIR / f"{output_stem}_report.json").write_text(json.dumps(report, indent=2))


for sample in SAMPLES:
    stem = Path(sample["file"]).stem
    run_sample(sample["file"], sample["confidence"], sample["expected_shelf_count"], debug=False, output_stem=f"{stem}_clean")
    run_sample(sample["file"], sample["confidence"], sample["expected_shelf_count"], debug=True, output_stem=f"{stem}_debug")

print("Done.")
