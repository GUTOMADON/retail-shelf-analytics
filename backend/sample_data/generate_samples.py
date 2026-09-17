"""One-off script to generate the annotated sample images used in the README.

Not part of the shipped application — run manually with:
    python sample_data/generate_samples.py
"""

import io
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.annotation import draw_annotations
from app.detection import ShelfDetector
from app.shelf_analysis import group_into_shelf_regions, summarize_compliance

SAMPLES = [
    {"file": "shelf_soda_bottles.jpg", "confidence": 0.15, "expected_shelf_count": 2},
    {"file": "shelf_sauce_aisle.jpg", "confidence": 0.12, "expected_shelf_count": 6},
]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

detector = ShelfDetector(model_path="yolov8n.pt", device="cpu")

for sample in SAMPLES:
    image_path = Path(__file__).parent / sample["file"]
    image_bgr = cv2.imread(str(image_path))
    height, width = image_bgr.shape[:2]

    detections = detector.detect(image_bgr, confidence_threshold=sample["confidence"], iou_threshold=0.45)
    print(f"{sample['file']}: {len(detections)} detections")
    for d in detections:
        print(f"  - {d.class_name} ({d.confidence:.2f})")

    regions = group_into_shelf_regions(
        detections=detections,
        image_width=width,
        image_height=height,
        row_gap_factor=1.6,
        gap_width_factor=1.35,
        understocked_occupancy_threshold=0.75,
        expected_shelf_count=sample["expected_shelf_count"],
    )
    compliance = summarize_compliance(regions)
    print(f"  compliance: {compliance.model_dump()}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotated_png = draw_annotations(image_rgb, regions)

    # Re-encoded as JPEG (from the freshly rendered PNG bytes) to keep sample
    # output small enough to check into the repo alongside the README.
    stem = Path(sample["file"]).stem
    annotated_image = Image.open(io.BytesIO(annotated_png)).convert("RGB")
    annotated_image.save(OUTPUT_DIR / f"{stem}_annotated.jpg", "JPEG", quality=88, optimize=True)
    report = {
        "image_width": width,
        "image_height": height,
        "shelf_regions": [r.model_dump() for r in regions],
        "compliance": compliance.model_dump(),
    }
    (OUTPUT_DIR / f"{stem}_report.json").write_text(json.dumps(report, indent=2))

print("Done.")
