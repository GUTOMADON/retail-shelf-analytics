# Prior art review

This document summarizes public repositories relevant to retail shelf analytics, each verified to exist via the GitHub API before being cited here, plus the datasets and techniques they rely on. The goal is to establish what a mature pipeline actually looks like before changing this project's architecture.

## Repositories reviewed

### Alijanloo/Retail-Shelf-Monitoring (14 stars)

The most complete reference found. A real-time out-of-stock and planogram compliance system built around:

- **Detection**: YOLOv12-M fine-tuned on SKU-110K, reporting mAP50-95 = 0.56.
- **SKU recognition**: MobileNetV3 embeddings matched via FAISS similarity search, 93% top-1 accuracy on their custom embedding set.
- **Row/grid grouping**: automated DBSCAN-based clustering of shelf items, not fixed geometric bands.
- **Temporal stability**: SORT + Kalman filter tracking across video frames, with a temporal consensus step specifically to reduce false positives before an alert fires.
- **Pipeline**: `CCTV Stream -> Keyframe Selection -> Shelf Detection -> Image Alignment -> YOLOv12 Detection -> SKU Recognition -> Grid Mapping -> Temporal Consensus -> Alert Generation -> Desktop UI -> Staff Confirmation`.

Takeaway: density-based clustering (DBSCAN) is the established technique for turning detections into shelf rows, not equal-height geometric bands. Temporal consensus across frames is what production systems use to suppress false positives; a single still image has no equivalent signal available.

### Adnanwadee/retail-shelf-dense-product-detection (0 stars, portfolio writeup)

Targets shelf images with 170 to 230 tightly packed product instances. Benchmarks YOLOv8m against RT-DETR-L on a single-class dense-detection dataset, then uses DINOv2 embeddings plus Mini-Batch K-Means to turn one-class detections into 91 unsupervised visual product groups, exposed through a Streamlit dashboard. The README is explicit that this is "not a full commercial billing system, point-of-sale system, or guaranteed SKU-recognition engine" and separates measured results (mAP, precision, recall, F1, latency, model size) into per-model model cards.

Takeaway: even a well-executed dense-detection project stops short of claiming full SKU identification, and keeps every reported number traceable to a specific evaluation run. That discipline is worth copying regardless of which detector this project ends up using.

### HenryyyLI/CNN-shelf-product-detection-project (0 stars)

Two-stage pipeline: YOLOv5 fine-tuned on SKU-110K for localization, ResNet-18 fine-tuned on the Grocery Store Dataset for per-product classification. Reported, real numbers: detection mAP@0.5 = 88.9%, precision/recall = 89.6% / 81.8%; classification accuracy 78.31% across 81 categories.

Takeaway: this is the closest real benchmark for "what does a SKU-110K-fine-tuned YOLO actually score." It confirms that fine-tuning is what closes the gap between a COCO-generic detector and usable retail recall, and gives an honest recall ceiling (81.8%) even after fine-tuning, so a fine-tuned model is a large improvement, not a silver bullet.

### albertferre/shelf-product-identifier (71 stars)

A YOLOv8m detector fine-tuned on SKU-110K for facing localization, then image embeddings (compared with cosine similarity, visualized with t-SNE) to cluster facing crops into product groups, which are manually labeled once to build a small knowledge base. New crops are then matched to the closest known cluster. The author's own reported result: 2 of 3 products in a held-out test photo were classified correctly, with the one miss attributed to a training-set logo mismatch, stated plainly rather than smoothed over.

The trained YOLOv8m checkpoint and training notebook are published on Kaggle (`albertferre/sku-facings-detector`), and the GitHub repository itself is MIT licensed. This checkpoint was **not** integrated into this project: downloading a Kaggle notebook's output artifact requires Kaggle account credentials this environment does not have, and the weights are not otherwise mirrored as a plain downloadable file. It is recorded here as a concrete, real candidate for a future upgrade (see the Roadmap in the main README) rather than something already verified end to end.

Takeaway: embeddings-based facing counting (as opposed to per-box class labels) is a real, working way to get product-level identity out of a class-agnostic dense detector, and this project independently confirms fine-tuning on SKU-110K is what other teams reach for first, before ever getting to the embeddings step.

### TalhaFarook/ShelfSight (15 stars)

Trains YOLOv8 to detect empty shelf space directly as its own object class, rather than inferring gaps geometrically from the spacing between product boxes. The README is a single short paragraph with no unverified claims.

Takeaway: a dedicated "empty space" detector is a legitimate alternative architecture to geometric gap-inference, but it requires a labeled dataset of empty-shelf regions, which this project does not have and cannot fabricate.

### akul-bharadwaj/Computer-Vision-Assessment-Planogram-Dataset (0 stars)

A Gradio app plus a YOLOv8 model fine-tuned specifically for shelf gap detection, trained on the "Shelf Images for Planograms" Kaggle dataset (2,095 images), published to Hugging Face at `akul-29/Retail-Shelf-Gap-Detection_Model`. Verified directly against the Hugging Face API: the model card declares no license, and the underlying Kaggle dataset's license field is `"Unknown"`.

Takeaway: this is a real, working example of the "train gaps as a first-class object class" approach and confirms it is achievable at small dataset scale (2K images). It was **not** integrated into this project because of the unresolved licensing status of both the model and its training data. It is cited here as a technical reference only.

### bilychen88/RetailDet (7 stars, IEEE Access paper, code not yet released)

An academic proposal, "RetailDet: An Efficient Fusion Attention Network for Joint Product and Vacancy Identification in Smart Retail" (IEEE Access, 2025). No code or weights are available; the repository currently only hosts figures and the citation. Cited here only to note that joint product-and-vacancy detection in a single network is an active research direction, not an available tool.

### Sudiptermux/Inventory-Management-System (1 star)

A YOLOv8 + Streamlit inventory dashboard with an unusually well-written literature review. Its related-work section cites third-party results worth carrying forward with attribution: Iyer (2023) reports 96% accuracy using YOLOv7 for joint product/empty-space detection; a modified YOLOv3 reached 96.82% mAP for cup counting; YOLOv8 combined with RT-DETR reached 99.5% mAP on an unspecified grocery dataset. These are secondary citations from that repository's own literature review, not numbers this project has verified independently, so they are repeated here only as pointers to the wider literature, not as claims about this project's own performance.

### intel-iot-devkit/smart-retail-analytics (81 stars, 2020, archived-style Intel reference app)

Uses the Intel Distribution of OpenVINO with a generic `mobilenet-ssd` (Caffe, COCO-style classes) for the "shelf" video feed type, simply counting detections of a user-specified label (for example "bottle") per frame, streamed into InfluxDB and visualized with Grafana. No shelf-row logic, no gap detection, no fine-tuning.

Takeaway: even an official vendor reference architecture for shelf inventory used a generic pretrained detector counting one COCO class, which is close to this project's starting point. It confirms a generic detector is a reasonable *baseline*, but the field has moved well past it since 2020, toward the fine-tuned, dense-detection approaches above.

### akalausichcodes/GroceryShelfVision (0 stars)

Classical computer vision (OpenCV edge detection at multiple thresholds, K-Means for dominant color, Tesseract OCR for price text), no deep learning detector at all. Relevant as a reminder that non-DL heuristics remain viable for narrow tasks like package counting on a single product category, but do not generalize to open-vocabulary shelf detection.

### aviral-guptaa/retail-intelligence-platform (0 stars, in-progress spec-driven project)

Not shelf-specific (footfall, queueing, dwell time), but its engineering discipline is directly relevant: it ships a `demo` mode that runs with no camera, no GPU, and no model download, and is explicit in the README that "without a YOLO checkpoint, detection falls back to tracking raw background-motion blobs." This is the pattern this project should copy for honesty about what runs out of the box versus what requires an optional model upgrade.

## Datasets confirmed to exist

| Dataset | Verified via | Notes |
|---|---|---|
| SKU-110K | `github.com/eg4000/SKU110K_CVPR19` (cited by HenryyyLI and referenced by Alijanloo) | ~11,762 densely packed retail shelf images, the de facto benchmark for dense product detection. Large download (multi-GB), CVPR 2019 paper. |
| Grocery Store Dataset | `github.com/marcusklasson/GroceryStoreDataset` (cited by HenryyyLI) | 5,125 images, 81 fine-grained produce/grocery classes, used for classification, not detection. |
| Shelf Images for Planograms | Kaggle, `aamiraliansari/shelf-images-for-planograms` (cited by akul-bharadwaj) | 2,095 images. License field on Kaggle is `"Unknown"` as of this review. |

Retail Product Checkout (RPC) and Products-10K were named in the research brief but were not independently re-verified in this pass; they are well-known public retail datasets referenced across the broader literature and are noted here as candidates for future work rather than as something already confirmed against a live source in this review.

## What a mature pipeline looks like, versus this repository

| Capability | Mature reference implementation | This repository (before this review) |
|---|---|---|
| Detector | Fine-tuned on a dense retail dataset (SKU-110K), reporting measured mAP/precision/recall | Stock `yolov8n`, COCO-pretrained, no retail fine-tuning |
| Row/shelf grouping | Density-based clustering (DBSCAN) or explicit shelf-rail detection | Either a fixed count of equal-height geometric bands, or a single global vertical-gap threshold |
| Gap detection | Either a dedicated trained "empty space" class, or geometry weighted by local product density and shelf boundaries | A single global width threshold compared against the row's own average product width, rendered as a full-height tinted rectangle |
| False-positive control | Temporal consensus across frames (video), FAISS/embedding re-identification | None; every detection above the confidence threshold is trusted as-is, across every COCO class returned |
| Status vocabulary | OK / understocked / empty, sometimes with a confidence-qualified state | OK / understocked / empty, with no way to express "not enough evidence to classify" |
| Visual output | Clean, non-overlapping annotations positioned for readability | Every detection labeled individually, shelf-status banners stamped across the image regardless of what they cover |

This review directly motivates the fixes proposed in `docs/AUDIT.md`.
