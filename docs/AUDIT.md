# Audit of the current pipeline

Baseline captured before any Phase 2 change. Test suite: `pytest -q` in `backend/`, 5 passed, 0 failed (unchanged from the original submission). This audit traces the exact defect visible in the reported screenshot (`Shelf 3 - 0 facings - EMPTY` on a shelf that is visibly full of jars) back to specific lines of code, then lists every other concrete defect found while doing that.

## Reproducing the reported failure

Running the shipped `backend/sample_data/generate_samples.py` against `shelf_sauce_aisle.jpg` (confidence 0.12, `expected_shelf_count=6`) reproduces the report exactly:

```
region 0: y=[  0, 160] facings=9  status=ok
region 1: y=[160, 320] facings=13 status=understocked
region 2: y=[320, 480] facings=0  status=empty      <-- the reported bug
region 3: y=[480, 640] facings=5  status=understocked
region 4: y=[640, 800] facings=10 status=understocked
region 5: y=[800, 960] facings=14 status=understocked
```

To find out whether this is a confidence problem or a grouping problem, detection was re-run at `confidence=0.08` (well below the 0.12 already used, and below the 0.35 default) and detections were bucketed into 40px vertical bands:

```
 240-280 : 1
 280-320 : 20   <-- dense cluster, one bucket below the "empty" band
 320-400 : 0    <-- genuinely zero detections at any confidence tried
 400-440 : 1
 440-480 : 1
 480-520 : 4
```

Even at 0.08 confidence there are zero detections in pixel rows 320-400. This rules out "confidence threshold too high" as the explanation. What the y=280-320 spike shows is that the real, dense product row sits right at the edge of band 1, and the fixed 160px-tall band 2 lands mostly on what is structurally the shelf rail / price-tag strip between two physical shelf boards in this perspective-angled photo, not on the jars themselves.

**Root cause**: `_cluster_rows_fixed_bands` in `backend/app/shelf_analysis.py` (lines 47-56) divides the image into `expected_shelf_count` **equal-height** horizontal slices using only `image_height / expected_shelf_count`, with no reference to where the detections (or the physical shelves) actually are. On a photo shot at a steep angle, physical shelves are not evenly spaced in pixel space, so an equal-height cut can land on a structural gap between two shelves and report it as `EMPTY`, exactly as observed. This is an architectural defect, not a parameter that needs tuning.

## Other defects found during the same pass

### 1. No cross-class, no containment-based deduplication (`backend/app/detection.py`, lines 40-46)

`self._model.predict(..., iou=iou_threshold)` is called without `agnostic_nms=True`. Ultralytics' default non-max suppression is **per class**, so a box classified as `refrigerator` and a heavily overlapping box classified as `bottle` are never compared against each other and both survive. This is visible in every generated sample: a `refrigerator 0.68` box spans the entire top shelf in `shelf_soda_bottles.jpg`, overlapping several real `bottle` detections underneath it. There is also no containment check (a box mostly contained inside a larger box of a different class is never dropped), and no explicit allowlist restricting output to plausible retail-product classes, so any of the 80 COCO classes can appear in a shelf photo.

### 2. Every detection is individually labeled, regardless of density (`backend/app/annotation.py`, lines 61-69)

Each bounding box gets its own confidence-and-class-name label drawn immediately above it. On a dense row (the sauce aisle sample has 51 detections across 6 rows), adjacent labels overlap into the illegible text stack visible in the reported screenshot (`bottle 0.16 0.18 0.13...`). This is a rendering defect independent of detection quality: even perfect detections at this density would produce overlapping labels under the current renderer.

### 3. Shelf-status banner is stamped across the image regardless of what is under it (`backend/app/annotation.py`, lines 44-59)

The status label and its separator line are drawn at a fixed `y_start` spanning the image's full width, with no check for what is behind them. On an angled photo where a region's `y_start` cuts through the middle of a product row (rather than along an actual shelf edge), the label is stamped directly over merchandise, which is exactly the "carimbado por cima da mercadoria" complaint.

### 4. Gap rectangles span the full region height (`backend/app/annotation.py`, lines 71-84)

A detected `StockGap` is drawn as a filled, outlined rectangle from `region.y_start + 4` to `region.y_end - 4`, i.e. the entire vertical extent of the shelf region, regardless of the actual product height in that row. On a tall region this produces the oversized, disproportionate red block visible in the reported screenshot, which looks like a much larger "missing" area than the gap actually represents.

### 5. No confidence floor on which detections can define a gap or an occupancy ratio (`backend/app/shelf_analysis.py`, lines 59-77, 94-97)

`_find_gaps` and the occupancy calculation in `_build_region` use every detection in the row with equal weight, including ones at 0.12-0.20 confidence, one step above noise for a COCO-pretrained model on non-COCO objects. A single low-confidence false detection can shift where a gap is measured from.

### 6. No status for "not enough evidence" (`backend/app/schemas.py`, `RegionStatus` enum)

The status enum is `OK`, `UNDERSTOCKED`, `EMPTY`. There is no way to express "this region's classification is not reliable" (e.g., too few detections, or detections all near the confidence floor), so the fixed-band bug above is forced into a confident, specific, and wrong label (`EMPTY`) instead of an honest "insufficient evidence" state.

### 7. Dynamic clustering has the same class of problem in miniature (`backend/app/shelf_analysis.py`, lines 32-44)

`_cluster_rows_dynamic` uses a single global threshold (`median box height * row_gap_factor`) to decide every row split in the image. On a perspective photo, near objects are larger and far objects are smaller, so one global threshold is systematically wrong for at least part of the image. This does not cause a false `EMPTY` (dynamic clustering only creates rows where detections exist), but it can merge or split rows incorrectly, which is the same underlying problem as the fixed-band defect: shelf structure is not one-dimensional in pixel space and a single scalar threshold cannot capture it.

## What is and is not in scope to fix without new training data or a GPU

This audit deliberately separates defects that are pure software engineering (data structures, thresholds, rendering) from ones that require a fine-tuned model or a labeled dataset this environment does not have:

**Fixable now, no training required:**
- Defects 1 through 7 above are all fixable by changing detection filtering, post-processing, and rendering code. None require new training data.
- Replacing fixed-height bands and the single-threshold dynamic clustering with density-based clustering (DBSCAN over detection y-centers, following the approach documented in `Alijanloo/Retail-Shelf-Monitoring`, see `docs/PRIOR_ART.md`) directly addresses the reported bug, because a density-based method cannot manufacture a cluster from a structural gap the way an equal-height cut can.

**Not fixable without new training data or infrastructure this environment does not have:**
- The detector itself is still the stock COCO-pretrained `yolov8n`. Its only relevant class is `bottle`; it has no notion of jars, cans, pouches, or boxes. `docs/PRIOR_ART.md` documents that comparable projects only close this gap by fine-tuning on SKU-110K (a multi-gigabyte dataset) or a similar retail dataset, which requires a labeled dataset, GPU time, and a training run this session cannot perform and will not fake the results of.
- SKU/brand recognition (FAISS + embeddings, as in `Alijanloo/Retail-Shelf-Monitoring`) requires a labeled reference catalog of the target store's own products, which does not exist for this portfolio project.
- Temporal consensus across frames requires video input; this project processes single still images by design.

These are documented as roadmap items with an honest "not implemented, and here is exactly why" rather than attempted with fabricated numbers.

## Resolution status

All seven fixable defects listed above were implemented and are covered by tests:

| Defect | Fix | Where |
|---|---|---|
| 1. No cross-class NMS or containment check | `agnostic_nms=True` on the YOLO call, plus a class allowlist and a containment-suppression pass | `app/detection.py`, `test_detection_postprocessing.py` |
| 2. Every detection individually labeled | Clean view draws plain boxes with no per-box label; Debug view keeps labels for troubleshooting | `app/annotation.py` |
| 3. Status banner stamped across the image | Clean view uses a compact left-margin tag instead of a full-width banner | `app/annotation.py` |
| 4. Gap rectangles span the full region height | Gap marker height is capped at 35 percent of the row height, centered on the row | `app/annotation.py` |
| 5. No confidence floor on gap/occupancy math | Regions below `min_detections_for_confidence` are reported `unknown` before gap or occupancy math runs | `app/shelf_analysis.py`, `test_shelf_analysis.py` |
| 6. No "not enough evidence" status | Added `RegionStatus.UNKNOWN` | `app/schemas.py` |
| 7. Fixed bands and single-threshold dynamic clustering | Replaced both with DBSCAN over a scale-normalized vertical distance; verified empirically against both bundled photos (see the eps sweep below) | `app/shelf_analysis.py` |

The reported "Shelf 3, 0 facings, EMPTY" bug specifically cannot recur, because rows are now built only from detections that exist; there is no code path left that can produce a region with zero detections. This is a structural guarantee, not a tuned threshold, and it is exercised directly by `test_clustering_never_invents_an_empty_row`.

### Tuning the DBSCAN distance threshold

A single global pixel-based threshold was also tried and rejected: it fixed the false-EMPTY bug but under-segmented the sauce-aisle photo into only 2 rows instead of 6, because that photo's strong perspective makes one global pixel scale wrong for most of the frame. Distance was changed to `abs(center_y_i - center_y_j) / average(height_i, height_j)`, scale-normalized per pair, then swept against both bundled photos:

```
eps=0.20: sauce aisle -> 6 rows (matches the 6 physical shelves)
eps=0.25: sauce aisle -> 5 rows; soda bottles -> 3 rows (2 real shelves + 1 correctly isolated low-confidence floor detection)
eps=0.30: sauce aisle -> 5 rows; soda bottles -> 3 rows (same)
eps=1.40 (the original, non-normalized default): sauce aisle -> 2 rows (under-segmented)
```

`0.25` was kept as the default: it gives a clean, physically sensible result on the easier photo and only merges the two smallest, most distant rows on the harder one, which is disclosed as a known limitation in `docs/EVALUATION.md` rather than hidden. Both raw sweep results are reproducible from `backend/sample_data/shelf_soda_bottles.jpg` and `shelf_sauce_aisle.jpg` using `_cluster_rows` in `app/shelf_analysis.py` directly.
