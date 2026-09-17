# Evaluation

This is a small, manually-reviewed spot check against the two bundled sample photos, not a statistical benchmark. There is no labeled ground-truth dataset in this project, and this session has no GPU or the multi-gigabyte SKU-110K dataset needed to run a real precision/recall/mAP evaluation against a fine-tuned model. Anything not measured here is stated as not measured, rather than estimated or invented.

## Methodology

For each sample photo, facings were counted by direct visual inspection, using this definition: **a facing is any product unit whose body, cap, or label is visible and can be distinguished from its neighbors as a separate object**, including partially occluded units in a second row where the bottle shape and color are still separable. This is more inclusive than a strict "front row only" retail-audit definition, and was chosen because it is closer to what an object detector can plausibly resolve from a single 2D photo; a fully occluded bottle with no visible surface at all cannot be recovered by any detector working from one image, and is excluded from both the manual count and any reasonable detection target.

Counting was done once, by inspection, without pixel-level measurement tools. Dense clusters have real ambiguity (see the honest range given for `shelf_soda_bottles.jpg` shelf 1). This is disclosed rather than hidden.

## `shelf_soda_bottles.jpg` (confidence 0.15, IoU 0.45)

| Shelf | Manual facing estimate | Detected facings | Note |
|---|---|---|---|
| 1 (fritz-limo / fritz-spritz cluster + apple-schorle bottles) | approx. 11 (a dense multi-pack case makes the exact rear-row count genuinely uncertain) | 10 | One likely miss, plausibly a partially occluded rear bottle in the case |
| 2 (Coca-Cola / Mezzo Mix / Fanta / Lift) | approx. 12-13 | 12 | Consistent within the manual estimate's own uncertainty |

Both shelves have one real, visually obvious gap in the photo (bare shelf metal visible: between the dark cluster and the apple-schorle bottles on shelf 1, and between the Mezzo Mix and Fanta clusters on shelf 2). The pipeline flagged exactly one gap on each shelf, matching both. No gap was flagged where none is visible, on either shelf.

This single image is not enough to report a precision/recall percentage that would mean anything statistically. What it does show directly: the detected facing counts are within 1 unit of a careful manual count on both shelves, and both real gaps were found with no false ones, at this specific confidence and IoU setting.

## `shelf_sauce_aisle.jpg` (confidence 0.12, IoU 0.45, `expected_shelf_count=6`)

This photo is a harder case by construction: a steep camera angle and roughly 51 kept detections across small, tightly packed jars and bottles, several of which are outside the stock `yolov8n` model's only closely relevant COCO class (`bottle`), as documented in `docs/AUDIT.md`. Manually counting every individual facing in this photo to a trustworthy precision was not attempted, since the occlusion and density make a reliable-by-eye count impractical; claiming a specific number here would be exactly the kind of fabricated precision this evaluation is meant to avoid.

What was checked directly: the photo has 6 physical shelf rows, visible by eye. The pipeline's row clustering resolved 5 distinct product rows and reported the count mismatch explicitly (`shelf_count_note`) instead of guessing at the sixth. Manually tracing the detections back onto the photo shows the two lowest, most distant, most tightly packed rows in this steep-angle shot were merged into a single detected cluster; every other row lines up with a real physical shelf. No detected region was mislabeled `EMPTY` on a shelf that visibly holds products, which is the specific failure this pass was meant to fix (see `docs/AUDIT.md` for the original bug and its reproduction).

## What this evaluation does not show

- No mAP, precision, or recall number against a labeled dataset. That requires SKU-110K or a comparable labeled retail set and a training/evaluation run this environment cannot perform; see `docs/PRIOR_ART.md` for real numbers reported by comparable fine-tuned projects (for context, not as a claim about this project's own detector).
- No evaluation beyond these two photos. Two images, both photographed in daylight-equivalent indoor lighting, at two camera angles, are not representative of the range of real shelf photography (low light, motion blur, different store fixtures, different product categories).
- No SKU or brand-level accuracy, since the detector has no SKU recognition stage (see Limitations in the README).

## How to reproduce

```bash
cd backend
source .venv/Scripts/activate
python sample_data/generate_samples.py
```

This regenerates both annotated images and their JSON reports in `sample_data/output/`, which is where the facing counts and gap counts cited above came from.
