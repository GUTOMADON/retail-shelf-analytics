/**
 * Mirrors the Pydantic models in `backend/app/schemas.py`. Keeping these in
 * lockstep with the backend response shape is what lets the rest of the
 * frontend consume the API result without any runtime validation layer.
 */

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  bbox: BoundingBox;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface StockGap {
  x_start: number;
  x_end: number;
  width_px: number;
  estimated_missing_facings: number;
}

export type RegionStatus = "ok" | "understocked" | "empty";

export interface ShelfRegion {
  region_id: number;
  y_start: number;
  y_end: number;
  facing_count: number;
  detections: Detection[];
  gaps: StockGap[];
  occupancy_ratio: number;
  status: RegionStatus;
}

export interface ComplianceSummary {
  total_regions: number;
  ok_regions: number;
  understocked_regions: number;
  empty_regions: number;
  overall_occupancy_ratio: number;
  total_facings: number;
  total_estimated_missing_facings: number;
}

export interface AnalysisConfig {
  confidence_threshold: number;
  iou_threshold: number;
  row_gap_factor: number;
  gap_width_factor: number;
  understocked_occupancy_threshold: number;
}

export interface AnalysisReport {
  image_width: number;
  image_height: number;
  config: AnalysisConfig;
  shelf_regions: ShelfRegion[];
  compliance: ComplianceSummary;
  annotated_image_base64: string;
}

export interface AnalysisRequestParams {
  confidence: number;
  iou: number;
  expectedShelfCount?: number;
}
