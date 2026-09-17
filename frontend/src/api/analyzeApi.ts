import type { AnalysisReport, AnalysisRequestParams } from "../types/analysis";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class AnalysisApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "AnalysisApiError";
  }
}

/**
 * Uploads a shelf photo to the backend and returns the full analysis
 * report, including the annotated image and per-shelf breakdown.
 */
export async function analyzeShelfImage(
  file: File,
  { confidence, iou, expectedShelfCount, debug }: AnalysisRequestParams,
): Promise<AnalysisReport> {
  const formData = new FormData();
  formData.append("file", file);

  const searchParams = new URLSearchParams({
    confidence: confidence.toString(),
    iou: iou.toString(),
    debug: (debug ?? false).toString(),
  });
  if (expectedShelfCount) {
    searchParams.set("expected_shelf_count", expectedShelfCount.toString());
  }

  const response = await fetch(`${API_BASE_URL}/api/analyze?${searchParams.toString()}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new AnalysisApiError(
      detail?.detail ?? `Analysis request failed with status ${response.status}.`,
      response.status,
    );
  }

  return response.json() as Promise<AnalysisReport>;
}
