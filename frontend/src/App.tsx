import { useState } from "react";
import { analyzeShelfImage, AnalysisApiError } from "./api/analyzeApi";
import { AnnotatedImageView } from "./components/AnnotatedImageView";
import { ComplianceSummaryPanel } from "./components/ComplianceSummaryPanel";
import { ShelfSummaryTable } from "./components/ShelfSummaryTable";
import { UploadPanel } from "./components/UploadPanel";
import type { AnalysisReport, AnalysisRequestParams } from "./types/analysis";

export default function App() {
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(file: File, params: AnalysisRequestParams) {
    setIsLoading(true);
    setError(null);
    try {
      const result = await analyzeShelfImage(file, params);
      setReport(result);
    } catch (err) {
      const message = err instanceof AnalysisApiError ? err.message : "Unexpected error while analyzing the image.";
      setError(message);
      setReport(null);
    } finally {
      setIsLoading(false);
    }
  }

  function handleDownloadReport() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "shelf-analysis-report.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>Retail Shelf Analytics</h1>
          <p>Upload a shelf photo to detect product facings, measure occupancy, and flag stock gaps.</p>
        </div>
      </header>

      <main className="app__main">
        <section className="app__panel">
          <UploadPanel isLoading={isLoading} onAnalyze={handleAnalyze} />
          {error && <p className="error-banner">{error}</p>}
        </section>

        {report && (
          <section className="app__results">
            <div className="app__results-image">
              <AnnotatedImageView
                base64Png={report.annotated_image_base64}
                width={report.image_width}
                height={report.image_height}
              />
            </div>

            <div className="app__results-data">
              {report.shelf_count_note && <p className="note-banner">{report.shelf_count_note}</p>}
              <ComplianceSummaryPanel compliance={report.compliance} processingTimeMs={report.processing_time_ms} />
              <ShelfSummaryTable regions={report.shelf_regions} />
              <button className="secondary-button" onClick={handleDownloadReport}>
                Download JSON report
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
