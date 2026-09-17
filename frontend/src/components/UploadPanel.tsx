import { useRef, useState } from "react";
import type { AnalysisRequestParams } from "../types/analysis";

interface UploadPanelProps {
  isLoading: boolean;
  onAnalyze: (file: File, params: AnalysisRequestParams) => void;
}

/**
 * Handles file selection plus the tunable inference parameters
 * (confidence / IoU thresholds and optional fixed shelf count) that are
 * forwarded to the backend on every analysis request.
 */
export function UploadPanel({ isLoading, onAnalyze }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.35);
  const [iou, setIou] = useState(0.45);
  const [expectedShelfCount, setExpectedShelfCount] = useState<string>("");
  const [debug, setDebug] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedFile) return;
    onAnalyze(selectedFile, {
      confidence,
      iou,
      expectedShelfCount: expectedShelfCount ? Number(expectedShelfCount) : undefined,
      debug,
    });
  }

  return (
    <form className="upload-panel" onSubmit={handleSubmit}>
      <div
        className={`dropzone ${isDragging ? "dropzone--active" : ""}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected shelf preview" className="dropzone__preview" />
        ) : (
          <div className="dropzone__placeholder">
            <p>Drag a shelf photo here, or click to browse</p>
            <span>JPEG or PNG, up to 15&nbsp;MB</span>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      <div className="upload-panel__controls">
        <label className="control">
          <span>
            Confidence threshold <strong>{confidence.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0.05}
            max={0.9}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
          />
        </label>

        <label className="control">
          <span>
            IoU threshold <strong>{iou.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={iou}
            onChange={(e) => setIou(Number(e.target.value))}
          />
        </label>

        <label className="control">
          <span>Expected shelf count (optional)</span>
          <input
            type="number"
            min={1}
            max={20}
            placeholder="not provided"
            value={expectedShelfCount}
            onChange={(e) => setExpectedShelfCount(e.target.value)}
          />
        </label>
      </div>

      <label className="checkbox-control">
        <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
        <span>
          Debug view <small>(per-box confidence labels and full-width shelf banners, for troubleshooting)</small>
        </span>
      </label>

      <button type="submit" className="primary-button" disabled={!selectedFile || isLoading}>
        {isLoading ? "Analyzing…" : "Analyze shelf"}
      </button>
    </form>
  );
}
