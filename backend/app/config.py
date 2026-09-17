"""Application configuration.

Settings are sourced from environment variables (or a local .env file) so the
detection model, thresholds, and server behavior can be tuned per deployment
without touching code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Retail Shelf Analytics backend."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SHELF_", extra="ignore")

    # Detection model
    yolo_model_path: str = "yolov8n.pt"
    device: str = "cpu"

    # Default inference thresholds (overridable per-request)
    default_confidence: float = 0.35
    default_iou: float = 0.45

    # Detection post-processing
    cross_class_iou_threshold: float = 0.5
    """Class-agnostic greedy NMS threshold applied after detection. Two
    boxes of different classes with IoU at or above this value are treated
    as the same physical object and only the higher-confidence one is kept.
    Ultralytics only runs NMS within each class, so this catches duplicates
    that its own NMS cannot."""

    containment_suppression_threshold: float = 0.85
    """If a detection's box overlaps a larger detection's box by at least
    this fraction of its own area, it is dropped as a duplicate or a
    structural false positive (e.g. a "refrigerator" box swallowing several
    real "bottle" boxes)."""

    # Shelf-region grouping (density-based clustering, see shelf_analysis.py)
    dbscan_eps_factor: float = 0.25
    """Detections are clustered into shelf rows using DBSCAN with a
    scale-normalized distance: two detections are in the same row if their
    vertical centers are within (dbscan_eps_factor * their average box
    height) of each other. Normalizing by box height keeps the threshold
    meaningful across a single perspective-distorted photo, where near
    objects are much larger than far ones. Tuned empirically against both
    bundled sample photos; see docs/AUDIT.md."""

    min_detections_for_confidence: int = 2
    """A region with fewer detections than this is reported as UNKNOWN
    rather than OK/UNDERSTOCKED/EMPTY, since occupancy and gap statistics
    are not meaningful from 0 or 1 data points."""

    gap_width_factor: float = 1.35
    """A horizontal gap between neighboring facings larger than
    (gap_width_factor * average facing width) is flagged as a stock gap."""

    understocked_occupancy_threshold: float = 0.75
    """A shelf region with occupancy below this ratio is flagged as
    under-stocked even if it is not fully empty."""

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Server
    max_upload_size_mb: int = 15


@lru_cache
def get_settings() -> Settings:
    return Settings()
