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

    # Shelf-region grouping
    row_gap_factor: float = 1.6
    """A vertical gap between detections larger than (row_gap_factor * median
    box height) is treated as the boundary between two shelf rows."""

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
