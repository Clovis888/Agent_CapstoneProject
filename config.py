import os
from dataclasses import dataclass
import google.auth
# Auth setup (similar to reference)
_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
@dataclass
class ConciergeConfig:
    """Configuration for concierge agents."""
    model_name: str = "gemini-2.5-flash"
    max_retries: int = 3
    memory_retention_days: int = 30  # For MemoryBank
config = ConciergeConfig()
