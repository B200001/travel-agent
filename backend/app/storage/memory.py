"""Long-term preference persistence helpers."""

import json
from typing import Dict

from app.services.constants import LONG_TERM_STORAGE_PATH


def ensure_storage_dir() -> None:
    """Create data directory if missing."""
    LONG_TERM_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_long_term_memory() -> Dict[str, Dict]:
    """Load all session preferences from disk."""
    ensure_storage_dir()
    if not LONG_TERM_STORAGE_PATH.exists():
        return {}
    try:
        with open(LONG_TERM_STORAGE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_long_term_memory(data: Dict[str, Dict]) -> None:
    """Persist all session preferences to disk."""
    ensure_storage_dir()
    with open(LONG_TERM_STORAGE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def initialize_long_term_memory_store() -> None:
    """Ensure storage file is present at startup."""
    ensure_storage_dir()
    if not LONG_TERM_STORAGE_PATH.exists():
        save_long_term_memory({})
