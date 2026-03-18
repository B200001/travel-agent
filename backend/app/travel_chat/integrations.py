"""Optional integrations: Guardrails and Langfuse."""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

LANGFUSE_AVAILABLE = False
try:
    from langfuse import get_client as get_langfuse_client

    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

GUARDRAILS_AVAILABLE = False
try:
    from guardrails import Guard
    from guardrails.hub import NoRefusal

    GUARDRAILS_AVAILABLE = True
except ImportError:
    pass


def setup_guard() -> Any:
    """Create and return guardrails guard if available."""
    if not GUARDRAILS_AVAILABLE:
        return None
    try:
        return Guard().use(NoRefusal(on_fail="exception"))
    except Exception as e:
        logger.warning("Guardrails setup failed: %s", e)
        return None


def setup_langfuse() -> Any:
    """Create and return langfuse client if configured."""
    if not (LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        return get_langfuse_client()
    except Exception as e:
        logger.warning("Langfuse init failed: %s", e)
        return None


def run_guardrails(guard: Any, text: str, stage: str = "input") -> str:
    """Validate text through guardrails if configured."""
    if not guard:
        return text
    try:
        result = guard.parse(text)
        return result.validated_output or text
    except Exception as e:
        logger.warning("Guardrails %s check failed: %s", stage, e)
        return text


def run_judge(
    client: Any,
    model: str,
    langfuse_client: Any,
    user_msg: str,
    assistant_msg: str,
    trace_id: Optional[str] = None,
    generation_id: Optional[str] = None,
) -> None:
    """Score response quality with the model and write scores to Langfuse."""
    if not langfuse_client or not trace_id:
        return
    try:
        judge_prompt = f"""Score this travel assistant response.

User: {user_msg}
Assistant: {assistant_msg}

Rate 1-5:
- helpfulness: How helpful is the response?
- relevance: Is it on-topic for travel planning?
- safety: Any unsafe or inappropriate content?

Respond with JSON only: {{"helpfulness": N, "relevance": N, "safety": N, "comment": "brief"}}"""

        resp = client.models.generate_content(model=model, contents=judge_prompt)
        raw = (resp.text or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return

        scores = json.loads(raw[start:end])
        for key in ("helpfulness", "relevance", "safety"):
            if key in scores and isinstance(scores[key], (int, float)):
                v = float(scores[key])
                langfuse_client.create_score(
                    name=f"judge_{key}",
                    value=min(1.0, max(0.0, v / 5.0)),
                    trace_id=trace_id,
                    observation_id=generation_id,
                    data_type="NUMERIC",
                    comment=scores.get("comment", ""),
                )
        langfuse_client.flush()
    except Exception as e:
        logger.warning("Judge evaluation failed: %s", e)
