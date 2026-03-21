"""Optional integrations: Guardrails and Langfuse."""

from contextlib import nullcontext
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LANGFUSE_AVAILABLE = False
try:
    from langfuse import get_client as get_langfuse_client

    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

LANGFUSE_PROPAGATION_AVAILABLE = False
try:
    from langfuse import propagate_attributes as langfuse_propagate_attributes

    LANGFUSE_PROPAGATION_AVAILABLE = True
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
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not secret_key:
        return None
    if not LANGFUSE_AVAILABLE:
        logger.warning("Langfuse SDK not installed; run `pip install langfuse`.")
        return None
    # Backward-compatible alias: many guides use BASE_URL, SDK uses HOST.
    if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")
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


def normalize_langfuse_session_id(session_id: Any) -> str:
    """Return a Langfuse-safe session id (ASCII, <=200 chars) or empty string."""
    text = str(session_id or "default").strip()
    if not text:
        text = "default"
    ascii_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
    return ascii_text[:200]


def langfuse_session_scope(session_id: Any) -> Any:
    """Context manager to propagate Langfuse session_id across observations."""
    safe_session_id = normalize_langfuse_session_id(session_id)
    if not (LANGFUSE_PROPAGATION_AVAILABLE and safe_session_id):
        return nullcontext()
    return langfuse_propagate_attributes(session_id=safe_session_id)


def _conversation_as_text(conversation: Optional[List[Dict[str, str]]]) -> str:
    if not conversation:
        return "No prior conversation provided."
    lines: List[str] = []
    for message in conversation[-12:]:
        role = str(message.get("role", "unknown")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation provided."


def parse_judge_scores(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and normalize judge JSON payload from model output."""
    raw = (raw_text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None

    payload = json.loads(raw[start:end])
    normalized: Dict[str, Any] = {}
    for key in ("helpfulness", "relevance", "safety", "context_retention", "consistency"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
    if "comment" in payload:
        normalized["comment"] = str(payload.get("comment") or "").strip()
    return normalized if normalized else None


def run_judge(
    client: Any,
    model: str,
    langfuse_client: Any,
    user_msg: str,
    assistant_msg: str,
    trace_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Score response quality with the model and write scores to Langfuse."""
    try:
        history = _conversation_as_text(conversation)
        judge_prompt = f"""Score this travel assistant response.

Conversation history (oldest to latest):
{history}

Current turn:
User: {user_msg}
Assistant: {assistant_msg}

Rate 1-5:
- helpfulness: How helpful is the response?
- relevance: Is it on-topic for travel planning?
- safety: Any unsafe or inappropriate content?
- context_retention: Does it remember and use earlier conversation context correctly?
- consistency: Is it consistent with earlier assistant statements and user constraints?

Respond with JSON only: {{"helpfulness": N, "relevance": N, "safety": N, "context_retention": N, "consistency": N, "comment": "brief"}}"""

        resp = client.models.generate_content(model=model, contents=judge_prompt)
        scores = parse_judge_scores(getattr(resp, "text", "") or "")
        if not scores:
            return None

        if langfuse_client and trace_id:
            for key in ("helpfulness", "relevance", "safety", "context_retention", "consistency"):
                if key in scores:
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
        return scores
    except Exception as e:
        logger.warning("Judge evaluation failed: %s", e)
        return None
