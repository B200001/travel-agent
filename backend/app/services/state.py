"""Typed state used by the LangGraph workflow."""

from typing import Annotated, Dict, List, Optional, TypedDict


def append_messages(
    current: Optional[List[Dict[str, str]]],
    new: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Reducer to append new messages to conversation history."""
    if not current:
        return new or []
    if not new:
        return current
    return [*current, *new]


class TravelChatState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, str]], append_messages]
    session_id: str
    user_content: str
    short_term: List[Dict[str, str]]
    prompt: str
    cache_key: str
    cache_hit: bool
    cached_result: str
    task_graph: List[Dict[str, object]]
    task_outputs: Dict[str, str]
    execution_trace: List[str]
    all_tasks_completed: bool
    planning_summary: str
    blocked: bool
    blocked_reason: str
    quick_reply: bool
    emergency_query: bool
    fast_search_query: bool
    result: str
