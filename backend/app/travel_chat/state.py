"""Typed state used by the LangGraph workflow."""

from typing import Annotated, Dict, List, Literal, Optional, TypedDict


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
    needs_search: bool
    tool_mode: Literal["search", "custom"]
    result: str
