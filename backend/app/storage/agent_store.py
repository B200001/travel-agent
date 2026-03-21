"""In-memory singleton store for the travel chat agent."""

from __future__ import annotations

from typing import Optional

from app.services.travel_chat_agent import TravelChatAgent

_chat_agent: Optional[TravelChatAgent] = None


def set_chat_agent(agent: Optional[TravelChatAgent]) -> None:
    global _chat_agent
    _chat_agent = agent


def get_chat_agent() -> Optional[TravelChatAgent]:
    return _chat_agent


def clear_chat_agent() -> None:
    set_chat_agent(None)
