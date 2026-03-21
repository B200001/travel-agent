"""Business logic for chat agent lifecycle and message handling."""

from __future__ import annotations

import os
from typing import Dict, Iterator, List

from app.storage.agent_store import clear_chat_agent, get_chat_agent, set_chat_agent
from app.services.travel_chat_agent import TravelChatAgent


def initialize_chat_agent() -> None:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required in .env")
    set_chat_agent(TravelChatAgent(api_key=api_key))


def shutdown_chat_agent() -> None:
    agent = get_chat_agent()
    if agent is not None:
        agent.close()
    clear_chat_agent()


def get_ready_chat_agent() -> TravelChatAgent:
    agent = get_chat_agent()
    if agent is None:
        raise ValueError("Travel chat agent is not initialized.")
    return agent


def build_chat_payload(messages: List[Dict[str, str]], session_id: str | None = None) -> Dict[str, object]:
    agent = get_ready_chat_agent()
    return agent.chat_payload(messages, session_id=session_id)


def build_chat_stream(messages: List[Dict[str, str]], session_id: str | None = None) -> Iterator[Dict[str, object]]:
    agent = get_ready_chat_agent()
    return agent.chat_payload_stream(messages, session_id=session_id)
