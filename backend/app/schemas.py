# backend/app/schemas.py
from pydantic import BaseModel
from typing import Dict, List, Optional

class TripPreferences(BaseModel):
    type: str = "leisure"   # leisure, cultural, adventure, party, family
    pace: str = "relaxed"   # relaxed, moderate, packed

class TripRequest(BaseModel):
    origin: str
    destination: str
    start_date: str        # "2026-04-10"
    end_date: str
    travelers: int = 1
    budget_total: float = 50000
    preferences: TripPreferences = TripPreferences()


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class TravelChatRequest(BaseModel):
    messages: List[ChatMessage]