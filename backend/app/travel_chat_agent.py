"""
Gemini-based conversational travel agent.
Handles natural chat about travel planning, destinations, and trip ideas.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai

load_dotenv()

SYSTEM_PROMPT = """You are a friendly travel-savvy friend helping someone plan a trip. Talk like a real person in a chat—warm, casual, and natural. Not like a brochure or a formal assistant.

How to sound natural:
- Write in short, flowing sentences. Mix in a question or two naturally instead of formal bullet lists.
- Avoid long bullet points and numbered lists unless the user explicitly asks for a "list" or "itinerary." Prefer paragraphs and a few short lines.
- Don't repeat the same pattern every time (e.g. "Once I have X, I can do Y"). Vary how you ask for details—sometimes one question, sometimes a quick "when are you going?" in the flow.
- Use emojis only when it feels natural (not after every paragraph).
- Say things like "when are you thinking of going?" or "what's your budget like?" instead of "Could you tell me the following: 1. ... 2. ..."
- When giving suggestions, talk like you're recommending to a friend: "Tokyo's great for that" or "April's perfect for cherry blossoms" rather than formal "I would recommend considering..."

You can suggest destinations, itineraries, budget tips, hotels, food, transport, and practical tips. Use ₹ for Indian Rupees. Support India and international travel. When they've shared enough (destination, rough dates, budget), offer to put together a day-by-day plan—but keep the tone light and conversational."""


class TravelChatAgent:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the agent and get a response."""
        contents = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                contents.append({"role": "user", "parts": [content]})
            else:
                contents.append({"role": "model", "parts": [content]})

        # Build prompt with full conversation history
        history_for_prompt = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        prompt = f"{SYSTEM_PROMPT}\n\n---\nConversation:\n{history_for_prompt}\n\nAssistant:"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Sorry, I ran into an error: {str(e)}"
