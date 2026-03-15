# Step-by-Step: Travel Agent with Gemini Chat Interface

This guide explains how to build the full stack: **Backend (FastAPI + Gemini chat + existing travel planner)** and **Frontend (Next.js with Chat + Form tabs)**. Nothing is skipped.

---

## Prerequisites

- **Python 3.9+** (3.10+ recommended)
- **Node.js 18+** and npm
- **API key**: **GEMINI_API_KEY** or **GOOGLE_API_KEY** – used for both the chat tab and the form-based trip planner (Gemini)

---

## Part 1: Project Structure

Your repo should look like this:

```
your-project/
├── backend/
│   ├── .env                 # API keys (see below)
│   ├── requirements.txt     # Python deps
│   └── app/
│       ├── main.py          # FastAPI app + routes
│       ├── schemas.py       # Pydantic models (Trip + Chat)
│       ├── travel_chat_agent.py   # Gemini chat agent
│       └── complete_travel_agent_hinglish.py  # Gemini-based trip planner
├── frontend/
│   ├── .env                 # NEXT_PUBLIC_API_URL (optional)
│   └── src/
│       ├── app/page.tsx     # UI (tabs + chat + form)
│       ├── lib/api.ts       # API client
│       └── types.ts         # TypeScript types
└── TRAVEL_CHAT_SETUP.md     # This file
```

---

## Part 2: Backend Setup

### Step 2.1 – Environment variables

Create or edit `backend/.env`:

```env
# Required for both Chat tab and Plan Trip (Gemini). Use one of:
GEMINI_API_KEY=your-gemini-key
# OR
GOOGLE_API_KEY=your-google-key

# If using Vertex AI instead of Gemini API, set:
# GOOGLE_GENAI_USE_VERTEXAI=True
# For standard Gemini API (recommended):
GOOGLE_GENAI_USE_VERTEXAI=False
```

- Get **Gemini API key**: https://aistudio.google.com/app/apikey

### Step 2.2 – Python dependencies

In `backend/requirements.txt` ensure you have:

```text
fastapi>=0.109.0
google-genai>=1.0.0
uvicorn[standard]>=0.27.0
python-dotenv>=1.0.0
langgraph
langchain-core
pydantic>=2.0
```

Then install:

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2.3 – Chat API schemas

In `backend/app/schemas.py` add the chat request/response models **after** your existing `TripRequest`:

```python
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class TravelChatRequest(BaseModel):
    messages: List[ChatMessage]
```

- Keep existing imports: `from typing import Dict, List, Optional` (and `List` is used by `TravelChatRequest`).

### Step 2.4 – Gemini travel chat agent

Create `backend/app/travel_chat_agent.py`:

```python
"""
Gemini-based conversational travel agent.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()

SYSTEM_PROMPT = """You are a friendly, knowledgeable travel agent assistant. Your role is to help users plan trips through natural conversation.

You can:
- Suggest destinations based on preferences (budget, season, interests)
- Help with itinerary ideas, things to do, and places to visit
- Recommend hotels, restaurants, and local transport options
- Give travel tips, visa info, and packing suggestions
- Answer questions about weather, best time to visit, and cultural tips

Guidelines:
- Be conversational, warm, and helpful
- Ask follow-up questions when you need more info (destination, dates, budget, travelers)
- Use emojis occasionally to keep it friendly
- Keep responses concise but informative
- When user provides enough details, offer to create a full trip plan
- Use ₹ for Indian Rupees when discussing budget
- Support both India and international travel"""


class TravelChatAgent:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send conversation history and get the next assistant reply."""
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
```

- **Behavior**: One turn per call; you send full history (including the latest user message) and get one assistant message back.

### Step 2.5 – FastAPI app: lifespan and chat route

In `backend/app/main.py`:

1. **Imports** – add chat schema and travel chat agent:

```python
from app.schemas import TripRequest, TravelChatRequest
from app.complete_travel_agent_hinglish import CompleteTravelAgent
from app.travel_chat_agent import TravelChatAgent
```

2. **Globals** – add a global for the chat agent:

```python
agent = None
chat_agent = None
```

3. **Lifespan** – create both agents with the same Gemini key:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, chat_agent
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required in .env")
    agent = CompleteTravelAgent(api_key)
    try:
        chat_agent = TravelChatAgent(api_key=api_key)
    except ValueError:
        chat_agent = None
    yield
```

4. **CORS** – keep allowing the frontend origin (e.g. `http://localhost:3000`).

5. **New route** – add the chat endpoint:

```python
@app.post("/api/travel-chat")
async def travel_chat(request: TravelChatRequest):
    if chat_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Travel chat requires GEMINI_API_KEY or GOOGLE_API_KEY in .env"
        )
    try:
        messages = [m.model_dump() for m in request.messages]
        response = chat_agent.chat(messages)
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- **Contract**: Request body `{ "messages": [ { "role": "user"|"assistant", "content": "..." } ] }`, response `{ "message": "..." }`.

---

## Part 3: Frontend Setup

### Step 3.1 – Environment (optional)

In `frontend/.env`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If you omit this, the client uses `http://localhost:8000` by default.

### Step 3.2 – TypeScript types for chat

In `frontend/src/types.ts` add **before** `TravelPlanResult`:

```typescript
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface TravelChatResponse {
  message: string;
}
```

### Step 3.3 – API client for chat

In `frontend/src/lib/api.ts`:

1. Extend imports:

```typescript
import type { TripRequest, TravelPlanResult, ChatMessage, TravelChatResponse } from "@/types";
```

2. Add the chat function (same file, e.g. before `planTrip`):

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function sendTravelChat(messages: ChatMessage[]): Promise<TravelChatResponse> {
  const res = await fetch(`${API_URL}/api/travel-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Chat API error");
  }
  return res.json();
}
```

### Step 3.4 – Page: tabs and chat UI

In `frontend/src/app/page.tsx`:

1. **Imports** – add `useRef`, `useEffect`, `sendTravelChat`, and `ChatMessage`:

```typescript
import { useState, useRef, useEffect } from "react";
import { planTrip, sendTravelChat } from "@/lib/api";
import type { TripRequest, TravelPlanResult, ChatMessage } from "@/types";
```

2. **Tab type and state** – add a tab and chat state at the top of the component:

```typescript
type Tab = "plan" | "chat";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  // ... existing formData, loading, error, result ...
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your travel assistant. Tell me where you'd like to go, your budget, dates, or what kind of trip you're looking for - I'll help you plan it!",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [chatMessages]);
```

3. **Send handler** – when user sends a message, append it, call the API, append assistant reply (or error):

```typescript
  const handleChatSend = async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: msg }]);
    setChatLoading(true);
    try {
      const nextMessages: ChatMessage[] = [...chatMessages, { role: "user", content: msg }];
      const res = await sendTravelChat(nextMessages);
      setChatMessages((prev) => [...prev, { role: "assistant", content: res.message }]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };
```

4. **Tabs in the UI** – after the header, render two tabs and conditionally show Chat or Form:

- **Tabs**: Two buttons, e.g. "💬 Chat" and "📋 Plan Trip (Form)". `activeTab` toggles between `"chat"` and `"plan"`.
- **Chat tab**:
  - A scrollable area that maps `chatMessages` to bubbles (user right, assistant left).
  - A "Typing..." indicator when `chatLoading` is true.
  - A `div` with `ref={chatEndRef}` at the bottom so `scrollToBottom` runs when messages change.
  - Input and "Send" button: input value = `chatInput`, onChange updates `chatInput`, onKeyDown (Enter without Shift) and Send button call `handleChatSend()`.
- **Plan tab**: Your existing form and results (same as before).

5. **Wrap the form and results** – so they only show when `activeTab === "plan"`:

- Wrap the form + error + results in `{activeTab === "plan" && ( <> ... </> )}`.

That gives you one page with two modes: **Chat** and **Plan Trip** (both use Gemini).

---

## Part 4: Run and Test

### Start backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `curl http://localhost:8000/api/health`
- Chat:  
  `curl -X POST http://localhost:8000/api/travel-chat -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Weekend trip from Mumbai under 20k?"}]}'`

### Start frontend

```bash
cd frontend
npm install
npm run dev
```

- Open `http://localhost:3000`.
- **Chat** tab: send a message; you should get a Gemini reply.
- **Plan Trip** tab: submit the form; you should get the full Gemini-generated plan.

### If chat returns 503

- Backend could not create `TravelChatAgent` (missing or invalid Gemini/Google key).
- Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `backend/.env` and restart the backend.

### If chat returns 404 (model not found)

- With Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=True`), the project may not have access to the model.
- Set `GOOGLE_GENAI_USE_VERTEXAI=False` and use a Gemini API key from AI Studio instead.

---

## Summary Checklist

| Step | What |
|------|------|
| 1 | `backend/.env` with GEMINI_API_KEY (or GOOGLE_API_KEY), and GOOGLE_GENAI_USE_VERTEXAI=False if using Gemini API |
| 2 | `backend/requirements.txt` includes `google-genai>=1.0.0` and install deps in a venv |
| 3 | `backend/app/schemas.py`: add `ChatMessage` and `TravelChatRequest` |
| 4 | `backend/app/travel_chat_agent.py`: Gemini client, system prompt, `chat(messages)` |
| 5 | `backend/app/main.py`: import chat agent and schema, lifespan creates `TravelChatAgent`, add `POST /api/travel-chat` |
| 6 | `frontend/src/types.ts`: add `ChatMessage` and `TravelChatResponse` |
| 7 | `frontend/src/lib/api.ts`: add `sendTravelChat(messages)` |
| 8 | `frontend/src/app/page.tsx`: tab state, chat state, `handleChatSend`, tabs UI, chat bubbles + input, form only when Plan tab |

Once these are in place, you have the full travel agent: **Gemini-based chat** plus **form-based full trip plan** in one app.
