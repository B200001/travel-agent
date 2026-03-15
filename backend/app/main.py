# backend/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from dotenv import load_dotenv
import os

from app.schemas import TripRequest, TravelChatRequest
from app.complete_travel_agent_hinglish import CompleteTravelAgent
from app.travel_chat_agent import TravelChatAgent

# Load .env from backend/ (parent of app/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

agent = None
chat_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, chat_agent
    # One Gemini key for both trip planner and chat
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required in .env")
    agent = CompleteTravelAgent(api_key)
    try:
        chat_agent = TravelChatAgent(api_key=api_key)
    except ValueError:
        chat_agent = None  # fallback if something else goes wrong
    yield


app = FastAPI(title="Travel Agent API", version="1.0", lifespan=lifespan)

# CORS - allow localhost and production frontend (set CORS_ORIGINS in production)
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").strip().split(",")
_origins_list = [o.strip() for o in _cors_origins if o.strip()]
# Allow any *.vercel.app so production and preview deploys work without exact URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the Travel Agent API"}

@app.post("/api/plan-trip")
async def plan_trip(request: TripRequest):
    try:
        trip_data = request.model_dump()
        result = agent.plan_trip(trip_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}