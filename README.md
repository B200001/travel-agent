# Travel Agent

A full-stack travel planning application with an AI-powered chat assistant and a detailed trip planner. The chat uses natural conversation to suggest destinations, itineraries, and travel tips; the plan form generates day-by-day itineraries, hotels, transport, and budget breakdowns.

## Features

- **Chat interface** — Conversational travel assistant (Gemini) for destination ideas, budgets, and recommendations
- **Plan Trip form** — Structured trip planner with flights, hotels, local transport, restaurants, sightseeing, weather, packing list, and budget breakdown
- **Modern stack** — Next.js frontend, FastAPI backend, LangGraph for orchestration

## Tech Stack

| Layer      | Technology                    |
|-----------|-------------------------------|
| Frontend   | Next.js 14, React, Tailwind CSS |
| Backend    | FastAPI, Python 3.9+          |
| AI / Chat  | Google Gemini (google-genai)   |
| Trip plan  | LangGraph, Claude/Gemini       |

## Project Structure

```
├── frontend/          # Next.js app (Chat + Plan Trip UI)
├── backend/           # FastAPI app (travel-chat + plan-trip APIs)
│   ├── app/
│   │   ├── main.py
│   │   ├── travel_chat_agent.py
│   │   ├── complete_travel_agent_hinglish.py
│   │   └── schemas.py
│   ├── requirements.txt
│   └── .env           # Not committed; see Environment below
├── render.yaml        # Optional: Render.com backend config
├── vercel.json        # Vercel: deploy frontend from frontend/
└── README.md
```

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- API key: **GEMINI_API_KEY** or **GOOGLE_API_KEY** ([Google AI Studio](https://aistudio.google.com/app/apikey))

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
# or GOOGLE_API_KEY=your-google-api-key
GOOGLE_GENAI_USE_VERTEXAI=False
CORS_ORIGINS=http://localhost:3000
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The app uses `http://localhost:8000` as the API URL by default.

## Deployment

- **Frontend** — Deploy the `frontend/` directory to [Vercel](https://vercel.com). Set Root Directory to `frontend` and add `NEXT_PUBLIC_API_URL` to your backend URL.
- **Backend** — Deploy the `backend/` directory to [Render](https://render.com) or [Railway](https://railway.app). Set root to `backend`, build with `pip install -r requirements.txt`, start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Add the same env vars as above and set `CORS_ORIGINS` to your Vercel frontend URL.

See the dashboard docs for each platform for exact steps. The repo includes a `render.yaml` for Render and a `vercel.json` for Vercel.

## Environment Variables

| Variable | Where | Description |
|----------|--------|-------------|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Backend | Required for chat and trip planning. |
| `GOOGLE_GENAI_USE_VERTEXAI` | Backend | Set to `False` for Gemini API; `True` for Vertex AI. |
| `CORS_ORIGINS` | Backend | Comma-separated allowed origins (e.g. your Vercel URL). |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend base URL (e.g. `https://your-api.onrender.com`). |

## License

MIT
