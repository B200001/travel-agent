# Deploy Travel Agent on Vercel

Your app has two parts:
- **Frontend** (Next.js) → deploy on **Vercel**
- **Backend** (FastAPI) → deploy on **Render** or **Railway**; Vercel cannot run long-lived Python servers

Follow these steps in order. Use **Part 1A (Render)** or **Part 1B (Railway)** for the backend.

---

## Part 1A: Deploy Backend on Render (free tier)

Render gives a public URL for your FastAPI app (e.g. `https://travel-agent-api.onrender.com`).

### 1A.1 Create Render account and connect repo

1. Go to [render.com](https://render.com) and sign in (e.g. GitHub).
2. Click **Dashboard** → **New** → **Web Service**.
3. Connect your GitHub account if needed, then select the repo (e.g. `travel-agent`).
4. Render may detect the `render.yaml` in the repo. If it asks to create from Blueprint, choose **Apply** so it uses the existing `render.yaml` (which points to the `backend` folder and runs FastAPI).

**If you create the service manually instead:**

1. After selecting the repo, set **Root Directory** to `backend`.
2. **Runtime**: Python 3.
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Instance type**: Free (or Starter if you need more uptime).

### 1A.2 Add environment variables on Render

In the service → **Environment** tab, add:

| Key | Value |
|-----|--------|
| `ANTHROPIC_API_KEY` | your-anthropic-key (for Plan Trip form) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | your-gemini-key (for Chat) |
| `GOOGLE_GENAI_USE_VERTEXAI` | `False` (if using Gemini API) |
| `CORS_ORIGINS` | Set after Part 2 – your Vercel URL, e.g. `https://your-app.vercel.app` |

Save. Render will redeploy.

### 1A.3 Get the public URL

1. In the service, open **Settings** or the top of the page for the service URL.
2. Copy it, e.g. `https://travel-agent-api.onrender.com` (no trailing slash).
3. Use this as `NEXT_PUBLIC_API_URL` in Vercel and in `CORS_ORIGINS` on Render.

### 1A.4 Test the backend

```bash
curl https://YOUR-RENDER-URL/api/health
```

You should get `{"status":"ok"}`.

**Note:** On the free tier, the service may spin down after inactivity; the first request after that can take 30–60 seconds (cold start).

---

## Part 1B: Deploy Backend on Railway (free tier)

Railway gives a public URL for your FastAPI app. You’ll add your env vars and get a URL like `https://your-app.up.railway.app`.

### 1.1 Create Railway account and project

1. Go to [railway.app](https://railway.app) and sign in (e.g. GitHub).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Connect GitHub and select this repo (or push the code to a repo first).
4. When asked “What do you want to deploy?”, choose **Add a service** and then **Empty service** (we’ll add the backend config next).

### 1.2 Configure the service to run the backend

Your backend lives in the `backend/` folder. Railway needs to know that.

**Option A – Using Railway dashboard**

1. In the new service, open **Settings**.
2. Set **Root Directory** to `backend`.
3. Set **Build Command** to:  
   `pip install -r requirements.txt`
4. Set **Start Command** to:  
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set **Watch Paths** to `backend/**` (so only backend changes trigger deploys).

**Option B – Using config files in the repo (recommended)**

Create these in your repo so Railway can use them.

**File: `backend/railway.json`** (or `backend/railway.toml`)

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

Create **`backend/nixpacks.toml`** so Nixpacks uses Python and installs deps from `requirements.txt`:

```toml
[phases.setup]
nixPkgs = ["python311"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
```

Then in Railway:

- **Root Directory**: `backend`
- Leave build/start command empty if you use `nixpacks.toml`; otherwise set Build = `pip install -r requirements.txt`, Start = `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

### 1.3 Add environment variables on Railway

In the same service, go to **Variables** and add (use the same values you use locally):

| Variable | Value (example) | Required |
|----------|------------------|----------|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | your-gemini-key | Yes (chat + plan trip) |
| `GOOGLE_GENAI_USE_VERTEXAI` | `False` | If using Gemini API |
| `CORS_ORIGINS` | (set in Part 2 after Vercel URL) | Yes for production |

Do **not** commit real keys to the repo. Only set them in Railway (and later in Vercel for the frontend).

### 1.4 Get the public URL

1. In the service, open **Settings** → **Networking** → **Generate Domain** (or use the default one).
2. Copy the URL, e.g. `https://travel-agent-api.up.railway.app`.  
   You’ll use this as `NEXT_PUBLIC_API_URL` and in `CORS_ORIGINS`.

### 1.5 Redeploy and test

After saving variables, redeploy. Then:

```bash
curl https://YOUR-RAILWAY-URL/api/health
```

You should get `{"status":"ok"}`.

---

## Part 2: Deploy Frontend on Vercel

### 2.1 Push code and connect repo

1. Push your project to GitHub (if not already).
2. Go to [vercel.com](https://vercel.com) and sign in (e.g. GitHub).
3. Click **Add New** → **Project** and import the same repo.

### 2.2 Set Root Directory to `frontend`

1. In the import screen, find **Root Directory**.
2. Click **Edit** and set it to **`frontend`**.
3. Leave **Framework Preset** as Next.js (auto-detected).

### 2.3 Set environment variable for API URL

1. In the same screen (or later in **Project → Settings → Environment Variables**), add:

| Name | Value |
|------|--------|
| `NEXT_PUBLIC_API_URL` | Your backend URL (no trailing slash) |

Examples:  
- Render: `https://travel-agent-api.onrender.com`  
- Railway: `https://travel-agent-api.up.railway.app`

2. Apply to **Production**, **Preview**, and **Development** if you want.

### 2.4 Deploy

Click **Deploy**. Vercel will build and deploy the Next.js app from the `frontend` folder.

### 2.5 Allow the Vercel URL in backend CORS

Your backend reads `CORS_ORIGINS` from the environment. Add your Vercel URL so the browser can call the API.

1. **Render**: Service → **Environment** → add or edit **`CORS_ORIGINS`** = `https://your-app.vercel.app`  
   **Railway**: Service → **Variables** → add or edit **`CORS_ORIGINS`** = `https://your-app.vercel.app`  
   For preview deployments you can add multiple origins separated by commas, e.g.  
   `https://your-app.vercel.app,https://your-app-git-preview-xxx.vercel.app`

2. Redeploy the backend after changing variables.

Now the frontend on Vercel can call the backend without CORS errors.

---

## Part 3: Optional – `vercel.json` at repo root

If your repo root is the Git root and the Next app is in `frontend/`, you can add at the **repo root**:

**`vercel.json`** (in project root):

```json
{
  "rootDirectory": "frontend"
}
```

This tells Vercel to treat `frontend` as the project root when you don’t set it in the dashboard. If you already set Root Directory to `frontend` in the Vercel project settings, this file is optional but keeps config in code.

---

## Part 4: Checklist

- [ ] Backend on **Render** or **Railway**: Root = `backend`, env vars set (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `CORS_ORIGINS`, optional `GOOGLE_GENAI_USE_VERTEXAI`).
- [ ] Backend health check: `curl https://YOUR-BACKEND-URL/api/health` returns `{"status":"ok"}`.
- [ ] Frontend on Vercel: Root = `frontend`, `NEXT_PUBLIC_API_URL` = your backend URL.
- [ ] `CORS_ORIGINS` on the backend includes your Vercel URL (e.g. `https://your-app.vercel.app`).
- [ ] Open the Vercel app URL: Chat and Plan Trip should work.

---

## Troubleshooting

**CORS errors in browser**  
- Ensure `CORS_ORIGINS` on Render/Railway includes the exact origin (e.g. `https://your-app.vercel.app`).  
- No trailing slash in the origin.

**Chat / Plan Trip return 503 or 500**  
- Check backend logs (Render: Logs tab; Railway: Deployments → View Logs).  
- Verify `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) are set correctly.

**Frontend shows “Something went wrong”**  
- Confirm `NEXT_PUBLIC_API_URL` in Vercel matches your backend URL (no trailing slash).  
- Rebuild and redeploy the frontend after changing env vars.

**Render free tier cold starts**  
- After 15 min of no traffic, the service sleeps. The first request may take 30–60 seconds; then it’s fast.

**Preview deployments (e.g. PRs)**  
- Each Vercel preview gets a URL like `https://your-app-git-branch-username.vercel.app`.  
- Add that URL to `CORS_ORIGINS` (comma-separated) if you want to test previews against the same backend.
