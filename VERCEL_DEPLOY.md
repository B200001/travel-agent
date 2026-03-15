# Deploy Travel Agent on Vercel

Your app has two parts:
- **Frontend** (Next.js) → deploy on **Vercel**
- **Backend** (FastAPI) → deploy on **Railway** (or Render); Vercel cannot run long-lived Python servers

Follow these steps in order.

---

## Part 1: Deploy Backend on Railway (free tier)

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
| `NEXT_PUBLIC_API_URL` | `https://YOUR-RAILWAY-URL` (no trailing slash) |

Example: `https://travel-agent-api.up.railway.app`

2. Apply to **Production**, **Preview**, and **Development** if you want.

### 2.4 Deploy

Click **Deploy**. Vercel will build and deploy the Next.js app from the `frontend` folder.

### 2.5 Allow the Vercel URL in backend CORS

Your backend already reads `CORS_ORIGINS` (see repo). Add your Vercel URL so the browser can call the API.

1. In **Railway** → your backend service → **Variables**, add or edit:
   - **`CORS_ORIGINS`** = `https://your-app.vercel.app`
   - For preview deployments you can use:  
     `https://your-app.vercel.app,https://*.vercel.app`  
     (if your backend supports wildcards; the current code splits by comma, so list main domain + preview domain if needed).

2. Redeploy the backend on Railway after changing variables.

Now the frontend on Vercel can call the backend on Railway without CORS errors.

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

- [ ] Backend on Railway: Root = `backend`, env vars set (`GEMINI_API_KEY` or `GOOGLE_API_KEY`, `CORS_ORIGINS`, optional `GOOGLE_GENAI_USE_VERTEXAI`).
- [ ] Backend health check: `curl https://YOUR-RAILWAY-URL/api/health` returns `{"status":"ok"}`.
- [ ] Frontend on Vercel: Root = `frontend`, `NEXT_PUBLIC_API_URL` = `https://YOUR-RAILWAY-URL`.
- [ ] `CORS_ORIGINS` on Railway includes your Vercel URL (e.g. `https://your-app.vercel.app`).
- [ ] Open the Vercel app URL: Chat and Plan Trip should work.

---

## Troubleshooting

**CORS errors in browser**  
- Ensure `CORS_ORIGINS` on Railway includes the exact origin (e.g. `https://your-app.vercel.app`).  
- No trailing slash in the origin.

**Chat / Plan Trip return 503 or 500**  
- Check Railway logs (Deployments → View Logs).  
- Verify `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set correctly on Railway.

**Frontend shows “Something went wrong”**  
- Confirm `NEXT_PUBLIC_API_URL` in Vercel matches the Railway URL (no trailing slash).  
- Rebuild and redeploy the frontend after changing env vars.

**Preview deployments (e.g. PRs)**  
- Each preview gets a URL like `https://your-app-git-branch-username.vercel.app`.  
- Add that URL (or a pattern if your backend supports it) to `CORS_ORIGINS` if you want to test previews against the same backend.
