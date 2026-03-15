# How to Upload This Project to GitHub

Follow these steps in order.

---

## 1. Create a new repo on GitHub (empty)

1. Go to [github.com](https://github.com) and sign in.
2. Click the **+** (top right) → **New repository**.
3. Choose a **Repository name** (e.g. `travel-agent-app`).
4. Set visibility to **Public** (or Private).
5. **Do not** check "Add a README" or "Add .gitignore" (we already have .gitignore).
6. Click **Create repository**.

---

## 2. Open terminal in your project folder

```bash
cd "/Users/bhuwanjain/Desktop/untitled folder 3"
```

---

## 3. Initialize Git and make the first commit

Run these commands one by one:

```bash
# Initialize a new Git repo
git init

# Stage all files (respects .gitignore)
git add .

# First commit
git commit -m "Initial commit: Travel agent with Gemini chat and Vercel deploy"
```

---

## 4. Connect to GitHub and push

GitHub will show you a URL like `https://github.com/YOUR_USERNAME/travel-agent-app.git`.

Run (replace with **your** repo URL):

```bash
# Rename default branch to main (if needed)
git branch -M main

# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git push -u origin main
```

Example if your username is `johndoe` and repo is `travel-agent-app`:

```bash
git remote add origin https://github.com/johndoe/travel-agent-app.git
git push -u origin main
```

---

## 5. If Git asks for login

- **HTTPS**: GitHub may ask for username + **Personal Access Token** (not your password). Create one: GitHub → Settings → Developer settings → Personal access tokens → Generate new token. Use the token as the password when pushing.
- **SSH**: If you use SSH keys, use the SSH URL instead: `git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git`

---

## Important: .env is not uploaded

The `.gitignore` file excludes `.env`, so your **API keys stay local** and are never pushed. On a new machine or for deployment (e.g. Vercel, Railway), you’ll add the same variables in their dashboard or in a new `.env` there.

---

## Later: push changes

After editing code:

```bash
git add .
git commit -m "Describe what you changed"
git push
```
