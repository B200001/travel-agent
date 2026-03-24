# ============================================================================
# IMPORTS - Bringing in code from other libraries and files
# ============================================================================

# contextlib provides utilities for working with context managers (setup/cleanup)
from contextlib import asynccontextmanager

# FastAPI is the web framework - handles HTTP requests/responses
from fastapi import FastAPI

# CORS middleware allows browsers to make requests from different domains
from fastapi.middleware.cors import CORSMiddleware

# Path helps work with file/folder paths in a cross-platform way
from pathlib import Path

# dotenv loads environment variables from a .env file (API keys, secrets, etc.)
from dotenv import load_dotenv

# os provides access to environment variables and operating system features
import os
import logging
from logging.handlers import RotatingFileHandler

# Import route handlers (endpoints) for chat functionality
from app.routes.chat import router as chat_router

# Import route handlers for system endpoints (health checks, etc.)
from app.routes.system import router as system_router

# Import functions to set up and tear down the chat agent
from app.services.chat_service import initialize_chat_agent, shutdown_chat_agent


# ============================================================================
# ENVIRONMENT SETUP - Load configuration from .env file
# ============================================================================

# Load environment variables from .env file
# __file__ = current file path (main.py)
# .resolve() = convert to absolute path
# .parent.parent = go up 2 directories (from app/main.py to backend/)
# / ".env" = append .env filename
# This finds the .env file in the backend/ directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ============================================================================
# LOGGING SETUP - Write application logs to a rotating file
# ============================================================================
def configure_logging() -> None:
    """Configure app + server logging to write into a file."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    default_log_file = Path(__file__).resolve().parent.parent / "logs" / "app.log"
    log_file = Path(os.getenv("LOG_FILE_PATH", str(default_log_file)))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, "_app_file_handler", True)

    # Attach file handler to root and uvicorn access logger.
    # Root captures most application/server logs without duplication.
    for logger_name in ("", "uvicorn.access"):
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(log_level)
        if not any(getattr(h, "_app_file_handler", False) for h in target_logger.handlers):
            target_logger.addHandler(file_handler)

configure_logging()


# ============================================================================
# LIFESPAN MANAGER - Code that runs when the app starts and stops
# ============================================================================

@asynccontextmanager  # Decorator that makes this function a context manager
async def lifespan(app: FastAPI):
    """Manages startup and shutdown of the application.
    
    This function runs:
    - BEFORE the app starts accepting requests (code before 'yield')
    - AFTER the app stops (code after 'yield')
    
    Think of it like:
        try:
            # Startup code here
            yield  # App runs here
        finally:
            # Shutdown code here
    
    Args:
        app: The FastAPI application instance
    """
    # STARTUP: Initialize the chat agent (load models, connect to database, etc.)
    # This happens once when the server starts
    initialize_chat_agent()
    
    # YIELD: The app runs here and handles requests
    # Everything between startup and shutdown happens at this point
    yield
    
    # SHUTDOWN: Clean up resources (close database connections, etc.)
    # This happens when the server is shutting down (Ctrl+C or deployment stops)
    shutdown_chat_agent()


# ============================================================================
# CREATE THE FASTAPI APPLICATION
# ============================================================================

# Create the main FastAPI application instance
app = FastAPI(
    title="Travel Chat API",      # Shows up in auto-generated documentation
    version="1.0",                 # API version number
    lifespan=lifespan              # Use our startup/shutdown manager
)


# ============================================================================
# CORS CONFIGURATION - Allow web browsers to access this API
# ============================================================================

# CORS (Cross-Origin Resource Sharing) allows browsers to make requests
# from one domain (e.g., localhost:3000) to another (e.g., localhost:8000)
# Without CORS, browsers block these "cross-origin" requests for security

# Get allowed origins from environment variable, default to localhost:3000
# Example: CORS_ORIGINS="http://localhost:3000,https://myapp.com"
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").strip().split(",")

# Clean up the list: remove whitespace and empty strings
# Input:  ["http://localhost:3000 ", " ", "https://myapp.com"]
# Output: ["http://localhost:3000", "https://myapp.com"]
_origins_list = [o.strip() for o in _cors_origins if o.strip()]

# Add CORS middleware to the app
# Middleware = code that runs on every request/response
app.add_middleware(
    CORSMiddleware,  # The CORS handler from FastAPI
    
    # allow_origins: List of exact URLs that can access this API
    # Example: ["http://localhost:3000", "https://myapp.com"]
    allow_origins=_origins_list,
    
    # allow_origin_regex: Pattern to match URLs dynamically
    # r"https://.*\.vercel\.app" matches:
    #   - https://myapp.vercel.app
    #   - https://myapp-preview-abc123.vercel.app
    #   - Any subdomain of vercel.app
    # This is useful for preview deployments that have random URLs
    allow_origin_regex=r"https://.*\.vercel\.app",
    
    # allow_methods: Which HTTP methods are allowed
    # GET = fetch data, POST = send data, OPTIONS = browser preflight check
    allow_methods=["GET", "POST", "OPTIONS"],
    
    # allow_headers: Which request headers the browser can send
    # "*" means all headers are allowed (Authorization, Content-Type, etc.)
    allow_headers=["*"],
    
    # expose_headers: Which response headers the browser can read
    # "*" means the browser can access all response headers
    expose_headers=["*"],
)


# ============================================================================
# REGISTER ROUTES - Connect URL endpoints to handler functions
# ============================================================================

# Include the system router (endpoints like /health, /status, etc.)
# All routes in system_router become available at their defined paths
app.include_router(system_router)

# Include the chat router (endpoints like /chat, /history, etc.)
# All routes in chat_router become available at their defined paths
app.include_router(chat_router)


# ============================================================================
# HOW THIS ALL WORKS TOGETHER
# ============================================================================

# 1. Server starts → lifespan() runs → initialize_chat_agent() sets up resources
# 2. App is ready to receive requests
# 3. Browser at localhost:3000 sends POST to localhost:8000/chat
# 4. CORS middleware checks if localhost:3000 is allowed → Yes → allows request
# 5. FastAPI routes request to chat_router → handler processes it → returns response
# 6. Server stops (Ctrl+C) → lifespan() cleanup → shutdown_chat_agent() closes resources