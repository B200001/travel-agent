"""System and health routes.

This file defines simple utility endpoints for checking if the API
is running and working properly. These endpoints don't do any complex
processing - they just return basic status information.

Common uses:
- Health checks: Monitoring systems ping /api/health to ensure the server is up
- Load balancers: Check if server is healthy before sending traffic to it
- Deployment: Verify the app started successfully after deployment
- Welcome page: A friendly message at the root URL
"""

# ============================================================================
# IMPORTS
# ============================================================================

# APIRouter: Groups related endpoints together (similar to what we saw in chat.py)
from fastapi import APIRouter


# ============================================================================
# CREATE ROUTER - Groups all system-related endpoints
# ============================================================================

# Create a router for system/utility endpoints
# tags=["system"] groups these endpoints together in the auto-generated docs
# No prefix is set, so routes use their exact paths (/ and /api/health)
router = APIRouter(tags=["system"])


# ============================================================================
# ENDPOINT 1: Root/Home Endpoint
# ============================================================================

@router.get("/")  # Creates a GET endpoint at the root path "/"
def home():
    """Return a welcome message at the API root.
    
    This is the simplest possible endpoint - just returns a static message.
    It's what users see if they visit the API base URL in their browser.
    
    HTTP Method: GET (for retrieving information, not modifying anything)
    
    URL: http://localhost:8000/
    
    Returns:
        dict: A simple dictionary with a welcome message
              FastAPI automatically converts this to JSON
    
    Response Example:
        {
            "message": "Welcome to the Travel Chat API"
        }
    
    Why this endpoint exists:
    - Friendly message for people who navigate to the root URL
    - Quick test that the server is running
    - Alternative to showing a 404 error at /
    
    Note: This is a regular function (not async) because it doesn't do
          any I/O operations (database, file system, network calls).
          For simple synchronous operations, regular functions work fine.
    """
    # Return a dictionary that FastAPI will automatically convert to JSON
    return {"message": "Welcome to the Travel Chat API"}


# ============================================================================
# ENDPOINT 2: Health Check Endpoint
# ============================================================================

@router.get("/api/health")  # Creates a GET endpoint at /api/health
async def health():
    """Health check endpoint for monitoring and deployment systems.
    
    This endpoint is used by:
    - Docker health checks: Ensures container is working
    - Kubernetes liveness probes: Restarts pod if this fails
    - Load balancers: Only sends traffic to healthy servers
    - Monitoring tools: Tracks uptime and availability
    - CI/CD pipelines: Verifies deployment was successful
    
    HTTP Method: GET (read-only, doesn't change anything)
    
    URL: http://localhost:8000/api/health
    
    Returns:
        dict: A simple status indicator showing the API is operational
              FastAPI automatically converts this to JSON
    
    Response Example:
        {
            "status": "ok"
        }
    
    HTTP Status Code: 200 OK (automatically set by FastAPI for successful responses)
    
    How health checks work:
    1. Monitoring system sends: GET /api/health
    2. Server responds: {"status": "ok"} with 200 status code
    3. Monitoring system sees 200 → server is healthy ✓
    4. If server is down/crashed → no response or error → server is unhealthy ✗
    
    Advanced health checks (not implemented here):
    - Check database connection
    - Check external API availability
    - Check disk space
    - Check memory usage
    - Return different status codes based on severity
    
    Example advanced implementation:
        @router.get("/api/health")
        async def health():
            try:
                # Test database connection
                await db.execute("SELECT 1")
                
                # Test API dependency
                await external_api.ping()
                
                return {
                    "status": "healthy",
                    "database": "connected",
                    "external_api": "available"
                }
            except DatabaseError:
                # Return 503 Service Unavailable if critical component is down
                raise HTTPException(status_code=503, detail="Database unreachable")
    
    Note: This is an async function even though it doesn't await anything.
          This is fine - FastAPI handles both sync and async functions.
          Using async doesn't hurt, and makes it easy to add async checks later.
    """
    # Return a simple status dictionary
    # In production, you might check database connectivity, etc.
    return {"status": "ok"}


# ============================================================================
# HOW THESE ENDPOINTS ARE USED IN PRACTICE
# ============================================================================

# DOCKER HEALTH CHECK (in Dockerfile):
# ------------------------------------
# HEALTHCHECK --interval=30s --timeout=3s \
#   CMD curl -f http://localhost:8000/api/health || exit 1
#
# Docker pings /api/health every 30 seconds. If it fails, container is marked unhealthy.

# KUBERNETES LIVENESS PROBE (in deployment.yaml):
# ----------------------------------------------
# livenessProbe:
#   httpGet:
#     path: /api/health
#     port: 8000
#   initialDelaySeconds: 10
#   periodSeconds: 30
#
# Kubernetes checks /api/health every 30 seconds. If it fails, pod is restarted.

# MONITORING SCRIPT:
# -----------------
# import requests
# 
# response = requests.get("http://api.example.com/api/health")
# if response.status_code == 200 and response.json()["status"] == "ok":
#     print("✓ API is healthy")
# else:
#     print("✗ API is down! Alert the team!")
#     send_alert_to_slack()

# LOAD BALANCER HEALTH CHECK:
# ---------------------------
# AWS Application Load Balancer settings:
# - Health check path: /api/health
# - Healthy threshold: 2 consecutive successes
# - Unhealthy threshold: 3 consecutive failures
# - Interval: 30 seconds
#
# If health check fails 3 times in a row, traffic stops routing to that server.

# DEPLOYMENT VERIFICATION:
# -----------------------
# After deploying new code:
# 1. Deploy new version
# 2. Wait 10 seconds
# 3. curl http://new-server/api/health
# 4. If returns {"status": "ok"} → deployment successful ✓
# 5. If error → rollback deployment ✗


# ============================================================================
# COMPARISON: SYNC vs ASYNC FUNCTIONS
# ============================================================================

# SYNCHRONOUS (def):
# def home():
#     # Blocks the thread while executing
#     # Fine for simple operations with no I/O
#     return {"message": "Welcome"}

# ASYNCHRONOUS (async def):
# async def health():
#     # Doesn't block - can handle other requests while waiting
#     # Useful when doing I/O operations:
#     # await db.query()
#     # await external_api.call()
#     return {"status": "ok"}

# FastAPI handles both types automatically!
# Use 'async def' when you need to await something
# Use 'def' for simple synchronous operations
# When in doubt, 'async def' works fine for both cases