"""Travel chat HTTP routes.

This file defines the API endpoints (URLs) that clients can call
to interact with the travel chat system. It has two endpoints:
1. /api/travel-chat - Returns a complete response at once
2. /api/travel-chat/stream - Returns responses as a stream (word-by-word)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# __future__ annotations allow using modern type hints in older Python versions
from __future__ import annotations

import asyncio
# json library for converting Python objects to JSON strings
import json

# FastAPI components:
# - APIRouter: Groups related endpoints together
# - HTTPException: Used to return error responses (404, 500, etc.)
from fastapi import APIRouter, HTTPException

# StreamingResponse: Sends data to client piece-by-piece instead of all at once
from fastapi.responses import StreamingResponse

# TravelChatRequest: Defines the structure of incoming request data
# This is a Pydantic model that validates the JSON sent by the client
from app.routes.schemas import TravelChatRequest

# Service functions that contain the actual chat logic:
# - build_chat_payload: Gets a complete response all at once
# - build_chat_stream: Gets response as a stream of chunks
from app.services.chat_service import build_chat_payload, build_chat_stream


# ============================================================================
# CREATE ROUTER - Groups all chat-related endpoints
# ============================================================================

# Create a router with common settings for all routes in this file
router = APIRouter(
    prefix="/api",              # All routes start with /api (e.g., /api/travel-chat)
    tags=["travel-chat"]        # Groups endpoints in auto-generated documentation
)


# ============================================================================
# ENDPOINT 1: Non-Streaming Chat (Complete Response at Once)
# ============================================================================

@router.post("/travel-chat")  # This creates a POST endpoint at /api/travel-chat
async def travel_chat(request: TravelChatRequest):
    """Handle a travel chat request and return a complete response.
    
    This endpoint processes the entire conversation and returns the full
    response in one go. Good for simple requests, but slower for long responses.
    
    Args:
        request: TravelChatRequest object containing:
                 - messages: List of conversation history (user/assistant messages)
                 - session_id: Unique ID to track this conversation thread
    
    Returns:
        dict: The complete chat response with the assistant's message
    
    Raises:
        HTTPException 503: If there's a configuration/availability error
        HTTPException 500: If there's any other unexpected error
    
    Example request body:
        {
            "messages": [
                {"role": "user", "content": "Plan a trip to Paris"}
            ],
            "session_id": "user123-session456"
        }
    """
    try:
        # Convert Pydantic message objects to plain Python dictionaries
        # request.messages is a list of Pydantic models
        # .model_dump() converts each model to a dict: {"role": "user", "content": "..."}
        # List comprehension iterates over all messages and converts each one
        messages = [m.model_dump() for m in request.messages]
        
        # Call the service function to process the chat request
        # This function:
        # 1. Loads conversation history using session_id
        # 2. Runs the LangGraph workflow with the messages
        # 3. Returns the complete response
        return build_chat_payload(messages, session_id=request.session_id)
        
    except ValueError as exc:
        # ValueError = expected error (missing config, invalid input, etc.)
        # 503 Service Unavailable = server is temporarily unable to handle the request
        # 'from exc' preserves the original error traceback for debugging
        raise HTTPException(status_code=503, detail=str(exc)) from exc
        
    except Exception as exc:
        # Catch any other unexpected errors
        # 500 Internal Server Error = something went wrong on the server
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================================
# ENDPOINT 2: Streaming Chat (Response Arrives in Chunks)
# ============================================================================

@router.post("/travel-chat/stream")  # POST endpoint at /api/travel-chat/stream
async def travel_chat_stream(request: TravelChatRequest):
    """Handle a travel chat request and stream the response word-by-word.
    
    This endpoint uses Server-Sent Events (SSE) to stream the response
    as it's being generated. Great for showing typing indicators and
    progressive responses like ChatGPT does.
    
    Args:
        request: TravelChatRequest object (same as non-streaming endpoint)
    
    Returns:
        StreamingResponse: A stream of Server-Sent Events (SSE)
                          Each event contains a chunk of the response
    
    How streaming works:
        1. Client connects to this endpoint
        2. Connection stays open (doesn't close after first response)
        3. Server sends data chunks as they become available
        4. Client displays each chunk as it arrives (progressive rendering)
        5. Connection closes when response is complete
    """
    
    # Define an async generator function that yields response chunks
    # Generators produce values one at a time instead of all at once
    # 'async def' allows it to work with async/await code
    async def event_generator():
        """Generate Server-Sent Events for streaming response.
        
        This function yields data in SSE format:
            data: {"chunk": "Hello"}\n\n
            data: {"chunk": " world"}\n\n
            data: {"chunk": "!"}\n\n
        
        Each 'yield' sends one chunk to the client immediately.
        """
        try:
            # Convert Pydantic message models to plain dictionaries
            # Same as in the non-streaming endpoint
            messages = [m.model_dump() for m in request.messages]
            
            # build_chat_stream() is a generator that yields response chunks
            # Each iteration of this loop processes one chunk as it arrives
            for event in build_chat_stream(messages, session_id=request.session_id):
                # Format the event in SSE format:
                # - "data: " prefix is required by SSE protocol
                # - json.dumps() converts Python dict to JSON string
                # - ensure_ascii=False allows non-English characters (中文, العربية, etc.)
                # - "\n\n" marks the end of this event (SSE requires double newline)
                #
                # Example output: data: {"type": "chunk", "content": "Hello"}\n\n
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # Yield control so Uvicorn/ASGI can flush each chunk promptly.
                await asyncio.sleep(0)
                
        except ValueError as exc:
            # If there's a configuration error, send it as an error event
            # Client can detect this and show an error message
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            
        except Exception as exc:
            # Catch any unexpected errors and send as error event
            # Better than crashing - client can handle the error gracefully
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    # Return a StreamingResponse that wraps our generator
    # This tells FastAPI to:
    # 1. Keep the connection open
    # 2. Send each yielded value immediately (don't buffer)
    # 3. Close connection when generator is exhausted
    return StreamingResponse(
        event_generator(),           # The generator function we defined above
        media_type="text/event-stream",  # SSE MIME type (tells browser how to parse)
        headers={
            # no-transform + X-Accel-Buffering help prevent intermediate buffering.
            "Cache-Control": "no-cache, no-transform",
            
            # Connection: keep-alive - Keep the HTTP connection open for streaming
            # Without this, the connection might close after the first chunk
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# HOW THESE ENDPOINTS ARE USED
# ============================================================================

# NON-STREAMING ENDPOINT (/api/travel-chat):
# ----------------------------------------
# Client: POST /api/travel-chat with {"messages": [...], "session_id": "..."}
# Server: Processes entire request... waits... returns complete response
# Client: Receives full response at once, displays it
# Use case: Simple requests, file uploads, when you want the full answer first

# STREAMING ENDPOINT (/api/travel-chat/stream):
# -------------------------------------------
# Client: POST /api/travel-chat/stream with {"messages": [...], "session_id": "..."}
# Server: Sends "data: {chunk1}\n\n" immediately
# Client: Displays "Hello"
# Server: Sends "data: {chunk2}\n\n"
# Client: Displays "Hello world"
# Server: Sends "data: {chunk3}\n\n"
# Client: Displays "Hello world!"
# Server: Closes connection
# Use case: Long responses, better UX (like ChatGPT), real-time feeling