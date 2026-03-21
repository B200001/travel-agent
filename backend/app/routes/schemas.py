"""Request/response schemas for route handlers.

This file defines the data structures (schemas) that the API expects
to receive from clients and send back to them. Think of these as
"contracts" or "blueprints" for the data format.

Pydantic models provide:
- Automatic validation (rejects invalid data)
- Type checking (ensures correct data types)
- Auto-generated documentation (shows up in /docs)
- Conversion between JSON and Python objects
"""

# ============================================================================
# IMPORTS
# ============================================================================

# List and Optional are type hints:
# - List[ChatMessage] means "a list containing ChatMessage objects"
# - Optional[str] means "either a string or None"
from typing import List, Optional

# BaseModel is the foundation of all Pydantic models
# When you inherit from BaseModel, you get automatic validation and serialization
from pydantic import BaseModel


# ============================================================================
# CHATMESSAGE MODEL - Represents a single message in the conversation
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in the chat conversation.
    
    This represents one message from either the user or the assistant.
    Each message has two parts: who said it (role) and what they said (content).
    
    Attributes:
        role: Who sent this message
              Valid values: "user" (the person chatting) or "assistant" (the AI)
              Example: "user"
        
        content: The actual text of the message
                 This is what was said in the conversation
                 Example: "Plan a 3-day trip to Tokyo"
    
    Example JSON:
        {
            "role": "user",
            "content": "What's the best time to visit Paris?"
        }
    
    Example Python usage:
        # Creating a message
        msg = ChatMessage(role="user", content="Hello!")
        
        # Accessing fields
        print(msg.role)     # Output: "user"
        print(msg.content)  # Output: "Hello!"
        
        # Converting to dictionary (for JSON serialization)
        msg.model_dump()  # Output: {"role": "user", "content": "Hello!"}
    """
    
    # The role field: must be a string
    # In a real app, you might use an Enum to restrict values to "user" | "assistant"
    # But for flexibility, this accepts any string
    role: str  # "user" | "assistant"
    
    # The content field: must be a string
    # This is the actual message text
    content: str


# ============================================================================
# TRAVELCHATREQUEST MODEL - The complete request sent to the API
# ============================================================================

class TravelChatRequest(BaseModel):
    """The request payload for travel chat endpoints.
    
    This is what clients send when they make a POST request to
    /api/travel-chat or /api/travel-chat/stream.
    
    Attributes:
        messages: The conversation history up to this point
                  This includes all previous messages from both user and assistant
                  The list is ordered chronologically (oldest first, newest last)
                  Type: List[ChatMessage] means a list of ChatMessage objects
        
        session_id: A unique identifier for this conversation thread
                    Optional (can be None if not provided)
                    Used to:
                    - Load previous conversation history from database
                    - Save new messages to the right conversation
                    - Keep multiple conversations separate
                    If None, creates a new conversation with no history
                    Type: Optional[str] means it can be a string or None
                    Default: None (if not provided in request)
    
    Example JSON request body:
        {
            "messages": [
                {"role": "user", "content": "I want to visit Japan"},
                {"role": "assistant", "content": "Great choice! When are you planning to go?"},
                {"role": "user", "content": "In spring, for the cherry blossoms"}
            ],
            "session_id": "user-123-conversation-456"
        }
    
    Example without session_id (new conversation):
        {
            "messages": [
                {"role": "user", "content": "Plan a trip to Paris"}
            ]
        }
    
    How Pydantic validates this:
        1. Checks that 'messages' exists and is a list
        2. Checks that each item in the list is a valid ChatMessage
        3. Checks that session_id is a string or None (missing is okay)
        4. If validation fails, returns detailed error to client
    
    Example Python usage in route handler:
        @router.post("/travel-chat")
        async def travel_chat(request: TravelChatRequest):
            # FastAPI automatically:
            # 1. Parses the JSON request body
            # 2. Validates it against TravelChatRequest schema
            # 3. Creates a TravelChatRequest object
            # 4. Passes it to this function
            
            # Access the fields
            print(request.messages)    # List of ChatMessage objects
            print(request.session_id)  # String or None
            
            # Convert messages to plain dicts for processing
            messages_dicts = [m.model_dump() for m in request.messages]
    """
    
    # messages field: A list of ChatMessage objects
    # Required field (no default value, must be provided)
    # FastAPI will reject requests that don't include this
    messages: List[ChatMessage]
    
    # session_id field: A string identifier for the conversation
    # Optional field (has default value of None)
    # If client doesn't send session_id, it defaults to None
    # = None means "this parameter is optional and defaults to None"
    session_id: Optional[str] = None  # For short/long-term memory


# ============================================================================
# WHY USE PYDANTIC MODELS?
# ============================================================================

# WITHOUT PYDANTIC (manual validation):
# --------------------------------------
# @router.post("/travel-chat")
# async def travel_chat(request: dict):
#     # Manual validation - tedious and error-prone!
#     if "messages" not in request:
#         raise HTTPException(400, "messages is required")
#     if not isinstance(request["messages"], list):
#         raise HTTPException(400, "messages must be a list")
#     for msg in request["messages"]:
#         if "role" not in msg or "content" not in msg:
#             raise HTTPException(400, "Each message needs role and content")
#         if not isinstance(msg["role"], str):
#             raise HTTPException(400, "role must be a string")
#     # ... more validation ...

# WITH PYDANTIC (automatic validation):
# -------------------------------------
# @router.post("/travel-chat")
# async def travel_chat(request: TravelChatRequest):
#     # Validation already done! If we get here, data is valid.
#     # Just use it directly:
#     messages = request.messages
#     session_id = request.session_id


# ============================================================================
# EXAMPLE API REQUEST/RESPONSE FLOW
# ============================================================================

# CLIENT SENDS (JSON):
# {
#     "messages": [
#         {"role": "user", "content": "Plan a trip to Tokyo"}
#     ],
#     "session_id": "abc123"
# }
#
# ↓
#
# FASTAPI RECEIVES AND VALIDATES:
# 1. Parses JSON string into Python dict
# 2. Creates TravelChatRequest object
# 3. Validates all fields according to schema
# 4. If valid → passes to route handler
# 5. If invalid → returns 422 error with details
#
# ↓
#
# ROUTE HANDLER PROCESSES:
# async def travel_chat(request: TravelChatRequest):
#     request.messages[0].role     # "user"
#     request.messages[0].content  # "Plan a trip to Tokyo"
#     request.session_id           # "abc123"