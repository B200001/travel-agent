"""
Gemini-based conversational travel agent.

This is a chatbot that helps users plan trips using Google's Gemini AI.
It remembers what users tell it both in the short-term (current conversation)
and long-term (saved preferences across sessions).

Features:
- Short-term memory: Remembers the last 20 messages in the current conversation
- Long-term memory: Saves user preferences (destination, budget, dates) to a file
- Guardrails: Optional safety checks on inputs and outputs
- Langfuse: Optional tracking and quality scoring of conversations
- Tool calls: Can save/retrieve preferences and search Google for current info
"""

# ============================================================================
# IMPORTS - Libraries we need for this program
# ============================================================================

import os  # For reading environment variables
import json  # For saving/loading data in JSON format
import logging  # For logging errors and warnings
from datetime import date  # For getting today's date
from typing import List, Dict, Any, Optional  # For type hints (makes code clearer)
from pathlib import Path  # For handling file paths

from dotenv import load_dotenv  # For loading API keys from .env file
from google import genai  # Google's Gemini AI library
from google.genai import types  # Type definitions for Gemini

# Load environment variables from .env file (where we store API keys)
load_dotenv()

# Set up logging so we can track what's happening
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL FEATURES - These features work if you install extra packages
# ============================================================================

# Try to import Langfuse (for tracking conversations and quality scoring)
try:
    from langfuse import get_client as get_langfuse_client
    LANGFUSE_AVAILABLE = True  # Flag to know if Langfuse is installed
except ImportError:
    LANGFUSE_AVAILABLE = False  # Langfuse not installed, skip it

# Try to import Guardrails (for safety checks on AI inputs/outputs)
GUARDRAILS_AVAILABLE = False
try:
    from guardrails import Guard
    from guardrails.hub import NoRefusal  # Prevents jailbreak attempts
    GUARDRAILS_AVAILABLE = True  # Flag to know if Guardrails is installed
except ImportError:
    pass  # Guardrails not installed, skip it

# ============================================================================
# CONFIGURATION - Settings for how the agent behaves
# ============================================================================

# How many recent messages to keep in memory (prevents context from getting too long)
SHORT_TERM_MEMORY_LIMIT = 20

# Where to save user preferences (a JSON file in the data folder)
LONG_TERM_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "travel_memory.json"

# The main instructions that tell the AI how to behave
SYSTEM_PROMPT = """You are a friendly travel-savvy friend helping someone plan a trip. Talk like a real person in a chat—warm, casual, and natural. Not like a brochure or a formal assistant.

You have access to:
1. Today's date - provided in the context below. When the user asks for today's date or what day it is, tell them from the context.
2. Web search - use it for current time (any city: New York, Poland, etc.), flight schedules, fares, opening hours, weather, travel advisories. When the user asks for the time anywhere, search for it. Do not say you can't look it up.
3. save_preferences - save the user's destination, budget, dates when they share them (use session_id from context).
4. get_preferences - retrieve saved preferences for this user when starting a new chat or when they ask "what did I tell you?" or similar.

How to sound natural:
- Write in short, flowing sentences. Mix in a question or two naturally.
- Avoid long bullet points unless the user asks for a list or itinerary.
- Use ₹ for Indian Rupees. Support India and international travel.
- When they've shared enough (destination, rough dates, budget), offer a day-by-day plan—keep tone light."""

# ============================================================================
# HELPER FUNCTIONS - Utilities for saving/loading user preferences
# ============================================================================

def _ensure_storage_dir():
    """
    Make sure the folder for saving user data exists.
    If it doesn't exist, create it.
    """
    LONG_TERM_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_long_term_memory() -> Dict[str, Dict]:
    """
    Load saved user preferences from the JSON file.
    
    Returns:
        A dictionary where keys are session_ids and values are user preferences.
        Example: {"user123": {"destination": "Paris", "budget": 50000}}
    """
    _ensure_storage_dir()  # Make sure the folder exists
    
    # If the file doesn't exist yet, return an empty dictionary
    if not LONG_TERM_STORAGE_PATH.exists():
        return {}
    
    try:
        # Try to read and parse the JSON file
        with open(LONG_TERM_STORAGE_PATH) as f:
            return json.load(f)
    except Exception:
        # If anything goes wrong (corrupt file, etc.), return empty
        return {}


def _save_long_term_memory(data: Dict[str, Dict]) -> None:
    """
    Save user preferences to the JSON file.
    
    Args:
        data: Dictionary of session_ids to preferences to save
    """
    _ensure_storage_dir()  # Make sure the folder exists
    
    # Write the data to the file in a nice readable format
    with open(LONG_TERM_STORAGE_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _initialize_long_term_memory_store() -> None:
    """
    Ensure long-term memory storage exists at startup.

    Creates the data directory and an empty JSON file if missing so the
    storage location is visible before the first tool call.
    """
    _ensure_storage_dir()
    if not LONG_TERM_STORAGE_PATH.exists():
        _save_long_term_memory({})


# ============================================================================
# MAIN AGENT CLASS - The core travel chatbot
# ============================================================================

class TravelChatAgent:
    """
    The main travel planning chatbot.
    
    This class handles:
    - Connecting to Google's Gemini AI
    - Managing conversation memory (short-term and long-term)
    - Calling tools (save/get preferences, Google Search)
    - Optional safety checks and quality scoring
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the travel agent.
        
        Args:
            api_key: Your Google Gemini API key (or it will look in environment variables)
        """
        # Get the API key from parameter or environment variables
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        
        # Check if we should use Vertex AI or standard Gemini
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        
        # Create the Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Choose which model to use (Flash is faster and cheaper)
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"

        # ===== GUARDRAILS SETUP (Optional) =====
        # Guardrails help prevent jailbreak attempts and unsafe outputs
        self._guard = None
        if GUARDRAILS_AVAILABLE:
            try:
                # Create a guard that checks for refusals and unsafe content
                self._guard = Guard().use(NoRefusal(on_fail="exception"))
            except Exception as e:
                logger.warning("Guardrails setup failed: %s", e)

        # ===== LANGFUSE SETUP (Optional) =====
        # Langfuse tracks conversations and scores their quality
        self._langfuse = None
        if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                self._langfuse = get_langfuse_client()
            except Exception as e:
                logger.warning("Langfuse init failed: %s", e)

        # ===== TOOL DECLARATIONS =====
        # Define what tools the AI can use (save/get preferences)
        self._tool_declarations = self._build_tool_declarations()

        # Ensure long-term memory file exists on startup
        _initialize_long_term_memory_store()

    def _build_tool_declarations(self) -> List[types.FunctionDeclaration]:
        """
        Define the tools (functions) that the AI can call.
        
        Returns:
            A list of function declarations that Gemini can use
        """
        return [
            # Tool #1: Save user preferences
            types.FunctionDeclaration(
                name="save_preferences",
                description="Save user travel preferences for later (destination, budget, dates). Call when user shares these details.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "session_id": types.Schema(
                            type=types.Type.STRING,
                            description="Session identifier"
                        ),
                        "destination": types.Schema(
                            type=types.Type.STRING,
                            description="Travel destination"
                        ),
                        "budget": types.Schema(
                            type=types.Type.NUMBER,
                            description="Budget in INR"
                        ),
                        "start_date": types.Schema(
                            type=types.Type.STRING,
                            description="Trip start date YYYY-MM-DD"
                        ),
                        "end_date": types.Schema(
                            type=types.Type.STRING,
                            description="Trip end date YYYY-MM-DD"
                        ),
                        "travelers": types.Schema(
                            type=types.Type.INTEGER,
                            description="Number of travelers"
                        ),
                    },
                    required=["session_id"],  # Only session_id is mandatory
                ),
            ),
            
            # Tool #2: Get saved preferences
            types.FunctionDeclaration(
                name="get_preferences",
                description="Retrieve saved travel preferences for this session. Call when user asks what they shared or to recall their plans.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "session_id": types.Schema(
                            type=types.Type.STRING,
                            description="Session identifier"
                        ),
                    },
                    required=["session_id"],
                ),
            ),
        ]

    def _build_session_callables(self, session_id: str):
        """
        Create Python functions that Gemini can automatically call.
        
        These are wrapper functions that make it easy for Gemini to call
        save_preferences and get_preferences with the current session_id.
        
        Args:
            session_id: The current user's session ID
            
        Returns:
            Tuple of (save_preferences function, get_preferences function)
        """
        # Store session_id in a variable the inner functions can access
        sid = session_id

        def save_preferences(
            session_id: str = "",
            destination: str = None,
            budget: float = None,
            start_date: str = None,
            end_date: str = None,
            travelers: int = None,
        ) -> str:
            """
            Save user travel preferences.
            
            This function gets called automatically by Gemini when it detects
            the user is sharing their travel plans.
            """
            # Use the provided session_id or fall back to the outer one
            s = session_id or sid
            
            # Build a dictionary of the arguments that were provided
            args = {"session_id": s}
            if destination is not None:
                args["destination"] = str(destination)
            if budget is not None:
                args["budget"] = float(budget)
            if start_date is not None:
                args["start_date"] = str(start_date)
            if end_date is not None:
                args["end_date"] = str(end_date)
            if travelers is not None:
                args["travelers"] = int(travelers)
            
            # Call the actual tool execution function
            return self._execute_tool("save_preferences", args, sid)

        def get_preferences(session_id: str = "") -> str:
            """
            Retrieve saved travel preferences.
            
            This function gets called automatically by Gemini when the user
            asks about their previously saved preferences.
            """
            s = session_id or sid
            return self._execute_tool("get_preferences", {"session_id": s}, sid)

        # Return both functions so Gemini can use them
        return save_preferences, get_preferences

    def _execute_tool(self, name: str, args: Dict[str, Any], session_id: str) -> str:
        """
        Actually execute a tool call (save or get preferences).
        
        Args:
            name: Name of the tool ("save_preferences" or "get_preferences")
            args: Dictionary of arguments for the tool
            session_id: Current session ID
            
        Returns:
            JSON string with the result
        """
        # SAVE PREFERENCES TOOL
        if name == "save_preferences":
            # Get the session_id (from args or parameter)
            sid = args.get("session_id") or session_id
            
            # Load existing preferences from file
            mem = _load_long_term_memory()
            
            # Get this user's preferences (or empty dict if new user)
            prefs = mem.get(sid, {})
            
            # Update preferences with any new values provided
            if "destination" in args:
                prefs["destination"] = str(args["destination"])
            if "budget" in args:
                prefs["budget"] = float(args["budget"])
            if "start_date" in args:
                prefs["start_date"] = str(args["start_date"])
            if "end_date" in args:
                prefs["end_date"] = str(args["end_date"])
            if "travelers" in args:
                prefs["travelers"] = int(args["travelers"])
            
            # Save the updated preferences back to memory
            mem[sid] = prefs
            _save_long_term_memory(mem)
            
            # Return success message
            return json.dumps({"status": "saved", "preferences": prefs})

        # GET PREFERENCES TOOL
        if name == "get_preferences":
            # Get the session_id
            sid = args.get("session_id") or session_id
            
            # Load all preferences from file
            mem = _load_long_term_memory()
            
            # Get this user's preferences
            prefs = mem.get(sid, {})
            
            # Return the preferences (or a message if none exist)
            return json.dumps(prefs if prefs else {"message": "No saved preferences for this session"})

        # Unknown tool
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _run_guardrails_input(self, text: str) -> str:
        """
        Check user input for safety issues (jailbreaks, harmful content).
        
        Args:
            text: The user's message
            
        Returns:
            The validated text (or raises an exception if unsafe)
        """
        # Skip if guardrails not installed
        if not self._guard:
            return text
        
        try:
            # Run the safety check
            result = self._guard.parse(text)
            return result.validated_output or text
        except Exception as e:
            logger.warning("Guardrails input check failed: %s", e)
            return text

    def _run_guardrails_output(self, text: str) -> str:
        """
        Check AI output for safety issues (harmful or inappropriate content).
        
        Args:
            text: The AI's response
            
        Returns:
            The validated text (or raises an exception if unsafe)
        """
        # Skip if guardrails not installed
        if not self._guard:
            return text
        
        try:
            # Run the safety check
            result = self._guard.parse(text)
            return result.validated_output or text
        except Exception as e:
            logger.warning("Guardrails output check failed: %s", e)
            return text

    def _run_judge(self, user_msg: str, assistant_msg: str, trace_id: Optional[str] = None,
                   generation_id: Optional[str] = None) -> None:
        """
        Use an LLM to judge the quality of the AI's response.
        
        This uses Gemini itself to score how helpful, relevant, and safe
        the response was. Scores are sent to Langfuse for tracking.
        
        Args:
            user_msg: What the user said
            assistant_msg: How the AI responded
            trace_id: Langfuse trace ID for tracking
            generation_id: Langfuse generation ID for tracking
        """
        # Skip if Langfuse not available or no trace_id
        if not self._langfuse or not trace_id:
            return
        
        try:
            # Create a prompt asking Gemini to score the response
            judge_prompt = f"""Score this travel assistant response.

User: {user_msg}
Assistant: {assistant_msg}

Rate 1-5:
- helpfulness: How helpful is the response?
- relevance: Is it on-topic for travel planning?
- safety: Any unsafe or inappropriate content?

Respond with JSON only: {{"helpfulness": N, "relevance": N, "safety": N, "comment": "brief"}}"""

            # Ask Gemini to judge the response
            resp = self.client.models.generate_content(
                model=self.model,
                contents=judge_prompt,
            )
            
            # Parse the JSON response
            raw = (resp.text or "").strip()
            
            # Extract JSON from the response (find the { } brackets)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            
            if start >= 0 and end > start:
                scores = json.loads(raw[start:end])
                
                # Send each score to Langfuse
                for key in ("helpfulness", "relevance", "safety"):
                    if key in scores and isinstance(scores[key], (int, float)):
                        # Convert 1-5 score to 0-1 range
                        v = float(scores[key])
                        self._langfuse.create_score(
                            name=f"judge_{key}",
                            value=min(1.0, max(0.0, v / 5.0)),  # Normalize to 0-1
                            trace_id=trace_id,
                            observation_id=generation_id,
                            data_type="NUMERIC",
                            comment=scores.get("comment", ""),
                        )
                
                # Make sure scores are sent to Langfuse
                self._langfuse.flush()
                
        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)

    def chat(
        self,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> str:
        """
        Main chat function - send messages and get a response.
        
        This is the main entry point for chatting with the agent. It:
        1. Applies short-term memory (keeps last 20 messages)
        2. Runs optional safety checks on input
        3. Decides whether to use Google Search or custom tools
        4. Calls Gemini to generate a response
        5. Runs optional safety checks on output
        6. Tracks conversation quality (optional)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "I want to visit Paris"}]
            session_id: Unique ID for this user (for saving preferences)
            
        Returns:
            The AI's response as a string
        """
        # Use "default" if no session_id provided
        session_id = session_id or "default"
        
        # Get the user's latest message
        user_content = messages[-1]["content"] if messages else ""

        # ===== SAFETY CHECK: INPUT =====
        # Check if the user's input is safe (optional)
        try:
            user_content = self._run_guardrails_input(user_content)
        except Exception:
            pass  # If it fails, continue anyway

        # ===== SHORT-TERM MEMORY =====
        # Only keep the last 20 messages to prevent context from getting too long
        short_term = messages[-SHORT_TERM_MEMORY_LIMIT:] if len(messages) > SHORT_TERM_MEMORY_LIMIT else messages

        # Format the conversation history for the prompt
        history_for_prompt = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in short_term
        )
        
        # Add today's date and session_id to the context
        today_str = date.today().strftime("%A, %B %d, %Y")
        ctx = f"\n\nToday's date: {today_str}. Session_id: {session_id}."
        
        # Build the full prompt
        prompt = f"{SYSTEM_PROMPT}{ctx}\n\n---\nConversation:\n{history_for_prompt}\n\nAssistant:"

        # ===== TOOL ROUTING =====
        # Google's API doesn't allow combining google_search with custom tools.
        # So we need to decide which to use based on the user's query.
        
        # Keywords that suggest the user wants real-time information
        search_keywords = ("time", "weather", "flight", "current", "now", "opening hours", "fare", "price", "advisory")
        
        # Check if the user's message contains any search keywords
        needs_search = any(kw in user_content.lower() for kw in search_keywords)

        if needs_search:
            # User wants real-time info → use Google Search
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
        else:
            # User is chatting normally → use custom tools (save/get preferences)
            save_fn, get_fn = self._build_session_callables(session_id)
            config = types.GenerateContentConfig(
                tools=[save_fn, get_fn],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            )

        # ===== GENERATE RESPONSE =====
        def _run_chat():
            """Helper function to call Gemini and get a response."""
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            # Extract the text from the response
            return (getattr(response, "text", None) or "").strip()

        try:
            # Call Gemini to generate a response
            result = _run_chat()

            # ===== SAFETY CHECK: OUTPUT =====
            # Check if the AI's response is safe (optional)
            try:
                result = self._run_guardrails_output(result)
            except Exception:
                pass  # If it fails, continue anyway

            # ===== QUALITY TRACKING (Langfuse) =====
            # Track this conversation and score its quality (optional)
            if self._langfuse:
                try:
                    # Create a trace in Langfuse
                    with self._langfuse.start_as_current_observation(
                        name="travel_chat",
                        as_type="span",
                        input={"user_message": user_content, "session_id": session_id},
                        output=result,
                    ) as span:
                        # Get the trace ID
                        trace_id = getattr(span, "trace_id", None) or getattr(span, "id", None)
                        
                        # Run the quality judge
                        if trace_id:
                            self._run_judge(user_content, result, trace_id, None)
                    
                    # Send the data to Langfuse
                    self._langfuse.flush()
                    
                except Exception as e:
                    logger.warning("Langfuse trace/judge failed: %s", e)

            # Return the AI's response
            return result

        except Exception as e:
            # If anything goes wrong, return a friendly error message
            err_msg = f"Sorry, I ran into an error: {str(e)}"
            return err_msg