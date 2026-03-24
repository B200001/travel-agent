"""Autonomous travel chat agent with planner and task graph execution.

This is the core brain of the travel chatbot. It uses LangGraph to create
a decision-making workflow that routes queries through different paths:
- Emergency help → fast response with official contacts
- Quick searches → fast Google search
- Complex planning → task graph execution with synthesis

The agent uses Gemini AI, Google Search, memory storage, and caching
to provide intelligent, context-aware travel assistance.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json        # For parsing and creating JSON data
import logging     # For logging debug/info messages
import os          # For accessing environment variables
import re          # For pattern matching (regex) in text
from contextlib import contextmanager, nullcontext  # Lightweight context helpers
from datetime import date  # For getting today's date
from typing import Any, Dict, Iterator, List, Optional  # Type hints for better code clarity

# Load environment variables from .env file (API keys, settings, etc.)
from dotenv import load_dotenv

# Google Generative AI client - connects to Gemini models
from google import genai
from google.genai import types

# LangGraph - creates state machine workflows with nodes and edges
# END = special marker that terminates the graph execution
from langgraph.graph import END, StateGraph

# Custom modules we built:
from app.storage.checkpointing import SqliteCheckpointerManager  # Saves conversation state
from app.storage.memory import initialize_long_term_memory_store  # User preferences storage
from app.storage.query_cache import TravelQueryCache  # Caches responses to avoid re-computing

# Configuration constants (file paths, prompts, limits)
from app.services.constants import (
    CACHE_DB_PATH,
    LANGGRAPH_CHECKPOINT_DB_PATH,
    SHORT_TERM_MEMORY_LIMIT,
    SYSTEM_PROMPT,
    TASK_GRAPH_MERMAID_PATH,
)

# External integrations (Langfuse for observability, guardrails for safety)
from app.services.integrations import (
    langfuse_session_scope,
    normalize_langfuse_session_id,
    run_guardrails,
    run_judge,
    setup_guard,
    setup_langfuse,
    setup_langfuse_gemini_instrumentation,
)

# State definition - the data structure that flows through the graph
from app.services.state import TravelChatState

# Tool execution functions (search, memory save/get)
from app.services.tooling import build_session_callables, execute_tool

# Load environment variables at module level (runs when this file is imported)
load_dotenv()

# Create a logger for this module to track what's happening
logger = logging.getLogger(__name__)


# ============================================================================
# KEYWORD COLLECTIONS - Used for routing and safety checks
# ============================================================================

# Keywords that indicate a travel-related query
# If user mentions any of these, we know it's about travel
TRAVEL_KEYWORDS = (
    "travel", "trip", "flight", "hotel", "itinerary", 
    "visa", "budget", "destination", "weather"
)

# Dangerous phrases that should be blocked immediately
# These indicate harmful intent - refuse to help with these
BLOCKED_INPUT = (
    "make a bomb", "how to hack", "steal", "kill", "suicide"
)

# Responses that should NOT be cached
# Generic safety messages aren't useful to cache since they're not informative
NON_CACHEABLE_RESPONSES = {
    "I can only provide safe travel planning help.",
}

# Attempts to manipulate the system prompt
# Block these "jailbreak" attempts where users try to override instructions
PROMPT_BYPASS = (
    "ignore all previous instructions", 
    "reveal system prompt", 
    "developer instructions"
)

# Keywords indicating someone needs urgent help
# Route these to emergency fast path for immediate assistance
EMERGENCY_KEYWORDS = (
    "emergency", "police", "ambulance", "hospital", "fire brigade",
    "helpline", "contact number", "phone number",
)

# Keywords indicating a quick factual lookup
# These queries benefit from fast Google Search instead of full planning
FAST_SEARCH_KEYWORDS = (
    "time", "weather", "price", "fare", "contact", "phone", "number",
    "where is", "nearest", "open now", "latest", "today",
    "advisory", "visa requirements",
)

# Keywords indicating complex planning needed
# These require the full planner → task executor → synthesizer pipeline
SLOW_PLANNER_KEYWORDS = (
    "itinerary", "day wise", "day-by-day", "full plan",
    "detailed plan", "budget breakdown",
)

# Simple greetings and casual conversation
# Handle these with a quick canned response instead of LLM call
CASUAL_CHAT = (
    "hi", "hello", "hey", "yo", "good morning", "good afternoon",
    "good evening", "how are you", "thanks", "thank you",
)


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class TravelChatAgent:
    """Planner-first autonomous agent.
    
    This class orchestrates the entire travel chat workflow:
    1. Receives user messages
    2. Routes through appropriate processing path
    3. Executes tasks (search, memory operations, reasoning)
    4. Synthesizes final response
    5. Caches results for efficiency
    
    The agent uses a LangGraph state machine with these nodes:
    - prepare: Initial setup and context gathering
    - input_guardrails: Safety checks and routing decisions
    - cache_lookup: Check if we've answered this before
    - emergency_fastpath: Urgent help with official contacts
    - search_fastpath: Quick Google search for simple queries
    - planner: Creates task graph for complex queries
    - task_executor: Runs tasks in dependency order
    - synthesizer: Combines task outputs into final answer
    - postprocess: Safety checks, caching, observability
    """

    def __init__(self, api_key: str = None):
        """Initialize the travel chat agent with all dependencies.
        
        Sets up:
        - Gemini AI client for LLM calls
        - Safety guardrails
        - Conversation checkpointing (save/restore state)
        - Response caching
        - Long-term memory storage
        - LangGraph workflow
        
        Args:
            api_key: Gemini API key (falls back to env vars if not provided)
        
        Raises:
            ValueError: If no API key found in params or environment
        """
        # Get API key from parameter, or fall back to environment variables
        # Try GEMINI_API_KEY first, then GOOGLE_API_KEY.
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")

        # Create the Gemini client.
        self.client = genai.Client(api_key=api_key)

        # Check if Vertex mode is enabled and pick the right default model.
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"

        self._verbose_logs = os.getenv("TRAVEL_CHAT_VERBOSE_LOGS", "true").lower() in ("true", "1", "yes")
        
        # _force_fast_search: Always use fast search path (for testing/debugging)
        self._force_fast_search = os.getenv("TRAVEL_CHAT_FORCE_FAST_SEARCH", "false").lower() in ("true", "1", "yes")
        
        # _fast_model: Which model to use for quick queries (defaults to main model)
        self._fast_model = os.getenv("TRAVEL_CHAT_FAST_MODEL", self.model)
        
        # Set up console logging if verbose mode is enabled
        self._configure_console_logging()

        # Initialize safety guardrails (content filtering, prompt injection detection)
        self._guard = setup_guard()
        
        # Initialize Langfuse (observability - tracks usage, quality, costs)
        self._langfuse = setup_langfuse()
        self._langfuse_gemini_auto_instrumented = setup_langfuse_gemini_instrumentation(
            self._langfuse
        )
        
        # Initialize checkpointer manager (saves conversation state to SQLite)
        # This allows resuming conversations after restarts
        self._checkpointer_manager = SqliteCheckpointerManager(LANGGRAPH_CHECKPOINT_DB_PATH)
        self._checkpointer = self._checkpointer_manager.setup()
        
        # Initialize response cache (stores query → response mappings)
        # Avoids re-computing identical queries
        self._cache = TravelQueryCache(CACHE_DB_PATH)
        
        # Initialize long-term memory store (user preferences, saved info)
        initialize_long_term_memory_store()

        # Build the LangGraph workflow (state machine with nodes and edges)
        self._graph = self._build_graph()
        
        # Log the graph structure to a Mermaid diagram file
        self._log_graph_structure()
        
        # Log startup configuration
        self._log(
            "startup | model=%s | fast_model=%s | force_fast_search=%s | gemini_auto_instrumented=%s",
            self.model,
            self._fast_model,
            self._force_fast_search,
            self._langfuse_gemini_auto_instrumented,
        )

    def _build_graph(self):
        """Construct the LangGraph state machine workflow.
        
        Creates a directed graph where:
        - Nodes = processing steps (functions that transform state)
        - Edges = transitions between steps (what happens next)
        - Conditional edges = routing decisions (which path to take)
        
        The graph flow:
        1. prepare → Set up context
        2. input_guardrails → Safety checks
        3. cache_lookup → Check for cached response
        4. [Branch based on query type:]
           - Emergency → emergency_fastpath → postprocess → END
           - Quick search → search_fastpath → postprocess → END
           - Complex → planner → task_executor → synthesizer → postprocess → END
        
        Returns:
            Compiled LangGraph workflow ready to execute
        """
        # Create a new graph with TravelChatState as the state type
        # StateGraph manages how state flows between nodes
        graph = StateGraph(TravelChatState)
        
        # Add nodes (processing steps) to the graph
        # Each node is a function that takes state and returns updated state
        graph.add_node("prepare", self._node_prepare)
        graph.add_node("input_guardrails", self._node_input_guardrails)
        graph.add_node("cache_lookup", self._node_cache_lookup)
        graph.add_node("emergency_fastpath", self._node_emergency_fastpath)
        graph.add_node("search_fastpath", self._node_search_fastpath)
        graph.add_node("planner", self._node_planner)
        graph.add_node("task_executor", self._node_task_executor)
        graph.add_node("synthesizer", self._node_synthesizer)
        graph.add_node("postprocess", self._node_postprocess)

        # Set the entry point (where execution starts)
        graph.set_entry_point("prepare")
        
        # Add fixed edges (always go from A to B)
        graph.add_edge("prepare", "input_guardrails")
        
        # Add conditional edge after guardrails (routing decision)
        # _route_after_guardrails() decides which path to take
        graph.add_conditional_edges(
            "input_guardrails",  # Source node
            self._route_after_guardrails,  # Decision function
            {
                "blocked": "postprocess",  # If blocked → skip to end
                "quick": "postprocess",    # If quick reply → skip to end
                "ok": "cache_lookup"       # If OK → continue to cache check
            },
        )
        
        # Conditional edge after cache lookup
        graph.add_conditional_edges(
            "cache_lookup",
            self._route_after_cache,
            {
                "hit": "postprocess",           # Cache hit → use cached response
                "emergency": "emergency_fastpath",  # Emergency → fast response
                "fast_search": "search_fastpath",   # Quick query → Google search
                "planner": "planner",              # Complex → full planning
            },
        )
        
        # Fixed edges for fast paths (both go straight to postprocess)
        graph.add_edge("emergency_fastpath", "postprocess")
        graph.add_edge("search_fastpath", "postprocess")
        
        # Fixed edge from planner to task executor
        graph.add_edge("planner", "task_executor")
        
        # Conditional edge after task execution
        graph.add_conditional_edges(
            "task_executor",
            self._route_after_execution,
            {
                "ready": "synthesizer",    # All tasks done → synthesize answer
                "failed": "postprocess"    # Tasks failed → skip synthesis
            }
        )
        
        # Fixed edges for final steps
        graph.add_edge("synthesizer", "postprocess")
        graph.add_edge("postprocess", END)  # END = terminate execution

        # Compile the graph into an executable workflow
        # If checkpointer exists, conversations can be resumed across sessions
        return graph.compile(checkpointer=self._checkpointer) if self._checkpointer else graph.compile()

    def close(self) -> None:
        """Clean up resources when shutting down.
        
        Closes database connections for:
        - Checkpointer (conversation state storage)
        - Cache (query response storage)
        """
        self._safe_langfuse_flush()
        checkpointer_manager = getattr(self, "_checkpointer_manager", None)
        if checkpointer_manager:
            try:
                checkpointer_manager.close()
            except Exception as e:
                logger.warning("Checkpointer close failed: %s", e)
        cache = getattr(self, "_cache", None)
        if cache:
            try:
                cache.close()
            except Exception as e:
                logger.warning("Cache close failed: %s", e)

    def __del__(self) -> None:
        """Destructor - automatically called when object is garbage collected.
        
        Ensures cleanup happens even if user forgets to call close().
        """
        try:
            self.close()
        except Exception:
            # Never raise from destructor paths.
            pass

    def _safe_langfuse_flush(self) -> None:
        """Best-effort flush; never interrupt request flow."""
        langfuse_client = getattr(self, "_langfuse", None)
        if not langfuse_client:
            return
        try:
            langfuse_client.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)

    @contextmanager
    def _langfuse_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        input_payload: Optional[Any] = None,
        output_payload: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """Create a Langfuse observation context with SDK compatibility fallbacks."""
        langfuse_client = getattr(self, "_langfuse", None)
        if not langfuse_client:
            yield None
            return

        safe_session_id = normalize_langfuse_session_id(session_id) if session_id else ""
        session_scope = (
            langfuse_session_scope(safe_session_id, langfuse_client)
            if safe_session_id
            else nullcontext()
        )

        try:
            with session_scope:
                kwargs: Dict[str, Any] = {"name": name, "as_type": as_type}
                if model:
                    kwargs["model"] = model
                if input_payload is not None:
                    kwargs["input"] = input_payload
                if output_payload is not None:
                    kwargs["output"] = output_payload
                if metadata:
                    kwargs["metadata"] = metadata

                try:
                    if trace_context:
                        obs_ctx = langfuse_client.start_as_current_observation(
                            trace_context=trace_context,
                            **kwargs,
                        )
                    else:
                        obs_ctx = langfuse_client.start_as_current_observation(**kwargs)
                except TypeError:
                    obs_ctx = langfuse_client.start_as_current_observation(**kwargs)

                with obs_ctx as observation:
                    if safe_session_id and hasattr(observation, "update_trace"):
                        try:
                            observation.update_trace(session_id=safe_session_id)
                        except Exception:
                            pass
                    yield observation
        except Exception as e:
            logger.warning("Langfuse observation failed (%s): %s", name, e)
            yield None

    def _build_observation_input(self, contents: Any) -> Dict[str, Any]:
        """Keep observation payload compact for long prompts."""
        text = str(contents or "")
        return {
            "content_preview": self._preview(text, max_len=1200),
            "content_length": len(text),
        }

    def _generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Optional[types.GenerateContentConfig] = None,
        session_id: Optional[str] = None,
        observation_name: str = "gemini.generate_content",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Wrapper around Gemini generate_content with Langfuse generation span."""
        kwargs: Dict[str, Any] = {"model": model, "contents": contents}
        if config is not None:
            kwargs["config"] = config

        with self._langfuse_observation(
            name=observation_name,
            as_type="generation",
            session_id=session_id,
            model=model,
            input_payload=self._build_observation_input(contents),
            metadata=metadata,
        ) as generation:
            response = self.client.models.generate_content(**kwargs)
            if generation:
                try:
                    generation.update(output=(getattr(response, "text", "") or "").strip())
                except Exception:
                    pass
            return response

    # ========================================================================
    # GRAPH NODES - Each node is a processing step in the workflow
    # ========================================================================

    def _node_prepare(self, state: TravelChatState) -> TravelChatState:
        """Prepare node - Initial setup and context gathering.
        
        This node:
        1. Extracts the latest user message
        2. Applies input guardrails (safety checks)
        3. Builds short-term conversation context
        4. Creates the system prompt with today's date
        5. Initializes tracking fields (cache key, trace, flags)
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with prepared context and initialized fields
        """
        # Get conversation messages from state (defaults to empty list)
        messages = state.get("messages", [])
        
        # Get session ID (conversation thread identifier)
        # Defaults to "default" if not provided
        session_id = state.get("session_id") or "default"
        
        # Extract the most recent user message from conversation history
        latest_user = self._latest_user_message(messages)
        
        # Apply input guardrails to the user message
        # This checks for safety issues, prompt injection, etc.
        user_content = run_guardrails(self._guard, latest_user, stage="input")
        
        # Build short-term context (last N messages)
        # If conversation is longer than limit, only keep recent messages
        # This prevents context from growing too large
        short_term = messages[-SHORT_TERM_MEMORY_LIMIT:] if len(messages) > SHORT_TERM_MEMORY_LIMIT else messages
        
        # Format conversation history as human-readable text
        # Example output:
        #   User: I want to visit Tokyo
        #   Assistant: Great! When are you planning to go?
        #   User: In April for cherry blossoms
        conversation = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" 
            for m in short_term
        )
        
        # Get today's date in human-readable format
        # Example: "Saturday, March 21, 2026"
        today = date.today().strftime("%A, %B %d, %Y")
        
        # Build the complete system prompt
        # This includes: system instructions + today's date + session ID + conversation history
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Today's date: {today}. Session_id: {session_id}.\n"
            f"Conversation:\n{conversation}\n\n"
            f"Assistant:"
        )
        
        # Return updated state with all prepared fields
        # Using **state spreads existing fields, then we override/add new ones
        self._log_flow("prepare", "input_guardrails", output=user_content)
        return {
            **state,  # Keep existing fields
            "session_id": session_id,
            "user_content": user_content,  # Clean user message after guardrails
            "short_term": short_term,      # Recent conversation context
            "prompt": prompt,               # Complete system prompt
            "cache_key": self._cache.make_cache_key(session_id, user_content),
            "execution_trace": [],          # Track which nodes executed
            "cache_hit": False,             # Will be set to True if cache hits
            "blocked": False,               # Will be set to True if content blocked
            "emergency_query": False,       # Will be set to True if emergency detected
            "fast_search_query": False,     # Will be set to True if quick search needed
            "all_tasks_completed": False,   # Will be set to True when tasks finish
        }

    def _latest_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Extract the most recent user message from conversation history.
        
        Searches backwards through messages to find the last user message.
        This handles cases where the conversation might end with an assistant
        message (shouldn't happen but we're defensive).
        
        Args:
            messages: List of conversation messages with 'role' and 'content'
        
        Returns:
            The most recent user message content, or empty string if none found
        
        Example:
            messages = [
                {"role": "user", "content": "Plan a trip"},
                {"role": "assistant", "content": "Sure! Where?"},
                {"role": "user", "content": "Tokyo"}  # This is returned
            ]
        """
        # Iterate backwards through messages (reversed gives us newest first)
        for message in reversed(messages or []):
            # Check if this is a user message (case-insensitive)
            if str(message.get("role", "")).lower() == "user":
                # Return the content, stripped of whitespace
                return str(message.get("content", "")).strip()
        
        # If no user message found, return the last message content as fallback
        if messages:
            return str(messages[-1].get("content", "")).strip()
        
        # If no messages at all, return empty string
        return ""

    def _node_input_guardrails(self, state: TravelChatState) -> TravelChatState:
        """Input guardrails node - Safety and routing checks.
        
        This node acts as a security gate and router:
        1. Checks for empty queries
        2. Detects casual chat (greetings) → quick reply
        3. Blocks harmful content (violence, hacking, etc.)
        4. Blocks prompt injection attempts
        5. Detects emergency queries → flag for fast path
        6. Classifies as travel/non-travel query
        7. Determines if fast search is appropriate
        
        Args:
            state: Current state with user_content
        
        Returns:
            Updated state with routing flags set:
            - quick_reply: True if casual chat (with canned response)
            - blocked: True if content violates safety rules
            - emergency_query: True if urgent help needed
            - fast_search_query: True if quick Google search is enough
        """
        # Get user query in lowercase for keyword matching
        query = state.get("user_content", "").lower().strip()
        reason = ""  # Will hold blocking reason if query is rejected
        
        # Check 1: Empty query
        if not query:
            reason = "Please share a travel-related question."
        
        # Check 2: Casual chat detection (greetings, thanks, etc.)
        elif self._is_casual_chat(query):
            # Return a friendly greeting instead of processing as travel query
            reply = (
                "Hey! I am here and ready to help. Tell me your travel plan "
                "and I can help with flights, itinerary, budget, hotels, visa, or weather."
            )
            self._log("guardrails quick_reply | query=%s", query)
            # Set quick_reply flag and result - this will short-circuit to postprocess
            return {**state, "cache_hit": True, "result": reply, "quick_reply": True}
        
        # Check 3: Blocked content (harmful intent)
        elif self._contains_blocked_intent(query):
            reason = "I can only help with safe travel planning."
        
        # Check 4: Prompt injection attempts
        elif any(x in query for x in PROMPT_BYPASS):
            reason = "I can help with travel planning, but cannot follow instruction-bypass requests."
        
        # Check 5: Emergency detection
        elif self._looks_like_emergency_help(query):
            self._log("guardrails emergency query allowed | query=%s", query)
            # Allow query to proceed but flag it for emergency fast path
            return {**state, "emergency_query": True}
        
        # Check 6: Non-travel queries
        elif not any(x in query for x in TRAVEL_KEYWORDS):
            # Allow safe non-travel queries; we still try to help instead of blocking.
            self._log("guardrails non-travel query allowed | query=%s", query)
            # Check if fast search is appropriate for this query
            return {**state, "fast_search_query": self._should_use_fast_search(query)}
        
        # Check 7: Travel queries - determine if fast search is appropriate
        else:
            return {**state, "fast_search_query": self._should_use_fast_search(query)}
        
        # If we have a blocking reason, block the query
        if reason:
            self._log("guardrails input blocked | reason=%s", reason)
            return {
                **state, 
                "blocked": True, 
                "blocked_reason": reason, 
                "result": reason  # This becomes the response to user
            }
        
        # No issues detected, allow query to proceed
        return state

    def _node_cache_lookup(self, state: TravelChatState) -> TravelChatState:
        """Cache lookup node - Check if we've answered this query before.
        
        Queries the cache to see if we have a stored response for this
        exact question (or a very similar one using semantic similarity).
        
        Cache hits avoid:
        - Expensive LLM calls
        - Search API calls
        - Task graph execution
        
        Args:
            state: Current state with session_id and user_content
        
        Returns:
            Updated state with:
            - cache_hit: True if cached response found
            - cached_result: The cached response text
            - result: Set to cached response (ready for postprocess)
        """
        # Query the cache database
        # lookup() returns (text, similarity_score) if found, None if not
        hit = self._cache.lookup(
            state.get("session_id", "default"),  # Which conversation thread
            state.get("user_content", "")        # What question was asked
        )
        
        if hit:
            # Cache hit! Extract the cached text and similarity score
            text, score = hit
            self._log("cache hit | score=%.3f", score)
            
            # Return state with cached response
            # cache_hit=True will make router skip to postprocess
            return {
                **state, 
                "cache_hit": True, 
                "cached_result": text, 
                "result": text  # This becomes the final response
            }
        
        # Cache miss - we need to compute the response
        self._log("cache miss")
        return {**state, "cache_hit": False}

    def _node_emergency_fastpath(self, state: TravelChatState) -> TravelChatState:
        """Emergency fastpath node - Urgent help with official contacts.
        
        This node handles emergency queries like:
        - "I lost my passport in Tokyo"
        - "Need hospital emergency number in Paris"
        - "Police contact for theft report"
        
        Uses Google Search to find official emergency contacts and
        provides actionable steps immediately.
        
        Args:
            state: Current state with user_content (emergency query)
        
        Returns:
            Updated state with:
            - result: Emergency response with contacts and steps
            - all_tasks_completed: True (so response gets cached)
        """
        user_query = state.get("user_content", "")
        self._log("emergency fastpath start")
        
        # Build a specialized prompt for emergency situations
        # Emphasizes: concise, actionable, official sources, location-specific
        prompt = (
            "User needs urgent emergency assistance. Provide concise, actionable steps.\n"
            "Use web search to fetch the most relevant official contact numbers/websites for the user's location.\n"
            "Keep response short and high-signal with this structure:\n"
            "1) Immediate actions now\n"
            "2) Emergency contacts (with location)\n"
            "3) Embassy/consulate help (if passport/documents issue)\n"
            "4) What to prepare next\n"
            "If nationality is unknown, avoid assumptions and ask one follow-up line at the end.\n\n"
            f"User query: {user_query}"
        )
        
        # Call Gemini with Google Search enabled
        # This allows the model to look up current emergency contact info
        response = self._generate_content(
            model=self._fast_model,  # Use fast model for quick response
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
            session_id=state.get("session_id", "default"),
            observation_name="travel_chat.emergency_fastpath",
            metadata={"path": "emergency_fastpath", "tool_type": "google_search"},
        )
        
        # Extract text from response, handling None case
        result = (getattr(response, "text", "") or "").strip()
        
        # Mark as completed so result can be cached in postprocess
        # Emergency responses are worth caching (contacts don't change often)
        self._log_flow("emergency_fastpath", "postprocess", output=result)
        return {**state, "result": result, "all_tasks_completed": True}

    def _node_search_fastpath(self, state: TravelChatState) -> TravelChatState:
        """Search fastpath node - Quick Google search for simple queries.
        
        Handles queries like:
        - "What's the weather in London?"
        - "Flight prices to Tokyo"
        - "Hotel contact number"
        - "Visa requirements for Japan"
        
        Uses Google Search directly instead of full task planning.
        
        Args:
            state: Current state with user_content (search query)
        
        Returns:
            Updated state with:
            - result: Search-based response
            - all_tasks_completed: True (ready to cache)
        """
        user_query = state.get("user_content", "")
        self._log("search fastpath start")
        
        # Build a prompt optimized for quick, factual responses
        prompt = (
            "Answer the user query quickly using web search.\n"
            "Return concise, practical information only.\n"
            "Rules:\n"
            "- Keep under 8 bullets or short paragraphs.\n"
            "- Prefer official/recent info.\n"
            "- If location/date missing, ask one short follow-up at end.\n\n"
            f"User query: {user_query}"
        )
        
        # Call Gemini with Google Search
        response = self._generate_content(
            model=self._fast_model,  # Fast model for quick queries
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
            session_id=state.get("session_id", "default"),
            observation_name="travel_chat.search_fastpath",
            metadata={"path": "search_fastpath", "tool_type": "google_search"},
        )
        
        result = (getattr(response, "text", "") or "").strip()
        
        # Mark completed so response gets cached
        self._log_flow("search_fastpath", "postprocess", output=result)
        return {**state, "result": result, "all_tasks_completed": True}

    def _node_planner(self, state: TravelChatState) -> TravelChatState:
        """Planner node - Creates task graph for complex queries.
        
        This is the "brain" that decomposes complex travel requests into
        smaller, executable tasks. For example:
        
        User: "Plan a 3-day Tokyo trip with budget breakdown"
        Planner creates:
        - Task 1: Research Tokyo attractions (search)
        - Task 2: Build day-by-day itinerary (reasoning)
        - Task 3: Estimate costs per day (search + calculation)
        - Task 4: Create budget breakdown (synthesis)
        
        Tasks can depend on each other (Task 3 depends on Task 2).
        
        Args:
            state: Current state with user_content and prompt
        
        Returns:
            Updated state with:
            - planning_summary: High-level plan description
            - task_graph: List of tasks with dependencies
        """
        # Build a specialized prompt asking for a JSON task graph
        # We ask for specific JSON structure to make parsing reliable
        planner_prompt = (
            "Create a JSON task graph for this user goal. Return JSON only:\n"
            '{"planning_summary":"...","tasks":[{"id":"T1","title":"...","description":"...","tool_type":"search|memory|none","depends_on":["..."],"success_criteria":"..."}]}\n'
            "Rules: 2-6 tasks; dependencies only to previous tasks; use search for live data; use memory for preference save/recall when relevant."
            f"\n\nUser goal: {state.get('user_content','')}\nContext:\n{state.get('prompt','')}"
        )
        
        planning_summary = "Heuristic plan"  # Default if parsing fails
        tasks: List[Dict[str, Any]]
        
        try:
            # Generate task graph using Gemini
            raw = (
                self._generate_content(
                    model=self.model, 
                    contents=planner_prompt,
                    session_id=state.get("session_id", "default"),
                    observation_name="travel_chat.planner",
                    metadata={"path": "planner"},
                ).text or ""
            ).strip()
            
            # Parse JSON from response (might have extra text before/after)
            parsed = self._parse_json_object(raw)
            
            # Extract planning summary
            planning_summary = str(parsed.get("planning_summary", planning_summary))
            
            # Extract and normalize tasks
            # Normalization ensures consistent structure, valid tool types, etc.
            tasks = self._normalize_tasks(parsed.get("tasks", []))
            
            # Validate that we got at least one task
            if not tasks:
                raise ValueError("Planner returned empty task list")
                
        except Exception as e:
            # If planner fails (bad JSON, no tasks, etc.), use fallback tasks
            # Fallback creates simple generic tasks based on query keywords
            self._log("planner fallback | error=%s", str(e))
            tasks = self._fallback_tasks(state.get("user_content", ""))
        
        # Log what the planner decided
        self._log("planner summary=%s", planning_summary)
        for t in tasks:
            self._log(
                "planner task id=%s tool=%s deps=%s title=%s", 
                t["id"], t["tool_type"], t["depends_on"], t["title"]
            )
        self._log_flow("planner", "task_executor", output=planning_summary)
        
        return {**state, "planning_summary": planning_summary, "task_graph": tasks}

    def _node_task_executor(self, state: TravelChatState) -> TravelChatState:
        """Task executor node - Runs tasks in dependency order.
        
        Takes the task graph from planner and executes each task:
        1. Sort tasks by dependencies (topological order)
        2. For each task:
           - Check if dependencies succeeded
           - Execute task (search, memory, or reasoning)
           - Store output
           - Log trace for debugging
        3. Track how many tasks completed successfully
        
        Tasks are executed sequentially with dependency checking.
        If a dependency fails, dependent tasks are skipped.
        
        Args:
            state: Current state with task_graph
        
        Returns:
            Updated state with:
            - task_outputs: Dict mapping task ID → output
            - execution_trace: List of execution events
            - all_tasks_completed: True if all succeeded
            - result: Error message if tasks failed (empty otherwise)
        """
        # Sort tasks in topological order (dependencies execute first)
        tasks = self._topological(state.get("task_graph", []))
        
        # Storage for task results
        outputs: Dict[str, str] = {}  # task_id → output text
        
        # Execution trace for debugging (list of "T1:start", "T1:done", etc.)
        trace = list(state.get("execution_trace", []))
        
        # Count successful completions
        ok_count = 0
        
        # Execute each task in order
        for task in tasks:
            tid = str(task["id"])  # Task ID (e.g., "T1", "T2")
            deps = [str(d) for d in task.get("depends_on", [])]  # Dependency IDs
            
            # Check if any dependency failed
            # If dependency output starts with "ERROR:", skip this task
            if any(outputs.get(d, "").startswith("ERROR:") for d in deps):
                outputs[tid] = f"ERROR: dependency_failed={deps}"
                trace.append(f"{tid}:dependency_failed")
                continue  # Skip to next task
            
            # Record task start in trace
            trace.append(f"{tid}:start")
            self._log("task start | id=%s | tool=%s", tid, task["tool_type"])
            
            try:
                # Execute the task
                # _run_task() decides how to execute based on tool_type
                outputs[tid] = self._run_task(
                    task,
                    state.get("user_content", ""),  # User's original goal
                    outputs,  # Previous task outputs (for context)
                    state.get("session_id", "default")  # For memory operations
                )
                
                # Task succeeded!
                ok_count += 1
                trace.append(f"{tid}:done")
                self._log("task done | id=%s | output=%s", tid, self._preview(outputs[tid]))
                
            except Exception as e:
                # Task failed - store error and continue
                # We don't crash the whole workflow, just mark this task as failed
                outputs[tid] = f"ERROR: {str(e)}"
                trace.append(f"{tid}:error")
                self._log("task error | id=%s | error=%s", tid, str(e))
        
        # Check if all tasks completed successfully
        all_done = ok_count == len(tasks)
        
        # If tasks failed and we don't have a result yet, set error message
        interim = state.get("result", "")
        if not all_done and not interim:
            interim = "I could not complete all planned tasks, so I cannot provide a reliable final answer yet."
        
        self._log("task graph execution | completed=%d/%d", ok_count, len(tasks))
        self._log(
            "task_executor outputs | %s",
            self._preview(json.dumps(outputs, ensure_ascii=False))
        )
        
        return {
            **state, 
            "task_outputs": outputs,           # Results for each task
            "execution_trace": trace,          # Execution log
            "all_tasks_completed": all_done,   # Success flag
            "result": interim                  # Error message if failed
        }

    def _node_synthesizer(self, state: TravelChatState) -> TravelChatState:
        """Synthesizer node - Combines task outputs into final answer.
        
        After tasks execute, their outputs are scattered pieces:
        - Task 1: List of Tokyo attractions
        - Task 2: Day-by-day schedule
        - Task 3: Cost estimates
        
        The synthesizer weaves these into a cohesive response:
        "Here's your 3-day Tokyo itinerary with budget..."
        
        Args:
            state: Current state with task_outputs
        
        Returns:
            Updated state with result set to synthesized response
        """
        # Build a prompt with all task information
        prompt = self._build_synthesizer_prompt(state)
        
        # Generate final response using Gemini
        result = (
            self._generate_content(
                model=self.model, 
                contents=prompt,
                session_id=state.get("session_id", "default"),
                observation_name="travel_chat.synthesizer",
                metadata={"path": "synthesizer"},
            ).text or ""
        ).strip()
        self._log_flow("synthesizer", "postprocess", output=result)
        
        return {**state, "result": result}

    def _node_postprocess(self, state: TravelChatState) -> TravelChatState:
        """Postprocess node - Final safety checks, caching, observability.
        
        This is the last step before returning response to user:
        1. Apply output guardrails (safety checks on generated text)
        2. Apply custom output rules (no API keys leaked, etc.)
        3. Cache the response if successful
        4. Log to Langfuse for observability
        5. Run quality judge to score the response
        
        Args:
            state: Current state with result
        
        Returns:
            Final state with cleaned result ready for user
        """
        # Apply output guardrails (checks generated text for safety issues)
        result = run_guardrails(self._guard, state.get("result", ""), stage="output")
        
        # Apply custom output safety rules
        result = self._apply_output_guardrails(result, state.get("user_content", ""))
        
        # Cache successful responses for future use
        # Only cache if:
        # - We have a result
        # - It's not a generic safety message
        # - All tasks completed successfully
        # - Not already from cache (avoid caching cached responses)
        # - Not blocked content
        if (
            result
            and result not in NON_CACHEABLE_RESPONSES
            and state.get("all_tasks_completed")
            and not state.get("cache_hit")
            and not state.get("blocked")
        ):
            self._cache.put(
                state.get("session_id", "default"),
                state.get("user_content", ""),
                result
            )
            self._log("cache store")
        
        # Log to Langfuse if enabled (observability/monitoring)
        if self._langfuse:
            try:
                session_id = normalize_langfuse_session_id(state.get("session_id", "default"))
                with self._langfuse_observation(
                    name="travel_chat",
                    as_type="span",
                    session_id=session_id,
                    input_payload={
                        "user_message": state.get("user_content", ""),
                        "session_id": session_id,
                    },
                    output_payload=result,
                    metadata={
                        "blocked": bool(state.get("blocked")),
                        "cache_hit": bool(state.get("cache_hit")),
                        "all_tasks_completed": bool(state.get("all_tasks_completed")),
                    },
                    trace_context={"session_id": session_id},
                ) as span:
                    trace_id = None
                    if span:
                        trace_id = getattr(span, "trace_id", None) or getattr(span, "id", None)
                    self._log(
                        "langfuse trace queued | session_id=%s | trace_id=%s",
                        session_id,
                        trace_id or "unknown",
                    )

                    if trace_id:
                        # Run quality judge to score the response
                        # Judge evaluates: helpfulness, accuracy, safety, etc.
                        run_judge(
                            client=self.client,
                            model=self.model,
                            langfuse_client=self._langfuse,
                            user_msg=state.get("user_content", ""),
                            assistant_msg=result,
                            trace_id=trace_id,
                            generation_id=None,
                            conversation=state.get("short_term", []),
                        )

                self._safe_langfuse_flush()
                
            except Exception as e:
                # Log failures but don't crash (observability is non-critical)
                logger.warning("Langfuse trace/judge failed: %s", e)
        
        self._log_flow("postprocess", "END", output=result)
        return {**state, "result": result}

    # ========================================================================
    # HELPER METHODS - Supporting functions used by nodes
    # ========================================================================

    def _run_task(
        self, 
        task: Dict[str, Any], 
        user_goal: str, 
        outputs: Dict[str, str], 
        session_id: str
    ) -> str:
        """Execute a single task based on its tool_type.
        
        Routes task execution to appropriate method:
        - tool_type="search" → Use Google Search
        - tool_type="memory" → Use memory save/get tools
        - tool_type="none" → Pure reasoning (no tools)
        
        Args:
            task: Task dictionary with id, title, description, tool_type, etc.
            user_goal: Original user query
            outputs: Previous task outputs (for context)
            session_id: Conversation thread ID (for memory operations)
        
        Returns:
            Task output as string
        """
        # Build a prompt for executing this specific task
        task_prompt = (
            "Execute this single travel task. Return task output only.\n"
            f"Task={json.dumps(task)}\n"
            f"User goal={user_goal}\n"
            f"Previous outputs={json.dumps(outputs)}"
        )
        
        # Get tool type (search, memory, or none)
        tool_type = str(task.get("tool_type", "none")).lower()
        
        # Route 1: Search tool
        if tool_type == "search":
            response = self._generate_content(
                model=self.model, 
                contents=task_prompt, 
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
                session_id=session_id,
                observation_name="travel_chat.task.search",
                metadata={"tool_type": "search", "task_id": str(task.get("id", ""))},
            )
            return (getattr(response, "text", "") or "").strip()
        
        # Route 2: Memory tool (save/get user preferences)
        if tool_type == "memory":
            # Build session-specific memory functions
            save_fn, get_fn = build_session_callables(session_id, self._execute_tool)
            
            response = self._generate_content(
                model=self.model,
                contents=task_prompt,
                config=types.GenerateContentConfig(
                    tools=[save_fn, get_fn],
                    # Enable automatic function calling so Gemini can call our tools
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
                ),
                session_id=session_id,
                observation_name="travel_chat.task.memory",
                metadata={"tool_type": "memory", "task_id": str(task.get("id", ""))},
            )
            return (getattr(response, "text", "") or "").strip()
        
        # Route 3: No tools (pure reasoning)
        response = self._generate_content(
            model=self.model, 
            contents=task_prompt,
            session_id=session_id,
            observation_name="travel_chat.task.reasoning",
            metadata={"tool_type": "none", "task_id": str(task.get("id", ""))},
        )
        return (getattr(response, "text", "") or "").strip()

    def _execute_tool(self, name: str, args: Dict[str, Any], session_id: str) -> str:
        """Execute a tool function (memory save/get).
        
        This is called by Gemini when it decides to use a tool.
        
        Args:
            name: Tool name ("save_user_preference" or "get_user_preference")
            args: Tool arguments (key, value, etc.)
            session_id: Conversation thread ID
        
        Returns:
            Tool execution result
        """
        self._log(
            "tool start | %s | args=%s", 
            name, 
            ",".join(sorted(args.keys())) if args else "none"
        )
        
        # Execute the tool using our tooling module
        result = execute_tool(name=name, args=args, session_id=session_id)
        
        self._log("tool done | %s | %s", name, self._preview(result))
        return result

    # ========================================================================
    # PUBLIC API METHODS - Called by route handlers
    # ========================================================================

    def chat(
        self, 
        messages: List[Dict[str, str]], 
        session_id: Optional[str] = None
    ) -> str:
        """Process a chat request and return the assistant's response.
        
        This is the main entry point for non-streaming chat.
        
        Workflow:
        1. Load conversation history from checkpoint if available
        2. Extract latest user message
        3. Execute the graph workflow
        4. Return final response
        
        Args:
            messages: Conversation history [{"role": "user", "content": "..."}]
            session_id: Conversation thread ID (for checkpointing)
        
        Returns:
            Assistant's response text
        """
        try:
            # Use session ID or default
            sid = session_id or "default"

            with self._langfuse_observation(
                name="travel_chat.request",
                as_type="span",
                session_id=sid,
                input_payload={
                    "session_id": sid,
                    "messages_count": len(messages),
                    "latest_user_message": self._preview(self._latest_user_message(messages), max_len=500),
                },
                metadata={"mode": "sync"},
                trace_context={"session_id": normalize_langfuse_session_id(sid)},
            ) as request_span:
                # Configuration for checkpointer (which conversation thread)
                cfg: Dict[str, Any] = {"configurable": {"thread_id": sid}}

                # Start with full message list
                incoming = messages

                # If checkpointing is enabled, try to load conversation state
                if self._checkpointer:
                    try:
                        # Get saved state for this conversation
                        snapshot = self._graph.get_state(cfg)

                        # If we have saved messages, only process the new user message
                        # (previous messages already in checkpointed state)
                        if (getattr(snapshot, "values", {}) or {}).get("messages"):
                            latest_user_content = self._latest_user_message(messages)
                            incoming = (
                                [{"role": "user", "content": latest_user_content}]
                                if latest_user_content
                                else (messages[-1:] if messages else [])
                            )
                    except Exception:
                        # If checkpoint loading fails, use full message history
                        pass

                # Build initial state
                state_in = {"messages": incoming,
                 "session_id": sid,
                 "langfuse_trace_id": getattr(request_span, "trace_id", None)
                 }

                # Execute the graph workflow
                # If checkpointing enabled, state is saved after each node
                final_state = (
                    self._graph.invoke(state_in, config=cfg)
                    if self._checkpointer
                    else self._graph.invoke(state_in)
                )

                # Return the final result
                result = final_state.get("result", "")
                if request_span:
                    request_span.update(output=result)
                self._safe_langfuse_flush()
                return result
            
        except Exception as e:
            # Log full exception for debugging
            logger.exception("chat failed")
            # Return user-friendly error message
            return f"Sorry, I ran into an error: {str(e)}"

    def chat_payload(
        self, 
        messages: List[Dict[str, str]], 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return assistant response as structured JSON for frontend rendering.
        
        This wraps the plain text response in a structured format that
        the frontend can render nicely (headings, lists, facts, etc.).
        
        Args:
            messages: Conversation history
            session_id: Conversation thread ID
        
        Returns:
            Dictionary with:
            - message: Plain text response
            - structured: Parsed blocks for rich rendering
        """
        # Get the text response
        message = self.chat(messages, session_id=session_id)
        
        # Return structured payload
        return {
            "message": message,
            "structured": {
                "blocks": self._to_structured_blocks(message),
            },
        }

    def chat_payload_stream(
        self, 
        messages: List[Dict[str, str]], 
        session_id: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield SSE-friendly delta events while generating a response.
        
        This is the streaming version that sends response chunks as they're
        generated, creating a ChatGPT-like typing effect.
        
        The workflow:
        1. Run nodes manually (not via compiled graph)
        2. For final generation step, stream the response
        3. Yield {"delta": "chunk"} events as text arrives
        4. Yield {"done": True, "message": "...", ...} when complete
        
        Args:
            messages: Conversation history
            session_id: Conversation thread ID
        
        Yields:
            Stream events:
            - {"delta": "text chunk"} - Partial response
            - {"done": True, "message": "...", "structured": {...}} - Final
            - {"error": "..."} - If something fails
        """
        try:
            sid = session_id or "default"
            
            # Build initial state
            state: TravelChatState = {"messages": messages, "session_id": sid}
            # Emit an immediate delta so the UI shows activity right away.
            yield {"delta": "Thinking... "}
            
            # Run prepare node
            state = self._node_prepare(state)
            
            # Run input guardrails
            state = self._node_input_guardrails(state)
            guardrails_route = self._route_after_guardrails(state)

            # Check if blocked or quick reply
            if guardrails_route in {"blocked", "quick"}:
                final = self._node_postprocess(state).get("result", "")
                if final:
                    yield {"delta": final}  # Send entire response at once
                yield {
                    "done": True, 
                    "message": final, 
                    "structured": {"blocks": self._to_structured_blocks(final)}
                }
                return

            # Run cache lookup
            state = self._node_cache_lookup(state)
            cache_route = self._route_after_cache(state)
            
            # Check if cache hit
            if cache_route == "hit":
                final = self._node_postprocess(state).get("result", "")
                if final:
                    yield {"delta": final}
                yield {
                    "done": True, 
                    "message": final, 
                    "structured": {"blocks": self._to_structured_blocks(final)}
                }
                return

            # Emergency query path
            if cache_route == "emergency":
                yield {"delta": "Checking emergency info... "}
                prompt = (
                    "User needs urgent emergency assistance. Provide concise, actionable steps.\n"
                    "Use web search to fetch the most relevant official contact numbers/websites for the user's location.\n"
                    "Keep response short and high-signal with this structure:\n"
                    "1) Immediate actions now\n"
                    "2) Emergency contacts (with location)\n"
                    "3) Embassy/consulate help (if passport/documents issue)\n"
                    "4) What to prepare next\n"
                    "If nationality is unknown, avoid assumptions and ask one follow-up line at the end.\n\n"
                    f"User query: {state.get('user_content', '')}"
                )
                
                # Stream the emergency response
                final = self._stream_direct_answer(
                    model=self._fast_model,
                    prompt=prompt,
                    state=state,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    ),
                )
                for event in final:
                    yield event
                return

            # Fast search query path
            if cache_route == "fast_search":
                yield {"delta": "Searching the web... "}
                prompt = (
                    "Answer the user query quickly using web search.\n"
                    "Return concise, practical information only.\n"
                    "Rules:\n"
                    "- Keep under 8 bullets or short paragraphs.\n"
                    "- Prefer official/recent info.\n"
                    "- If location/date missing, ask one short follow-up at end.\n\n"
                    f"User query: {state.get('user_content', '')}"
                )
                
                # Stream the search response
                final = self._stream_direct_answer(
                    model=self._fast_model,
                    prompt=prompt,
                    state=state,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    ),
                )
                for event in final:
                    yield event
                return

            # Complex query path: planner → executor → synthesizer
            yield {"delta": "Planning... "}
            state = self._node_planner(state)
            yield {"delta": "Collecting details... "}
            state = self._node_task_executor(state)
            execution_route = self._route_after_execution(state)
            
            # If tasks failed, return error immediately
            if execution_route == "failed":
                final = self._node_postprocess(state).get("result", "")
                if final:
                    yield {"delta": final}
                yield {
                    "done": True, 
                    "message": final, 
                    "structured": {"blocks": self._to_structured_blocks(final)}
                }
                return

            # Stream the synthesis step
            yield {"delta": "Writing answer... \n\n"}
            synth_prompt = self._build_synthesizer_prompt(state)
            chunks: List[str] = []
            
            # Iterate over streamed text chunks
            for delta in self._iter_model_text(
                model=self.model,
                contents=synth_prompt,
                config=None,
            ):
                chunks.append(delta)
                yield {"delta": delta}  # Send chunk to frontend

            # Combine all chunks
            streamed_message = "".join(chunks).strip()
            
            # Run postprocess on complete message
            final_state = self._node_postprocess({**state, "result": streamed_message})
            final_message = final_state.get("result", "")
            
            # If postprocess added anything (safety warnings, etc.), stream it
            if final_message.startswith(streamed_message):
                tail = final_message[len(streamed_message):]
                if tail:
                    yield {"delta": tail}
            
            # Send final done event
            yield {
                "done": True,
                "message": final_message,
                "structured": {"blocks": self._to_structured_blocks(final_message)},
            }
            
        except Exception as e:
            # Log and return error event
            logger.exception("chat stream failed")
            yield {"error": f"Sorry, I ran into an error: {str(e)}"}

    # ========================================================================
    # STREAMING HELPERS
    # ========================================================================

    def _build_synthesizer_prompt(self, state: TravelChatState) -> str:
        """Build prompt for synthesizer node.
        
        Packages all task information into a JSON payload that the
        synthesizer can use to generate the final response.
        
        Args:
            state: Current state with task outputs
        
        Returns:
            Prompt string with instructions and task data
        """
        # Package all relevant information
        payload = {
            "user_goal": state.get("user_content", ""),
            "planning_summary": state.get("planning_summary", ""),
            "task_graph": state.get("task_graph", []),
            "task_outputs": state.get("task_outputs", {}),
            "execution_trace": state.get("execution_trace", []),
        }
        
        # Build prompt with instructions
        return (
            "Generate the final travel answer using completed tasks.\n"
            "Use task outputs as source-of-truth. Mention uncertainty if any task has ERROR.\n"
            "If itinerary requested, format with day headings and concise bullets.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    def _stream_direct_answer(
        self,
        model: str,
        prompt: str,
        state: TravelChatState,
        config: Optional[types.GenerateContentConfig] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream a direct answer (for emergency/search fast paths).
        
        Generates response with streaming, applies postprocess, and yields
        events in the format expected by the frontend.
        
        Args:
            model: Which Gemini model to use
            prompt: The prompt to send
            state: Current state (for postprocessing)
            config: Optional config (for enabling tools, etc.)
        
        Yields:
            {"delta": "..."} events followed by {"done": True, ...}
        """
        chunks: List[str] = []
        session_id = state.get("session_id", "default")
        
        # Stream text chunks from model
        for delta in self._iter_model_text(
            model=model,
            contents=prompt,
            config=config,
            session_id=session_id,
            observation_name="travel_chat.stream_direct_answer",
        ):
            chunks.append(delta)
            yield {"delta": delta}

        # Combine chunks
        streamed_message = "".join(chunks).strip()
        
        # Run postprocess with all_tasks_completed=True so it gets cached
        final_state = self._node_postprocess({
            **state, 
            "result": streamed_message, 
            "all_tasks_completed": True
        })
        final_message = final_state.get("result", "")
        
        # If postprocess added content, stream it
        if final_message.startswith(streamed_message):
            tail = final_message[len(streamed_message):]
            if tail:
                yield {"delta": tail}
        
        # Send done event
        yield {
            "done": True,
            "message": final_message,
            "structured": {"blocks": self._to_structured_blocks(final_message)},
        }

    def _iter_model_text(
        self,
        model: str,
        contents: str,
        config: Optional[types.GenerateContentConfig] = None,
    ) -> Iterator[str]:
        """Stream text from Gemini model.
        
        Tries to use streaming API, falls back to non-streaming if it fails.
        
        Args:
            model: Model to use
            contents: Prompt
            config: Optional generation config
        
        Yields:
            Text chunks as they arrive
        """
        pending = ""
        for piece in self._iter_model_text_raw(model=model, contents=contents, config=config):
            pending += piece
            emit_upto = self._word_flush_index(pending)
            if emit_upto <= 0:
                continue
            emit_text = pending[:emit_upto]
            pending = pending[emit_upto:]
            for token in self._split_word_tokens(emit_text):
                yield token

        if pending:
            for token in self._split_word_tokens(pending):
                yield token

    def _iter_model_text_raw(
        self,
        model: str,
        contents: str,
        config: Optional[types.GenerateContentConfig] = None,
    ) -> Iterator[str]:
        """Yield raw text chunks from Gemini without additional tokenization."""
        kwargs: Dict[str, Any] = {"model": model, "contents": contents}
        if config is not None:
            kwargs["config"] = config

        try:
            stream = self.client.models.generate_content_stream(**kwargs)
            for chunk in stream:
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield text
            return
        except Exception as e:
            self._log("stream fallback | error=%s", str(e))

        response = self.client.models.generate_content(**kwargs)
        text = (getattr(response, "text", "") or "").strip()
        if text:
            yield text

    def _word_flush_index(self, text: str) -> int:
        """Return safe flush index that keeps last partial word buffered."""
        if not text:
            return 0
        if text[-1].isspace():
            return len(text)
        last_space = -1
        for sep in (" ", "\n", "\t", "\r"):
            last_space = max(last_space, text.rfind(sep))
        return last_space + 1 if last_space >= 0 else 0

    def _split_word_tokens(self, text: str) -> List[str]:
        """Split text into small word-level tokens while preserving spacing."""
        return [token for token in re.findall(r"\S+\s*|\s+", text) if token]

    # ========================================================================
    # RESPONSE FORMATTING
    # ========================================================================

    def _to_structured_blocks(self, message: str) -> List[Dict[str, Any]]:
        """Parse plain text response into structured blocks for rich rendering.
        
        Converts text like:
        
        # Day 1: Tokyo
        - Visit Senso-ji Temple
        - Explore Shibuya
        Budget: $150
        
        Into structured blocks:
        [
          {"type": "heading", "text": "Day 1: Tokyo"},
          {"type": "list", "items": ["Visit Senso-ji Temple", "Explore Shibuya"]},
          {"type": "fact", "label": "Budget", "value": "$150"}
        ]
        
        This allows the frontend to render with proper formatting,
        colors, spacing, etc.
        
        Args:
            message: Plain text response from LLM
        
        Returns:
            List of structured blocks
        """
        text = (message or "").strip()
        if not text:
            return [{"type": "paragraph", "lines": [""]}]

        # Split into lines and normalize whitespace
        lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
        
        blocks: List[Dict[str, Any]] = []
        paragraph_lines: List[str] = []  # Accumulator for paragraph text
        list_items: List[str] = []       # Accumulator for list items

        def flush_paragraph() -> None:
            """Save accumulated paragraph lines as a block."""
            nonlocal paragraph_lines
            if paragraph_lines:
                blocks.append({"type": "paragraph", "lines": paragraph_lines})
                paragraph_lines = []

        def flush_list() -> None:
            """Save accumulated list items as a block."""
            nonlocal list_items
            if list_items:
                blocks.append({"type": "list", "items": list_items})
                list_items = []

        # Process each line
        for raw in lines:
            # Normalize multiple spaces to single space
            line = re.sub(r"\s{2,}", " ", raw).strip()
            
            # Empty line = end of current block
            if not line:
                flush_paragraph()
                flush_list()
                continue

            # Pattern 1: Markdown heading (# Heading, ## Subheading, etc.)
            heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
            if heading_match:
                flush_paragraph()
                flush_list()
                blocks.append({"type": "heading", "text": heading_match.group(1).strip()})
                continue

            # Pattern 2: Day heading (Day 1: Tokyo, Day 2 - Kyoto, etc.)
            day_match = re.match(r"^(Day\s+\d+[^:]*):?\s*(.*)$", line, flags=re.IGNORECASE)
            if day_match:
                flush_paragraph()
                flush_list()
                blocks.append({"type": "heading", "text": day_match.group(1).strip()})
                # If there's text after "Day 1:", add it as a paragraph
                detail = day_match.group(2).strip()
                if detail:
                    blocks.append({"type": "paragraph", "lines": [detail]})
                continue

            # Pattern 3: List item (- Item or * Item)
            list_match = re.match(r"^[-*]\s+(.+)$", line)
            if list_match:
                flush_paragraph()
                list_items.append(list_match.group(1).strip())
                continue

            # Pattern 4: Fact/label (Label: Value)
            fact_match = re.match(r"^([A-Za-z][^:]{2,100}):\s+(.+)$", line)
            if fact_match:
                flush_paragraph()
                flush_list()
                blocks.append({
                    "type": "fact",
                    "label": fact_match.group(1).strip(),
                    "value": fact_match.group(2).strip(),
                })
                continue

            # Pattern 5: Regular paragraph text
            paragraph_lines.append(line)

        # Flush any remaining accumulated content
        flush_paragraph()
        flush_list()
        
        # Return blocks, or default paragraph if nothing matched
        return blocks or [{"type": "paragraph", "lines": [text]}]

    # ========================================================================
    # ROUTING FUNCTIONS - Decide which path to take in the graph
    # ========================================================================

    def _route_after_guardrails(self, state: TravelChatState) -> str:
        """Route after input guardrails check.
        
        Decides where to go based on guardrails results:
        - quick_reply → Skip everything, just return canned response
        - blocked → Skip to postprocess with error message
        - ok → Continue to cache lookup
        
        Args:
            state: Current state with flags set by guardrails
        
        Returns:
            Edge name: "quick", "blocked", or "ok"
        """
        if state.get("quick_reply"):
            route = "quick"
        elif state.get("blocked"):
            route = "blocked"
        else:
            route = "ok"
        next_node = {"quick": "postprocess", "blocked": "postprocess", "ok": "cache_lookup"}[route]
        route_output = state.get("result", "") if route in {"quick", "blocked"} else ""
        self._log_flow("input_guardrails", next_node, route=route, output=route_output)
        return route

    def _route_after_cache(self, state: TravelChatState) -> str:
        """Route after cache lookup.
        
        Decides processing path based on cache result and query type:
        - cache_hit → Use cached response
        - emergency_query → Emergency fast path
        - fast_search_query → Search fast path
        - else → Full planner path
        
        Args:
            state: Current state with cache and query type flags
        
        Returns:
            Edge name: "hit", "emergency", "fast_search", or "planner"
        """
        if state.get("cache_hit"):
            route = "hit"
        elif state.get("emergency_query"):
            route = "emergency"
        elif state.get("fast_search_query"):
            route = "fast_search"
        else:
            route = "planner"
        next_node = {
            "hit": "postprocess",
            "emergency": "emergency_fastpath",
            "fast_search": "search_fastpath",
            "planner": "planner",
        }[route]
        route_output = state.get("result", "") if route == "hit" else ""
        self._log_flow("cache_lookup", next_node, route=route, output=route_output)
        return route

    def _route_after_execution(self, state: TravelChatState) -> str:
        """Route after task execution.
        
        Decides whether to synthesize or skip to postprocess:
        - all_tasks_completed → Synthesize final answer
        - else → Skip synthesis, use error message
        
        Args:
            state: Current state with execution results
        
        Returns:
            Edge name: "ready" or "failed"
        """
        route = "ready" if state.get("all_tasks_completed") else "failed"
        next_node = {"ready": "synthesizer", "failed": "postprocess"}[route]
        self._log_flow("task_executor", next_node, route=route)
        return route

    # ========================================================================
    # TASK GRAPH UTILITIES
    # ========================================================================

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from text that might contain extra content.
        
        LLMs sometimes wrap JSON in markdown or add explanatory text.
        This finds the first {...} block and parses it.
        
        Args:
            text: Text that might contain JSON
        
        Returns:
            Parsed JSON object
        
        Raises:
            ValueError: If no valid JSON found
        """
        # Find first { and last }
        s, e = text.find("{"), text.rfind("}")
        
        # Validate we found a JSON-like structure
        if s < 0 or e <= s:
            raise ValueError("planner json missing")
        
        # Extract and parse
        return json.loads(text[s : e + 1])

    def _normalize_tasks(self, tasks: Any) -> List[Dict[str, Any]]:
        """Normalize task list from planner output.
        
        Ensures:
        - Each task has required fields
        - Tool types are valid (search, memory, or none)
        - Dependencies only reference previous tasks
        - IDs are unique and sequential
        
        Args:
            tasks: Raw tasks from planner (might be malformed)
        
        Returns:
            Normalized task list
        """
        normalized: List[Dict[str, Any]] = []
        prior_ids: List[str] = []  # Track valid dependency targets
        
        # Process each task
        for idx, t in enumerate(tasks if isinstance(tasks, list) else [], start=1):
            # Get task ID or generate one
            tid = str(t.get("id") or f"T{idx}")
            
            # Validate tool type
            tool = str(t.get("tool_type", "none")).lower()
            if tool not in {"search", "memory", "none"}:
                tool = "none"  # Default to none if invalid
            
            # Filter dependencies to only valid previous tasks
            deps = [str(d) for d in t.get("depends_on", []) if str(d) in prior_ids]
            
            # Build normalized task
            normalized.append({
                "id": tid,
                "title": str(t.get("title", f"Task {idx}")),
                "description": str(t.get("description", "")),
                "tool_type": tool,
                "depends_on": deps
            })
            
            # Track this ID for future dependency validation
            prior_ids.append(tid)
        
        return normalized

    def _fallback_tasks(self, user: str) -> List[Dict[str, Any]]:
        """Generate fallback tasks when planner fails.
        
        Creates simple task lists based on query keywords:
        - Emergency query → Emergency contact lookup tasks
        - Search query → Search + draft response tasks
        - Other → Interpret + memory + draft tasks
        
        Args:
            user: User query
        
        Returns:
            List of fallback tasks
        """
        lowered = user.lower()
        
        # Check query characteristics
        searchy = any(k in lowered for k in ("time", "weather", "flight", "fare", "price", "advisory", "latest", "today"))
        emergency = self._looks_like_emergency_help(lowered)
        
        # Emergency fallback tasks
        if emergency:
            return [
                {
                    "id": "T1", 
                    "title": "Understand emergency request", 
                    "description": "Extract location and emergency type.", 
                    "tool_type": "none", 
                    "depends_on": []
                },
                {
                    "id": "T2", 
                    "title": "Find official emergency contacts", 
                    "description": "Search current police/ambulance/fire contacts for the location.", 
                    "tool_type": "search", 
                    "depends_on": ["T1"]
                },
                {
                    "id": "T3", 
                    "title": "Respond with safety-first guidance", 
                    "description": "Provide concise emergency contacts and immediate next steps.", 
                    "tool_type": "none", 
                    "depends_on": ["T2"]
                },
            ]
        
        # Search fallback tasks
        if searchy:
            return [
                {
                    "id": "T1", 
                    "title": "Interpret request", 
                    "description": "Understand travel goal.", 
                    "tool_type": "none", 
                    "depends_on": []
                },
                {
                    "id": "T2", 
                    "title": "Fetch live data", 
                    "description": "Search current facts.", 
                    "tool_type": "search", 
                    "depends_on": ["T1"]
                },
                {
                    "id": "T3", 
                    "title": "Draft answer", 
                    "description": "Compose final response.", 
                    "tool_type": "none", 
                    "depends_on": ["T2"]
                },
            ]
        
        # Default fallback tasks
        return [
            {
                "id": "T1", 
                "title": "Interpret request", 
                "description": "Understand travel goal.", 
                "tool_type": "none", 
                "depends_on": []
            },
            {
                "id": "T2", 
                "title": "Use memory", 
                "description": "Save/retrieve preferences.", 
                "tool_type": "memory", 
                "depends_on": ["T1"]
            },
            {
                "id": "T3", 
                "title": "Draft answer", 
                "description": "Compose final response.", 
                "tool_type": "none", 
                "depends_on": ["T2"]
            },
        ]

    def _topological(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tasks in topological order (dependencies first).
        
        Example:
        Input:  [T3 (deps: T1, T2), T1 (deps: []), T2 (deps: T1)]
        Output: [T1, T2, T3]
        
        Uses depth-first search to visit dependencies before dependents.
        
        Args:
            tasks: Unsorted task list
        
        Returns:
            Tasks sorted in dependency order
        """
        # Build lookup map
        by_id = {str(t["id"]): t for t in tasks}
        
        # Track visited tasks
        seen: Dict[str, bool] = {}
        
        # Result list
        ordered: List[Dict[str, Any]] = []

        def visit(tid: str) -> None:
            """Visit task and all its dependencies recursively."""
            # Skip if already visited
            if seen.get(tid):
                return
            
            # Mark as visited
            seen[tid] = True
            
            # Visit all dependencies first
            for dep in by_id[tid].get("depends_on", []):
                if str(dep) in by_id:
                    visit(str(dep))
            
            # Add this task after its dependencies
            ordered.append(by_id[tid])

        # Visit all tasks
        for task in tasks:
            visit(str(task["id"]))
        
        return ordered

    # ========================================================================
    # SAFETY AND VALIDATION
    # ========================================================================

    def _apply_output_guardrails(self, text: str, user: str) -> str:
        """Apply safety checks and enhancements to generated output.
        
        Safety checks:
        - Redact API keys/secrets
        - Block harmful content
        - Ensure travel relevance
        
        Args:
            text: Generated response
            user: Original user query
        
        Returns:
            Cleaned/enhanced response
        """
        out = (text or "").strip()
        
        # Empty response check
        if not out:
            return "I couldn't generate a response. Please rephrase your travel question."
        
        lowered = out.lower()
        
        # Block if harmful content detected in output
        if self._contains_blocked_intent(lowered):
            return "I can only provide safe travel planning help."
        
        # Redact API keys/secrets
        # Pattern matches: sk-abc123, api_key_xyz789, etc.
        out = re.sub(
            r"\b(sk|api)[-_]?[a-z0-9]{10,}\b", 
            "[redacted]", 
            out, 
            flags=re.IGNORECASE
        )
        
        # If user asked about travel but response doesn't mention it, add reminder
        if any(k in user.lower() for k in TRAVEL_KEYWORDS) and not any(k in lowered for k in TRAVEL_KEYWORDS):
            out += "\n\nI can continue with travel-specific details like itinerary, flights, hotels, or budget."
        
        return out

    def _contains_blocked_intent(self, text: str) -> bool:
        """Check if text contains harmful intent.
        
        Uses word boundaries to avoid false positives:
        - "kill" matches but "skill" doesn't
        - "make a bomb" matches but "bombastic" doesn't
        
        Args:
            text: Text to check
        
        Returns:
            True if blocked content detected
        """
        normalized = (text or "").lower()
        
        for blocked in BLOCKED_INPUT:
            token = blocked.strip().lower()
            if not token:
                continue
            
            # Multi-word phrase: simple substring match
            if " " in token:
                if token in normalized:
                    return True
                continue
            
            # Single word: use word boundary matching
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return True
        
        return False

    # ========================================================================
    # LOGGING AND DEBUGGING
    # ========================================================================

    def _configure_console_logging(self) -> None:
        """Set up console logging if verbose mode is enabled."""
        if not self._verbose_logs:
            return
        
        # Set logger level
        logger.setLevel(logging.INFO)
        logger.propagate = True
        
        # Check if handler already exists (avoid duplicates)
        if any(getattr(h, "_travel_chat_console_handler", False) for h in logger.handlers):
            return
        
        # Create and configure handler
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        
        # Mark handler so we don't add it twice
        setattr(h, "_travel_chat_console_handler", True)
        logger.addHandler(h)

    def _log_graph_structure(self) -> None:
        """Log graph structure and save Mermaid diagram."""
        # Log simplified conceptual flow
        self._log(
            "execution_graph conceptual=prepare->input_guardrails->cache_lookup->"
            "planner->task_executor->synthesizer->postprocess->END"
        )
        
        try:
            # Get compiled graph
            g = self._graph.get_graph()
            
            # Generate and save Mermaid diagram
            if g and hasattr(g, "draw_mermaid"):
                mermaid = g.draw_mermaid()
                TASK_GRAPH_MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
                TASK_GRAPH_MERMAID_PATH.write_text(mermaid, encoding="utf-8")
                self._log("execution_graph mermaid saved=%s", str(TASK_GRAPH_MERMAID_PATH))
        except Exception as e:
            logger.debug("graph mermaid unavailable: %s", e)

    def _log(self, message: str, *args: Any) -> None:
        """Log a message if verbose logging is enabled."""
        if self._verbose_logs:
            logger.info("[travel_chat] " + message, *args)

    def _log_flow(
        self,
        from_node: str,
        to_node: str,
        route: str = "",
        output: str = "",
    ) -> None:
        """Log graph flow transitions with optional output preview."""
        suffix = f" | route={route}" if route else ""
        output_preview = f" | output={self._preview(output)}" if output else ""
        self._log(
            "flow | from=%s | to=%s%s%s",
            from_node,
            to_node,
            suffix,
            output_preview,
        )

    def _preview(self, value: str, max_len: int = 140) -> str:
        """Create a preview of text for logging (truncate if too long)."""
        text = (value or "").replace("\n", " ").strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    # ========================================================================
    # QUERY CLASSIFICATION
    # ========================================================================

    def _is_casual_chat(self, query: str) -> bool:
        """Detect casual greetings and simple acknowledgments.
        
        Handles variations like:
        - "hi" → True
        - "hiii" → True (repeated letters)
        - "hello there" → True (greeting + extra words)
        - "ok" → True (simple acknowledgment)
        
        Args:
            query: User query (should be lowercased)
        
        Returns:
            True if casual chat detected
        """
        # Remove punctuation and extra spaces
        cleaned = re.sub(r"[^a-z\s]", "", (query or "").lower()).strip()
        if not cleaned:
            return False

        # Casual-chat short-circuit is intentionally conservative:
        # long messages are usually real intent ("hi i want to visit ...").
        tokens = cleaned.split()
        if len(tokens) > 4:
            return False

        # Check simple acknowledgments
        if cleaned in {"ok", "okay", "cool", "nice"}:
            return True

        # Collapse repeated letters
        # "hiii" → "hi", "heyyy" → "hey", "hellooo" → "hello"
        squashed = re.sub(r"(.)\1+", r"\1", cleaned)
        
        # Check if squashed version matches a greeting
        if any(squashed == phrase or squashed.startswith(phrase + " ") for phrase in CASUAL_CHAT):
            return True
        
        # Check original (non-squashed) version
        if any(cleaned == phrase or cleaned.startswith(phrase + " ") for phrase in CASUAL_CHAT):
            return True

        # Check for greeting-like patterns in short queries
        if len(tokens) <= 2 and tokens:
            first = tokens[0]
            # Match patterns like "hiii", "heyyy", "helloooo"
            if re.fullmatch(r"h+i+|h+e+y+|h+e+l+o+|h+e+l+l+o+|y+o+|s+u+p+", first):
                return True
        
        return False

    def _looks_like_emergency_help(self, query: str) -> bool:
        """Detect emergency/urgent help requests.
        
        Args:
            query: User query (should be lowercased)
        
        Returns:
            True if emergency keywords detected
        """
        normalized = (query or "").lower().strip()
        return any(keyword in normalized for keyword in EMERGENCY_KEYWORDS)

    def _should_use_fast_search(self, query: str) -> bool:
        """Decide if query should use fast search path.
        
        Fast search is appropriate for:
        - Factual lookups (weather, time, prices)
        - Contact information
        - Current/live data
        
        Not appropriate for:
        - Complex planning (itineraries, budgets)
        - Multi-step reasoning
        
        Args:
            query: User query (should be lowercased)
        
        Returns:
            True if fast search should be used
        """
        normalized = (query or "").lower().strip()
        if not normalized:
            return False
        
        # Force fast search if environment variable set (for testing)
        if self._force_fast_search:
            return True
        
        # Never use fast search for complex planning queries
        if any(k in normalized for k in SLOW_PLANNER_KEYWORDS):
            return False
        
        # Use fast search for quick factual queries
        return any(k in normalized for k in FAST_SEARCH_KEYWORDS)