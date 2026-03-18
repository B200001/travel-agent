"""Gemini tools and callable wrappers for travel memory."""

import json
from typing import Any, Callable, Dict, List, Tuple

from google.genai import types

from .storage import load_long_term_memory, save_long_term_memory


def build_tool_declarations() -> List[types.FunctionDeclaration]:
    """Declare custom memory tools exposed to Gemini."""
    return [
        types.FunctionDeclaration(
            name="save_preferences",
            description="Save user travel preferences for later (destination, budget, dates). Call when user shares these details.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "session_id": types.Schema(type=types.Type.STRING, description="Session identifier"),
                    "destination": types.Schema(type=types.Type.STRING, description="Travel destination"),
                    "budget": types.Schema(type=types.Type.NUMBER, description="Budget in INR"),
                    "start_date": types.Schema(type=types.Type.STRING, description="Trip start date YYYY-MM-DD"),
                    "end_date": types.Schema(type=types.Type.STRING, description="Trip end date YYYY-MM-DD"),
                    "travelers": types.Schema(type=types.Type.INTEGER, description="Number of travelers"),
                },
                required=["session_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_preferences",
            description="Retrieve saved travel preferences for this session. Call when user asks what they shared or to recall their plans.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "session_id": types.Schema(type=types.Type.STRING, description="Session identifier"),
                },
                required=["session_id"],
            ),
        ),
    ]


def execute_tool(name: str, args: Dict[str, Any], session_id: str) -> str:
    """Execute memory tools against JSON storage."""
    if name == "save_preferences":
        sid = args.get("session_id") or session_id
        mem = load_long_term_memory()
        prefs = mem.get(sid, {})

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

        mem[sid] = prefs
        save_long_term_memory(mem)
        return json.dumps({"status": "saved", "preferences": prefs})

    if name == "get_preferences":
        sid = args.get("session_id") or session_id
        mem = load_long_term_memory()
        prefs = mem.get(sid, {})
        return json.dumps(prefs if prefs else {"message": "No saved preferences for this session"})

    return json.dumps({"error": f"Unknown tool: {name}"})


def build_session_callables(
    session_id: str,
    tool_executor: Callable[[str, Dict[str, Any], str], str],
) -> Tuple[Callable[..., str], Callable[..., str]]:
    """Build session-bound functions passed to Gemini for automatic tool calling."""
    sid = session_id

    def save_preferences(
        session_id: str = "",
        destination: str = None,
        budget: float = None,
        start_date: str = None,
        end_date: str = None,
        travelers: int = None,
    ) -> str:
        s = session_id or sid
        args: Dict[str, Any] = {"session_id": s}
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
        return tool_executor("save_preferences", args, sid)

    def get_preferences(session_id: str = "") -> str:
        s = session_id or sid
        return tool_executor("get_preferences", {"session_id": s}, sid)

    return save_preferences, get_preferences
