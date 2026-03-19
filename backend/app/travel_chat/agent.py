"""LangGraph-based travel chat agent implementation."""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph

from .checkpointing import SqliteCheckpointerManager
from .constants import LANGGRAPH_CHECKPOINT_DB_PATH, SHORT_TERM_MEMORY_LIMIT, SYSTEM_PROMPT
from .integrations import run_guardrails, run_judge, setup_guard, setup_langfuse
from .state import TravelChatState
from .storage import initialize_long_term_memory_store
from .tooling import build_session_callables, build_tool_declarations, execute_tool

load_dotenv()

logger = logging.getLogger(__name__)


class TravelChatAgent:
    """Travel planning chatbot with short/long-term memory and tool calling."""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")

        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"

        self._guard = setup_guard()
        self._langfuse = setup_langfuse()
        self._checkpointer_manager = SqliteCheckpointerManager(LANGGRAPH_CHECKPOINT_DB_PATH)
        self._checkpointer = self._checkpointer_manager.setup()

        self._tool_declarations = build_tool_declarations()
        initialize_long_term_memory_store()
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TravelChatState)
        workflow.add_node("prepare", self._node_prepare)
        workflow.add_node("configure_search", self._node_configure_search)
        workflow.add_node("configure_custom_tools", self._node_configure_custom_tools)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("postprocess", self._node_postprocess)

        workflow.set_entry_point("prepare")
        workflow.add_conditional_edges(
            "prepare",
            self._route_tools,
            {"search": "configure_search", "custom": "configure_custom_tools"},
        )
        workflow.add_edge("configure_search", "generate")
        workflow.add_edge("configure_custom_tools", "generate")
        workflow.add_edge("generate", "postprocess")
        workflow.add_edge("postprocess", END)
        if self._checkpointer:
            return workflow.compile(checkpointer=self._checkpointer)
        return workflow.compile()

    def close(self) -> None:
        """Release checkpointer resources when shutting down."""
        self._checkpointer_manager.close()

    def __del__(self) -> None:
        # Best-effort cleanup only.
        self.close()

    def _node_prepare(self, state: TravelChatState) -> TravelChatState:
        messages = state.get("messages", [])
        session_id = state.get("session_id") or "default"
        user_content = messages[-1]["content"] if messages else ""
        user_content = run_guardrails(self._guard, user_content, stage="input")

        short_term = messages[-SHORT_TERM_MEMORY_LIMIT:] if len(messages) > SHORT_TERM_MEMORY_LIMIT else messages
        history_for_prompt = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in short_term
        )

        today_str = date.today().strftime("%A, %B %d, %Y")
        ctx = f"\n\nToday's date: {today_str}. Session_id: {session_id}."
        prompt = f"{SYSTEM_PROMPT}{ctx}\n\n---\nConversation:\n{history_for_prompt}\n\nAssistant:"

        search_keywords = (
            "time",
            "weather",
            "flight",
            "current",
            "now",
            "opening hours",
            "fare",
            "price",
            "advisory",
        )
        needs_search = any(kw in user_content.lower() for kw in search_keywords)

        return {
            **state,
            "session_id": session_id,
            "user_content": user_content,
            "short_term": short_term,
            "prompt": prompt,
            "needs_search": needs_search,
        }

    def _route_tools(self, state: TravelChatState) -> str:
        return "search" if state.get("needs_search") else "custom"

    def _node_configure_search(self, state: TravelChatState) -> TravelChatState:
        return {**state, "tool_mode": "search"}

    def _node_configure_custom_tools(self, state: TravelChatState) -> TravelChatState:
        return {**state, "tool_mode": "custom"}

    def _node_generate(self, state: TravelChatState) -> TravelChatState:
        tool_mode = state.get("tool_mode", "custom")
        if tool_mode == "search":
            config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        else:
            session_id = state.get("session_id", "default")
            save_fn, get_fn = build_session_callables(session_id, self._execute_tool)
            config = types.GenerateContentConfig(
                tools=[save_fn, get_fn],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=state["prompt"],
            config=config,
        )
        result = (getattr(response, "text", None) or "").strip()
        return {**state, "result": result}

    def _node_postprocess(self, state: TravelChatState) -> TravelChatState:
        result = run_guardrails(self._guard, state.get("result", ""), stage="output")
        if self._langfuse:
            try:
                user_content = state.get("user_content", "")
                session_id = state.get("session_id", "default")
                with self._langfuse.start_as_current_observation(
                    name="travel_chat",
                    as_type="span",
                    input={"user_message": user_content, "session_id": session_id},
                    output=result,
                ) as span:
                    trace_id = getattr(span, "trace_id", None) or getattr(span, "id", None)
                    if trace_id:
                        run_judge(
                            client=self.client,
                            model=self.model,
                            langfuse_client=self._langfuse,
                            user_msg=user_content,
                            assistant_msg=result,
                            trace_id=trace_id,
                            generation_id=None,
                        )
                self._langfuse.flush()
            except Exception as e:
                logger.warning("Langfuse trace/judge failed: %s", e)
        return {**state, "result": result}

    def _execute_tool(self, name: str, args: Dict, session_id: str) -> str:
        return execute_tool(name=name, args=args, session_id=session_id)

    def chat(self, messages: List[Dict[str, str]], session_id: Optional[str] = None) -> str:
        """Main chat entry point used by the FastAPI route."""
        try:
            sid = session_id or "default"
            graph_config: Dict[str, Any] = {"configurable": {"thread_id": sid}}

            # For checkpointed threads, send only the newest user turn to avoid
            # duplicating all prior messages at each API call.
            incoming_messages = messages
            if self._checkpointer:
                try:
                    snapshot = self._graph.get_state(graph_config)
                    prior_values = getattr(snapshot, "values", {}) or {}
                    if prior_values.get("messages"):
                        incoming_messages = messages[-1:] if messages else []
                except Exception:
                    incoming_messages = messages

            if self._checkpointer:
                final_state = self._graph.invoke(
                    {"messages": incoming_messages, "session_id": sid},
                    config=graph_config,
                )
            else:
                final_state = self._graph.invoke({"messages": incoming_messages, "session_id": sid})
            return final_state.get("result", "")
        except Exception as e:
            return f"Sorry, I ran into an error: {str(e)}"
