"""Autonomous travel chat agent with planner and task graph execution."""

import json
import logging
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph

from .cache import TravelQueryCache
from .checkpointing import SqliteCheckpointerManager
from .constants import (
    CACHE_DB_PATH,
    LANGGRAPH_CHECKPOINT_DB_PATH,
    SHORT_TERM_MEMORY_LIMIT,
    SYSTEM_PROMPT,
    TASK_GRAPH_MERMAID_PATH,
)
from .integrations import run_guardrails, run_judge, setup_guard, setup_langfuse
from .state import TravelChatState
from .storage import initialize_long_term_memory_store
from .tooling import build_session_callables, execute_tool

load_dotenv()
logger = logging.getLogger(__name__)

TRAVEL_KEYWORDS = ("travel", "trip", "flight", "hotel", "itinerary", "visa", "budget", "destination", "weather")
BLOCKED_INPUT = ("make a bomb", "how to hack", "steal", "kill", "suicide")
NON_CACHEABLE_RESPONSES = {
    "I can only provide safe travel planning help.",
}
PROMPT_BYPASS = ("ignore all previous instructions", "reveal system prompt", "developer instructions")
EMERGENCY_KEYWORDS = (
    "emergency",
    "police",
    "ambulance",
    "hospital",
    "fire brigade",
    "helpline",
    "contact number",
    "phone number",
)
FAST_SEARCH_KEYWORDS = (
    "time",
    "weather",
    "price",
    "fare",
    "contact",
    "phone",
    "number",
    "where is",
    "nearest",
    "open now",
    "latest",
    "today",
    "advisory",
    "visa requirements",
)
SLOW_PLANNER_KEYWORDS = (
    "itinerary",
    "day wise",
    "day-by-day",
    "full plan",
    "detailed plan",
    "budget breakdown",
)
CASUAL_CHAT = (
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "thanks",
    "thank you",
)


class TravelChatAgent:
    """Planner-first autonomous agent."""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        self.client = genai.Client(api_key=api_key)
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"
        self._verbose_logs = os.getenv("TRAVEL_CHAT_VERBOSE_LOGS", "true").lower() in ("true", "1", "yes")
        self._force_fast_search = os.getenv("TRAVEL_CHAT_FORCE_FAST_SEARCH", "false").lower() in ("true", "1", "yes")
        self._fast_model = os.getenv("TRAVEL_CHAT_FAST_MODEL", "gemini-1.5-flash")
        self._configure_console_logging()

        self._guard = setup_guard()
        self._langfuse = setup_langfuse()
        self._checkpointer_manager = SqliteCheckpointerManager(LANGGRAPH_CHECKPOINT_DB_PATH)
        self._checkpointer = self._checkpointer_manager.setup()
        self._cache = TravelQueryCache(CACHE_DB_PATH)
        initialize_long_term_memory_store()

        self._graph = self._build_graph()
        self._log_graph_structure()
        self._log(
            "startup | model=%s | fast_model=%s | force_fast_search=%s",
            self.model,
            self._fast_model,
            self._force_fast_search,
        )

    def _build_graph(self):
        graph = StateGraph(TravelChatState)
        graph.add_node("prepare", self._node_prepare)
        graph.add_node("input_guardrails", self._node_input_guardrails)
        graph.add_node("cache_lookup", self._node_cache_lookup)
        graph.add_node("emergency_fastpath", self._node_emergency_fastpath)
        graph.add_node("search_fastpath", self._node_search_fastpath)
        graph.add_node("planner", self._node_planner)
        graph.add_node("task_executor", self._node_task_executor)
        graph.add_node("synthesizer", self._node_synthesizer)
        graph.add_node("postprocess", self._node_postprocess)

        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "input_guardrails")
        graph.add_conditional_edges(
            "input_guardrails",
            self._route_after_guardrails,
            {"blocked": "postprocess", "quick": "postprocess", "ok": "cache_lookup"},
        )
        graph.add_conditional_edges(
            "cache_lookup",
            self._route_after_cache,
            {
                "hit": "postprocess",
                "emergency": "emergency_fastpath",
                "fast_search": "search_fastpath",
                "planner": "planner",
            },
        )
        graph.add_edge("emergency_fastpath", "postprocess")
        graph.add_edge("search_fastpath", "postprocess")
        graph.add_edge("planner", "task_executor")
        graph.add_conditional_edges("task_executor", self._route_after_execution, {"ready": "synthesizer", "failed": "postprocess"})
        graph.add_edge("synthesizer", "postprocess")
        graph.add_edge("postprocess", END)

        return graph.compile(checkpointer=self._checkpointer) if self._checkpointer else graph.compile()

    def close(self) -> None:
        self._checkpointer_manager.close()
        self._cache.close()

    def __del__(self) -> None:
        self.close()

    def _node_prepare(self, state: TravelChatState) -> TravelChatState:
        messages = state.get("messages", [])
        session_id = state.get("session_id") or "default"
        latest_user = self._latest_user_message(messages)
        user_content = run_guardrails(self._guard, latest_user, stage="input")
        short_term = messages[-SHORT_TERM_MEMORY_LIMIT:] if len(messages) > SHORT_TERM_MEMORY_LIMIT else messages
        conversation = "\n\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in short_term)
        today = date.today().strftime("%A, %B %d, %Y")
        prompt = f"{SYSTEM_PROMPT}\n\nToday's date: {today}. Session_id: {session_id}.\nConversation:\n{conversation}\n\nAssistant:"
        return {
            **state,
            "session_id": session_id,
            "user_content": user_content,
            "short_term": short_term,
            "prompt": prompt,
            "cache_key": self._cache.make_cache_key(session_id, user_content),
            "execution_trace": [],
            "cache_hit": False,
            "blocked": False,
            "emergency_query": False,
            "fast_search_query": False,
            "all_tasks_completed": False,
        }

    def _latest_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Return most recent user message, even if list ends with assistant text."""
        for message in reversed(messages or []):
            if str(message.get("role", "")).lower() == "user":
                return str(message.get("content", "")).strip()
        if messages:
            return str(messages[-1].get("content", "")).strip()
        return ""

    def _node_input_guardrails(self, state: TravelChatState) -> TravelChatState:
        query = state.get("user_content", "").lower().strip()
        reason = ""
        if not query:
            reason = "Please share a travel-related question."
        elif self._is_casual_chat(query):
            reply = "Hey! I am here and ready to help. Tell me your travel plan and I can help with flights, itinerary, budget, hotels, visa, or weather."
            self._log("guardrails quick_reply | query=%s", query)
            return {**state, "cache_hit": True, "result": reply, "quick_reply": True}
        elif self._contains_blocked_intent(query):
            reason = "I can only help with safe travel planning."
        elif any(x in query for x in PROMPT_BYPASS):
            reason = "I can help with travel planning, but cannot follow instruction-bypass requests."
        elif self._looks_like_emergency_help(query):
            self._log("guardrails emergency query allowed | query=%s", query)
            return {**state, "emergency_query": True}
        elif not any(x in query for x in TRAVEL_KEYWORDS):
            # Allow safe non-travel queries; we still try to help instead of blocking.
            self._log("guardrails non-travel query allowed | query=%s", query)
            return {**state, "fast_search_query": self._should_use_fast_search(query)}
        else:
            return {**state, "fast_search_query": self._should_use_fast_search(query)}
        if reason:
            self._log("guardrails input blocked | reason=%s", reason)
            return {**state, "blocked": True, "blocked_reason": reason, "result": reason}
        return state

    def _node_cache_lookup(self, state: TravelChatState) -> TravelChatState:
        hit = self._cache.lookup(state.get("session_id", "default"), state.get("user_content", ""))
        if hit:
            text, score = hit
            self._log("cache hit | score=%.3f", score)
            return {**state, "cache_hit": True, "cached_result": text, "result": text}
        self._log("cache miss")
        return {**state, "cache_hit": False}

    def _node_emergency_fastpath(self, state: TravelChatState) -> TravelChatState:
        user_query = state.get("user_content", "")
        self._log("emergency fastpath start")
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
        response = self.client.models.generate_content(
            model=self._fast_model,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        result = (getattr(response, "text", "") or "").strip()
        # Mark as completed so result can be cached in postprocess.
        return {**state, "result": result, "all_tasks_completed": True}

    def _node_search_fastpath(self, state: TravelChatState) -> TravelChatState:
        user_query = state.get("user_content", "")
        self._log("search fastpath start")
        prompt = (
            "Answer the user query quickly using web search.\n"
            "Return concise, practical information only.\n"
            "Rules:\n"
            "- Keep under 8 bullets or short paragraphs.\n"
            "- Prefer official/recent info.\n"
            "- If location/date missing, ask one short follow-up at end.\n\n"
            f"User query: {user_query}"
        )
        response = self.client.models.generate_content(
            model=self._fast_model,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
        )
        result = (getattr(response, "text", "") or "").strip()
        return {**state, "result": result, "all_tasks_completed": True}

    def _node_planner(self, state: TravelChatState) -> TravelChatState:
        planner_prompt = (
            "Create a JSON task graph for this user goal. Return JSON only:\n"
            '{"planning_summary":"...","tasks":[{"id":"T1","title":"...","description":"...","tool_type":"search|memory|none","depends_on":["..."],"success_criteria":"..."}]}\n'
            "Rules: 2-6 tasks; dependencies only to previous tasks; use search for live data; use memory for preference save/recall when relevant."
            f"\n\nUser goal: {state.get('user_content','')}\nContext:\n{state.get('prompt','')}"
        )
        planning_summary = "Heuristic plan"
        tasks: List[Dict[str, Any]]
        try:
            raw = (self.client.models.generate_content(model=self.model, contents=planner_prompt).text or "").strip()
            parsed = self._parse_json_object(raw)
            planning_summary = str(parsed.get("planning_summary", planning_summary))
            tasks = self._normalize_tasks(parsed.get("tasks", []))
            if not tasks:
                raise ValueError("Planner returned empty task list")
        except Exception as e:
            self._log("planner fallback | error=%s", str(e))
            tasks = self._fallback_tasks(state.get("user_content", ""))
        self._log("planner summary=%s", planning_summary)
        for t in tasks:
            self._log("planner task id=%s tool=%s deps=%s title=%s", t["id"], t["tool_type"], t["depends_on"], t["title"])
        return {**state, "planning_summary": planning_summary, "task_graph": tasks}

    def _node_task_executor(self, state: TravelChatState) -> TravelChatState:
        tasks = self._topological(state.get("task_graph", []))
        outputs: Dict[str, str] = {}
        trace = list(state.get("execution_trace", []))
        ok_count = 0
        for task in tasks:
            tid = str(task["id"])
            deps = [str(d) for d in task.get("depends_on", [])]
            if any(outputs.get(d, "").startswith("ERROR:") for d in deps):
                outputs[tid] = f"ERROR: dependency_failed={deps}"
                trace.append(f"{tid}:dependency_failed")
                continue
            trace.append(f"{tid}:start")
            self._log("task start | id=%s | tool=%s", tid, task["tool_type"])
            try:
                outputs[tid] = self._run_task(task, state.get("user_content", ""), outputs, state.get("session_id", "default"))
                ok_count += 1
                trace.append(f"{tid}:done")
                self._log("task done | id=%s | output=%s", tid, self._preview(outputs[tid]))
            except Exception as e:
                outputs[tid] = f"ERROR: {str(e)}"
                trace.append(f"{tid}:error")
                self._log("task error | id=%s | error=%s", tid, str(e))
        all_done = ok_count == len(tasks)
        interim = state.get("result", "")
        if not all_done and not interim:
            interim = "I could not complete all planned tasks, so I cannot provide a reliable final answer yet."
        self._log("task graph execution | completed=%d/%d", ok_count, len(tasks))
        return {**state, "task_outputs": outputs, "execution_trace": trace, "all_tasks_completed": all_done, "result": interim}

    def _node_synthesizer(self, state: TravelChatState) -> TravelChatState:
        payload = {
            "user_goal": state.get("user_content", ""),
            "planning_summary": state.get("planning_summary", ""),
            "task_graph": state.get("task_graph", []),
            "task_outputs": state.get("task_outputs", {}),
            "execution_trace": state.get("execution_trace", []),
        }
        prompt = (
            "Generate the final travel answer using completed tasks.\n"
            "Use task outputs as source-of-truth. Mention uncertainty if any task has ERROR.\n"
            "If itinerary requested, format with day headings and concise bullets.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )
        result = (self.client.models.generate_content(model=self.model, contents=prompt).text or "").strip()
        return {**state, "result": result}

    def _node_postprocess(self, state: TravelChatState) -> TravelChatState:
        result = run_guardrails(self._guard, state.get("result", ""), stage="output")
        result = self._apply_output_guardrails(result, state.get("user_content", ""))
        if (
            result
            and result not in NON_CACHEABLE_RESPONSES
            and state.get("all_tasks_completed")
            and not state.get("cache_hit")
            and not state.get("blocked")
        ):
            self._cache.put(state.get("session_id", "default"), state.get("user_content", ""), result)
            self._log("cache store")
        if self._langfuse:
            try:
                with self._langfuse.start_as_current_observation(
                    name="travel_chat",
                    as_type="span",
                    input={"user_message": state.get("user_content", ""), "session_id": state.get("session_id", "default")},
                    output=result,
                ) as span:
                    trace_id = getattr(span, "trace_id", None) or getattr(span, "id", None)
                    if trace_id:
                        run_judge(
                            client=self.client,
                            model=self.model,
                            langfuse_client=self._langfuse,
                            user_msg=state.get("user_content", ""),
                            assistant_msg=result,
                            trace_id=trace_id,
                            generation_id=None,
                        )
                self._langfuse.flush()
            except Exception as e:
                logger.warning("Langfuse trace/judge failed: %s", e)
        return {**state, "result": result}

    def _run_task(self, task: Dict[str, Any], user_goal: str, outputs: Dict[str, str], session_id: str) -> str:
        task_prompt = (
            "Execute this single travel task. Return task output only.\n"
            f"Task={json.dumps(task)}\nUser goal={user_goal}\nPrevious outputs={json.dumps(outputs)}"
        )
        tool_type = str(task.get("tool_type", "none")).lower()
        if tool_type == "search":
            response = self.client.models.generate_content(
                model=self.model, contents=task_prompt, config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
            )
            return (getattr(response, "text", "") or "").strip()
        if tool_type == "memory":
            save_fn, get_fn = build_session_callables(session_id, self._execute_tool)
            response = self.client.models.generate_content(
                model=self.model,
                contents=task_prompt,
                config=types.GenerateContentConfig(
                    tools=[save_fn, get_fn],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
                ),
            )
            return (getattr(response, "text", "") or "").strip()
        response = self.client.models.generate_content(model=self.model, contents=task_prompt)
        return (getattr(response, "text", "") or "").strip()

    def _execute_tool(self, name: str, args: Dict[str, Any], session_id: str) -> str:
        self._log("tool start | %s | args=%s", name, ",".join(sorted(args.keys())) if args else "none")
        result = execute_tool(name=name, args=args, session_id=session_id)
        self._log("tool done | %s | %s", name, self._preview(result))
        return result

    def chat(self, messages: List[Dict[str, str]], session_id: Optional[str] = None) -> str:
        try:
            sid = session_id or "default"
            cfg: Dict[str, Any] = {"configurable": {"thread_id": sid}}
            incoming = messages
            if self._checkpointer:
                try:
                    snapshot = self._graph.get_state(cfg)
                    if (getattr(snapshot, "values", {}) or {}).get("messages"):
                        latest_user_content = self._latest_user_message(messages)
                        incoming = [{"role": "user", "content": latest_user_content}] if latest_user_content else (messages[-1:] if messages else [])
                except Exception:
                    pass
            state_in = {"messages": incoming, "session_id": sid}
            final_state = self._graph.invoke(state_in, config=cfg) if self._checkpointer else self._graph.invoke(state_in)
            return final_state.get("result", "")
        except Exception as e:
            logger.exception("chat failed")
            return f"Sorry, I ran into an error: {str(e)}"

    def chat_payload(self, messages: List[Dict[str, str]], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Return assistant response as structured JSON for frontend rendering."""
        message = self.chat(messages, session_id=session_id)
        return {
            "message": message,
            "structured": {
                "blocks": self._to_structured_blocks(message),
            },
        }

    def _to_structured_blocks(self, message: str) -> List[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return [{"type": "paragraph", "lines": [""]}]

        lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
        blocks: List[Dict[str, Any]] = []
        paragraph_lines: List[str] = []
        list_items: List[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if paragraph_lines:
                blocks.append({"type": "paragraph", "lines": paragraph_lines})
                paragraph_lines = []

        def flush_list() -> None:
            nonlocal list_items
            if list_items:
                blocks.append({"type": "list", "items": list_items})
                list_items = []

        for raw in lines:
            line = re.sub(r"\s{2,}", " ", raw).strip()
            if not line:
                flush_paragraph()
                flush_list()
                continue

            heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
            if heading_match:
                flush_paragraph()
                flush_list()
                blocks.append({"type": "heading", "text": heading_match.group(1).strip()})
                continue

            day_match = re.match(r"^(Day\s+\d+[^:]*):?\s*(.*)$", line, flags=re.IGNORECASE)
            if day_match:
                flush_paragraph()
                flush_list()
                blocks.append({"type": "heading", "text": day_match.group(1).strip()})
                detail = day_match.group(2).strip()
                if detail:
                    blocks.append({"type": "paragraph", "lines": [detail]})
                continue

            list_match = re.match(r"^[-*]\s+(.+)$", line)
            if list_match:
                flush_paragraph()
                list_items.append(list_match.group(1).strip())
                continue

            fact_match = re.match(r"^([A-Za-z][^:]{2,100}):\s+(.+)$", line)
            if fact_match:
                flush_paragraph()
                flush_list()
                blocks.append(
                    {
                        "type": "fact",
                        "label": fact_match.group(1).strip(),
                        "value": fact_match.group(2).strip(),
                    }
                )
                continue

            paragraph_lines.append(line)

        flush_paragraph()
        flush_list()
        return blocks or [{"type": "paragraph", "lines": [text]}]

    def _route_after_guardrails(self, state: TravelChatState) -> str:
        if state.get("quick_reply"):
            return "quick"
        return "blocked" if state.get("blocked") else "ok"

    def _route_after_cache(self, state: TravelChatState) -> str:
        if state.get("cache_hit"):
            return "hit"
        if state.get("emergency_query"):
            return "emergency"
        if state.get("fast_search_query"):
            return "fast_search"
        return "planner"

    def _route_after_execution(self, state: TravelChatState) -> str:
        return "ready" if state.get("all_tasks_completed") else "failed"

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e <= s:
            raise ValueError("planner json missing")
        return json.loads(text[s : e + 1])

    def _normalize_tasks(self, tasks: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        prior_ids: List[str] = []
        for idx, t in enumerate(tasks if isinstance(tasks, list) else [], start=1):
            tid = str(t.get("id") or f"T{idx}")
            tool = str(t.get("tool_type", "none")).lower()
            if tool not in {"search", "memory", "none"}:
                tool = "none"
            deps = [str(d) for d in t.get("depends_on", []) if str(d) in prior_ids]
            normalized.append(
                {"id": tid, "title": str(t.get("title", f"Task {idx}")), "description": str(t.get("description", "")), "tool_type": tool, "depends_on": deps}
            )
            prior_ids.append(tid)
        return normalized

    def _fallback_tasks(self, user: str) -> List[Dict[str, Any]]:
        lowered = user.lower()
        searchy = any(k in lowered for k in ("time", "weather", "flight", "fare", "price", "advisory", "latest", "today"))
        emergency = self._looks_like_emergency_help(lowered)
        if emergency:
            return [
                {"id": "T1", "title": "Understand emergency request", "description": "Extract location and emergency type.", "tool_type": "none", "depends_on": []},
                {"id": "T2", "title": "Find official emergency contacts", "description": "Search current police/ambulance/fire contacts for the location.", "tool_type": "search", "depends_on": ["T1"]},
                {"id": "T3", "title": "Respond with safety-first guidance", "description": "Provide concise emergency contacts and immediate next steps.", "tool_type": "none", "depends_on": ["T2"]},
            ]
        if searchy:
            return [
                {"id": "T1", "title": "Interpret request", "description": "Understand travel goal.", "tool_type": "none", "depends_on": []},
                {"id": "T2", "title": "Fetch live data", "description": "Search current facts.", "tool_type": "search", "depends_on": ["T1"]},
                {"id": "T3", "title": "Draft answer", "description": "Compose final response.", "tool_type": "none", "depends_on": ["T2"]},
            ]
        return [
            {"id": "T1", "title": "Interpret request", "description": "Understand travel goal.", "tool_type": "none", "depends_on": []},
            {"id": "T2", "title": "Use memory", "description": "Save/retrieve preferences.", "tool_type": "memory", "depends_on": ["T1"]},
            {"id": "T3", "title": "Draft answer", "description": "Compose final response.", "tool_type": "none", "depends_on": ["T2"]},
        ]

    def _topological(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id = {str(t["id"]): t for t in tasks}
        seen: Dict[str, bool] = {}
        ordered: List[Dict[str, Any]] = []

        def visit(tid: str) -> None:
            if seen.get(tid):
                return
            seen[tid] = True
            for dep in by_id[tid].get("depends_on", []):
                if str(dep) in by_id:
                    visit(str(dep))
            ordered.append(by_id[tid])

        for task in tasks:
            visit(str(task["id"]))
        return ordered

    def _apply_output_guardrails(self, text: str, user: str) -> str:
        out = (text or "").strip()
        if not out:
            return "I couldn't generate a response. Please rephrase your travel question."
        lowered = out.lower()
        if self._contains_blocked_intent(lowered):
            return "I can only provide safe travel planning help."
        out = re.sub(r"\b(sk|api)[-_]?[a-z0-9]{10,}\b", "[redacted]", out, flags=re.IGNORECASE)
        if any(k in user.lower() for k in TRAVEL_KEYWORDS) and not any(k in lowered for k in TRAVEL_KEYWORDS):
            out += "\n\nI can continue with travel-specific details like itinerary, flights, hotels, or budget."
        return out

    def _contains_blocked_intent(self, text: str) -> bool:
        """Match harmful intent safely without false positives like 'skill'."""
        normalized = (text or "").lower()
        for blocked in BLOCKED_INPUT:
            token = blocked.strip().lower()
            if not token:
                continue
            if " " in token:
                if token in normalized:
                    return True
                continue
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return True
        return False

    def _configure_console_logging(self) -> None:
        if not self._verbose_logs:
            return
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if any(getattr(h, "_travel_chat_console_handler", False) for h in logger.handlers):
            return
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        setattr(h, "_travel_chat_console_handler", True)
        logger.addHandler(h)

    def _log_graph_structure(self) -> None:
        self._log("execution_graph conceptual=prepare->input_guardrails->cache_lookup->planner->task_executor->synthesizer->postprocess->END")
        try:
            g = self._graph.get_graph()
            if g and hasattr(g, "draw_mermaid"):
                mermaid = g.draw_mermaid()
                TASK_GRAPH_MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)
                TASK_GRAPH_MERMAID_PATH.write_text(mermaid, encoding="utf-8")
                self._log("execution_graph mermaid saved=%s", str(TASK_GRAPH_MERMAID_PATH))
        except Exception as e:
            logger.debug("graph mermaid unavailable: %s", e)

    def _log(self, message: str, *args: Any) -> None:
        if self._verbose_logs:
            logger.info("[travel_chat] " + message, *args)

    def _preview(self, value: str, max_len: int = 140) -> str:
        text = (value or "").replace("\n", " ").strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _is_casual_chat(self, query: str) -> bool:
        cleaned = re.sub(r"[^a-z\s]", "", (query or "").lower()).strip()
        if not cleaned:
            return False

        if cleaned in {"ok", "okay", "cool", "nice"}:
            return True

        # Collapse repeated letters so "hii", "heyyy", "hellooo" map to greetings.
        squashed = re.sub(r"(.)\1+", r"\1", cleaned)
        if any(squashed == phrase or squashed.startswith(phrase + " ") for phrase in CASUAL_CHAT):
            return True
        if any(cleaned == phrase or cleaned.startswith(phrase + " ") for phrase in CASUAL_CHAT):
            return True

        tokens = cleaned.split()
        if len(tokens) <= 2 and tokens:
            first = tokens[0]
            if re.fullmatch(r"h+i+|h+e+y+|h+e+l+o+|h+e+l+l+o+|y+o+|s+u+p+", first):
                return True
        return False

    def _looks_like_emergency_help(self, query: str) -> bool:
        normalized = (query or "").lower().strip()
        return any(keyword in normalized for keyword in EMERGENCY_KEYWORDS)

    def _should_use_fast_search(self, query: str) -> bool:
        normalized = (query or "").lower().strip()
        if not normalized:
            return False
        if self._force_fast_search:
            return True
        if any(k in normalized for k in SLOW_PLANNER_KEYWORDS):
            return False
        return any(k in normalized for k in FAST_SEARCH_KEYWORDS)
