"""LangGraph state machine for the shopping guide Agent.

Four-node graph:
  analyze  — load profile, classify stage, extract new profile signals
  retrieve — profile-augmented product search
  agent    — LLM with tools (loops max 3 rounds via tools node)
  tools    — ToolNode executes tool calls

Flow: analyze → retrieve → agent ⇄ tools → END

Why four nodes instead of flattening everything into one:
  - analyze and retrieve are rule-driven, deterministic steps — keeping them
    separate makes behavior predictable and debuggable
  - agent and tools are the LLM-driven loop — isolating them prevents the
    deterministic steps from being re-executed every tool round
  - The conditional edge (agent → tools or END) is the key control point:
    agent decides "do I need more info?" and the graph enforces the limit
"""

from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class ShoppingState(TypedDict):
    messages: Annotated[list, add_messages]
    conv_id: str
    stage: str
    product_context: str
    user_profile: str
    tool_rounds: int
    supervisor_rounds: int        # outer loop counter for multi-agent supervisor
    supervisor_decision: str      # "continue" | "finish"
    next_worker: str              # routing target: "discovery"|"search"|"compare"|"profile"|"recommend"|"end"
    error: str                    # error passthrough between agents


class ShoppingGuideGraph:
    """LangGraph state machine for shopping guide conversations.

    Four-node graph:
      analyze  — load profile, classify stage, extract new profile signals
      retrieve — profile-augmented product search
      agent    — LLM with tools (loops max 3 rounds via tools node)
                  dynamically selects per-stage prompt to stay focused
      tools    — ToolNode executes tool calls

    Flow: analyze → retrieve → agent ⇄ tools → END
    """

    def __init__(self, llm, tools: list, product_retriever, profile_store,
                 system_prompt: str, stage_classifier_prompt: str,
                 max_tool_rounds: int = 3, stage_prompts: dict = None):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools = tools
        self.product_retriever = product_retriever
        self.profile_store = profile_store
        self.system_prompt = system_prompt
        self.stage_classifier_prompt = stage_classifier_prompt
        self.max_tool_rounds = max_tool_rounds
        self.stage_prompts = stage_prompts or {}

        self.graph = self._build_graph(use_async=False)
        self._stream_graph = None  # lazily built for streaming

        # Build streaming LLM for async nodes
        try:
            from langchain_openai import ChatOpenAI
            if isinstance(llm, ChatOpenAI):
                sllm = ChatOpenAI(
                    model=llm.model_name,
                    openai_api_key=llm.openai_api_key,
                    openai_api_base=llm.openai_api_base,
                    streaming=True,
                    temperature=llm.temperature,
                    request_timeout=llm.request_timeout,
                )
                self.llm_stream = sllm.bind_tools(tools)
            else:
                self.llm_stream = self.llm_with_tools
        except Exception:
            self.llm_stream = self.llm_with_tools

    def _build_graph(self, use_async: bool = False):
        workflow = StateGraph(ShoppingState)

        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("agent", self._agent_node_async if use_async else self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _get_stream_graph(self):
        if self._stream_graph is None:
            self._stream_graph = self._build_graph(use_async=True)
        return self._stream_graph

    # ---- Nodes ----

    def _analyze_node(self, state: ShoppingState) -> dict:
        conv_id = state["conv_id"]
        messages = state["messages"]

        # Load current profile
        user_profile = self.profile_store.serialize_profile(conv_id)

        # Get last user message for stage classification
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        # Classify stage
        stage = self._classify_stage(last_user_msg, state.get("stage", "discovery"))

        # Extract profile signals from user message (lightweight extraction)
        if last_user_msg:
            self._extract_profile_signals(conv_id, last_user_msg)

        # Reload profile after extraction
        user_profile = self.profile_store.serialize_profile(conv_id)

        return {
            "stage": stage,
            "user_profile": user_profile,
        }

    def _retrieve_node(self, state: ShoppingState) -> dict:
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "")
        messages = state["messages"]

        # Only search products in relevant stages
        if stage not in ("search", "comparison", "recommendation", "objection_handling"):
            return {"product_context": ""}

        # Build profile-augmented query from last user message
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        if not last_user_msg:
            return {"product_context": ""}

        # Augment query with profile context
        augmented_query = last_user_msg
        if user_profile and user_profile != "(暂无画像)":
            augmented_query = f"{last_user_msg}\n用户画像: {user_profile}"

        try:
            product_context = self.product_retriever.retrieve(augmented_query, top_k=5)
        except Exception:
            product_context = "(产品检索暂时不可用)"

        return {"product_context": product_context}

    def _invoke_with_retry(self, messages: list, max_retries: int = 3):
        """Invoke LLM with exponential backoff on transient failures.

        Only retries on infrastructure errors (timeout, rate limit, connection
        reset, 502/503). Does NOT retry on model-level errors (bad request,
        context too long) — those need code or prompt fixes, not retries.
        Delay: 1s → 2s → 4s (3 attempts max).
        """
        import time as _time
        from backend.logging_config import log, Timer
        last_exc = None
        for attempt in range(max_retries):
            try:
                with Timer("llm_call", attempt=attempt + 1):
                    result = self.llm_with_tools.invoke(messages)
                return result
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if not any(kw in err_str for kw in ("timeout", "rate limit", "429", "connection", "reset", "503", "502")):
                    raise
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    log("llm_retry", attempt=attempt + 2, delay=delay)
                    _time.sleep(delay)
        log("llm_fail", attempts=max_retries, error=str(last_exc)[:100])
        raise last_exc

    def _agent_node(self, state: ShoppingState) -> dict:
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "(暂无画像)")
        product_context = state.get("product_context", "")
        conv_id = state.get("conv_id", "")
        tool_rounds = state.get("tool_rounds", 0)

        # Select per-stage prompt, fall back to default system prompt
        prompt = self.stage_prompts.get(stage, self.system_prompt)
        system_text = prompt.format(
            conv_id=conv_id,
            stage=stage,
            user_profile=user_profile,
            product_context=product_context or "(尚未搜索产品，请先挖掘用户需求)",
        )

        # Prepare messages for LLM: system + conversation
        full_messages = [SystemMessage(content=system_text)] + list(state["messages"])

        response = self._invoke_with_retry(full_messages)

        return {
            "messages": [response],
            "tool_rounds": tool_rounds + 1,
        }

    async def _agent_node_async(self, state: ShoppingState) -> dict:
        """Async agent node using streaming LLM for token-level events."""
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "(暂无画像)")
        product_context = state.get("product_context", "")
        conv_id = state.get("conv_id", "")
        tool_rounds = state.get("tool_rounds", 0)

        prompt = self.stage_prompts.get(stage, self.system_prompt)
        system_text = prompt.format(
            conv_id=conv_id, stage=stage,
            user_profile=user_profile,
            product_context=product_context or "(尚未搜索产品，请先挖掘用户需求)",
        )

        full_messages = [SystemMessage(content=system_text)] + list(state["messages"])

        # Use astream so astream_events can capture on_chat_model_stream
        response = None
        async for chunk in self.llm_stream.astream(full_messages):
            if response is None:
                response = chunk
            else:
                response += chunk

        return {
            "messages": [response] if response else [],
            "tool_rounds": tool_rounds + 1,
        }

    # ---- Routing ----

    def _route_after_agent(self, state: ShoppingState) -> Literal["tools", "end"]:
        """Route after agent node: continue to tools if LLM requested tool calls
        and we haven't hit the limit. Otherwise end the turn.

        The max_tool_rounds cap (default 3) prevents infinite agent-tool loops.
        When exceeded, the graph ends even if tool_calls are pending — _agent_node
        is responsible for producing a usable response before the limit.
        """
        messages = state["messages"]
        tool_rounds = state.get("tool_rounds", 0)

        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if tool_rounds >= self.max_tool_rounds:
                return "end"
            return "tools"
        return "end"

    # ---- Helpers ----

    def _classify_stage(self, user_message: str, current_stage: str) -> str:
        """Classify the conversation stage via lightweight LLM call."""
        return classify_stage(user_message, current_stage, self.llm, self.stage_classifier_prompt)

    def _extract_profile_signals(self, conv_id: str, user_message: str) -> None:
        """Lightweight profile signal extraction from user message."""
        extract_profile_signals(conv_id, user_message, self.profile_store)

    # ---- Public API ----

    async def run_stream(self, user_message: str, conv_id: str,
                         chat_history: list = None):
        """Async generator: emits SSE events as the graph progresses.

        Uses graph.astream() to yield after each node. Status events
        indicate phase transitions; token events carry the final assistant
        response progressively.
        """
        import asyncio
        import json as _json
        import re

        initial_state = {
            "messages": (chat_history or []) + [HumanMessage(content=user_message)],
            "conv_id": conv_id,
            "stage": "discovery",
            "product_context": "",
            "user_profile": "",
            "tool_rounds": 0,
            "supervisor_rounds": 0,
            "supervisor_decision": "continue",
            "next_worker": "discovery",
            "error": "",
        }

        def _emit(event_type, data):
            return f"event: {event_type}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        final_stage = "discovery"
        final_tool_rounds = 0
        seen_contents = set()

        try:
            async for chunk in self.graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name == "analyze":
                        final_stage = node_output.get("stage", final_stage)
                        yield _emit("stage", {"stage": final_stage})

                    elif node_name == "retrieve":
                        if node_output.get("product_context", ""):
                            yield _emit("status", {"message": "已找到相关产品"})

                    elif node_name == "tools":
                        yield _emit("status", {"message": "正在分析结果..."})

                    elif node_name == "agent":
                        final_tool_rounds = node_output.get("tool_rounds", final_tool_rounds)
                        for m in node_output.get("messages", []):
                            if not isinstance(m, AIMessage) or not m.content:
                                continue
                            if m.tool_calls:
                                names = [tc.get("name", "") for tc in m.tool_calls]
                                yield _emit("status", {"message": f"正在调用: {', '.join(names)}"})
                            elif m.content not in seen_contents:
                                seen_contents.add(m.content)
                                for chunk_text in re.split(r'(?<=[。！？\n])', m.content):
                                    if chunk_text.strip():
                                        yield _emit("token", {"content": chunk_text})
                                        await asyncio.sleep(0.01)

            yield _emit("done", {
                "stage": final_stage,
                "tool_rounds": final_tool_rounds,
            })

        except asyncio.TimeoutError:
            yield _emit("error", {"message": "请求超时，请稍后重试"})
        except Exception as exc:
            yield _emit("error", {"message": str(exc)})

    def run(self, user_message: str, conv_id: str,
            chat_history: list = None) -> dict:
        """Run the graph for one conversation turn.

        Args:
            user_message: The user's latest message.
            conv_id: Conversation ID for profile persistence.
            chat_history: Optional list of prior LangChain messages.

        Returns:
            dict with keys: messages, stage, product_context, user_profile, tool_rounds
        """
        initial_state = {
            "messages": (chat_history or []) + [HumanMessage(content=user_message)],
            "conv_id": conv_id,
            "stage": "discovery",
            "product_context": "",
            "user_profile": "",
            "tool_rounds": 0,
        }

        result = self.graph.invoke(initial_state)

        return {
            "messages": result["messages"],
            "stage": result.get("stage", "discovery"),
            "product_context": result.get("product_context", ""),
            "user_profile": result.get("user_profile", ""),
            "tool_rounds": result.get("tool_rounds", 0),
        }


# ---- Standalone helpers (usable by both old and new architecture) ----


def classify_stage(user_message: str, current_stage: str, llm=None,
                   stage_classifier_prompt: str = "") -> str:
    """Classify the conversation stage.

    Rule-first strategy: regex keywords cover ~70% of real-world inputs
    (zero latency, zero cost). LLM only invoked for ambiguous cases.
    This is a cost-latency-accuracy tradeoff: rules are fast and predictable
    but brittle; LLM is flexible but costs a call. The right balance depends
    on how well your keywords match your actual user input patterns.
    """
    if not user_message:
        return current_stage or "discovery"

    msg_lower = user_message.lower()

    # Short greeting → discovery
    if len(user_message) < 10 and any(kw in msg_lower for kw in ["你好", "hi", "hello", "在吗"]):
        return "discovery"

    # Comparison keywords → comparison
    if any(kw in msg_lower for kw in ["对比", "比较", "区别", "哪个好", "选哪个", "vs"]):
        return "comparison"

    # Objection/concern keywords → objection_handling
    if any(kw in msg_lower for kw in ["质量", "售后", "靠谱吗", "行不行", "问题多", "会不会",
                                        "散热", "卡不卡", "耐用", "翻车", "差评"]):
        return "objection_handling"

    # Needs keywords → needs_elicitation
    if any(kw in msg_lower for kw in ["预算", "打游戏", "办公", "出差", "学生", "轻薄",
                                        "画图", "剪视频", "编程", "做图", "渲染"]):
        return "needs_elicitation"

    # Search intent → search
    if any(kw in msg_lower for kw in ["推荐", "找", "搜索", "有没有", "买什么", "选一个",
                                        "有什么", "哪些"]):
        return "search"

    # Summary/closing
    if any(kw in msg_lower for kw in ["谢谢", "好的", "了解了", "就这个", "下单", "买了"]):
        return "summary"

    # Fallback: use LLM for ambiguous cases
    if llm is not None and stage_classifier_prompt:
        try:
            prompt = stage_classifier_prompt.format(
                current_stage=current_stage,
                user_message=user_message,
            )
            result = llm.invoke(prompt)
            stage = result.content.strip().lower()
            valid_stages = {"discovery", "needs_elicitation", "search", "comparison",
                            "objection_handling", "recommendation", "summary"}
            if stage in valid_stages:
                return stage
        except Exception:
            pass

    return current_stage or "discovery"


def extract_profile_signals(conv_id: str, user_message: str, profile_store) -> None:
    """Lightweight profile signal extraction from user message.

    Extracts: budget, product_category, primary_use, mobility,
    preferred_brand, exclude_brand. Callable from both
    ShoppingGuideGraph and SupervisorGraph.
    """
    import re
    msg = user_message

    # Budget patterns
    budget_patterns = [
        (r"预算\s*[:：]?\s*(\d{3,5})\s*[-到~至]\s*(\d{3,5})", lambda m: f"{m.group(1)}-{m.group(2)}"),
        (r"预算\s*[:：]?\s*(\d{3,5})", lambda m: f"{m.group(1)}-{int(m.group(1))*1.2:.0f}"),
        (r"(\d{4})\s*[-到~至]\s*(\d{4,5})", lambda m: f"{m.group(1)}-{m.group(2)}"),
        (r"([一二两三四五六七八九])\s*万", lambda m: f"{'一二两三四五六七八九'.index(m.group(1))*10000}-{('一二两三四五六七八九'.index(m.group(1))+1)*10000}"),
    ]
    for pattern, formatter in budget_patterns:
        match = re.search(pattern, msg)
        if match:
            try:
                budget = formatter(match)
                profile_store.update(conv_id, "budget", budget, confidence=0.8, source="deduced")
            except Exception:
                pass
            break

    # Product category detection
    category_map = {
        "手机": "手机", "iPhone": "手机", "华为mate": "手机", "小米14": "手机",
        "笔记本": "笔记本电脑", "电脑": "笔记本电脑", "游戏本": "笔记本电脑",
        "轻薄本": "笔记本电脑", "macbook": "笔记本电脑", "thinkpad": "笔记本电脑",
        "平板": "平板电脑", "iPad": "平板电脑", "pad": "平板电脑",
        "耳机": "无线耳机", "airpods": "无线耳机", "降噪耳机": "无线耳机",
        "手表": "智能手表", "手环": "智能手表", "watch": "智能手表",
    }
    msg_lower = msg.lower()
    for keyword, cat_val in category_map.items():
        if keyword.lower() in msg_lower:
            profile_store.update(conv_id, "product_category", cat_val, confidence=0.75, source="deduced")
            break

    # Primary use detection
    use_map = {
        "游戏": "gaming", "打游戏": "gaming", "吃鸡": "gaming", "3a": "gaming",
        "办公": "office", "文档": "office", "ppt": "office", "excel": "office",
        "编程": "coding", "代码": "coding", "开发": "coding",
        "设计": "design", "ps": "design", "pr": "design", "剪视频": "design",
        "上课": "student", "学生": "student", "作业": "student",
        "出差": "office", "携带": "office",
    }
    for keyword, use_val in use_map.items():
        if keyword in msg_lower:
            profile_store.update(conv_id, "primary_use", use_val, confidence=0.75, source="deduced")
            break

    # Mobility detection
    if any(kw in msg for kw in ["出差", "携带", "通勤", "带去", "轻便", "轻薄", "经常带"]):
        profile_store.update(conv_id, "mobility", "high", confidence=0.8, source="deduced")

    # Brand preference
    brands = ["联想", "华硕", "苹果", "华为", "惠普", "戴尔", "小米", "宏碁", "thinkpad", "macbook"]
    for brand in brands:
        if brand.lower() in msg_lower:
            profile_store.update(conv_id, "preferred_brand", brand, confidence=0.7, source="deduced")
            break

    # Brand exclusion
    for brand in brands:
        if any(kw in msg for kw in [f"不要{brand}", f"排除{brand}", f"不买{brand}", f"除{brand}"]):
            profile_store.update(conv_id, "exclude_brand", brand, confidence=0.8, source="deduced")
            break
