"""Test LangGraph routing and node logic (no LLM calls)."""
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.langgraph_engine import ShoppingGuideGraph, ShoppingState, classify_stage


def _make_tool_call_msg(tool_call_id="call_1", tool_name="search"):
    """Create an AIMessage with tool_calls using the dict format accepted by AIMessage."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": tool_name,
            "args": {"query": "test"},
            "id": tool_call_id,
            "type": "tool_call",
        }],
    )


class TestClassifyStage:
    """Stage classification logic — deterministic paths."""

    def test_greeting_returns_discovery(self):
        assert classify_stage("你好", "discovery") == "discovery"
        assert classify_stage("hi", "discovery") == "discovery"

    def test_comparison_keywords(self):
        assert classify_stage("联想和华硕哪个好", "search") == "comparison"
        assert classify_stage("帮我对比一下", "search") == "comparison"

    def test_search_intent(self):
        assert classify_stage("推荐游戏本", "discovery") == "search"
        assert classify_stage("有哪些笔记本", "discovery") == "search"

    def test_needs_keywords(self):
        assert classify_stage("预算8000打游戏", "discovery") == "needs_elicitation"

    def test_objection_keywords(self):
        assert classify_stage("这个质量靠谱吗", "search") == "objection_handling"
        assert classify_stage("散热行不行", "search") == "objection_handling"

    def test_summary_keywords(self):
        assert classify_stage("谢谢，就这个了", "recommendation") == "summary"

    def test_empty_message_returns_current_stage(self):
        assert classify_stage("", "search") == "search"


class TestGraphRouting:
    """Routing logic — no LLM needed."""

    def test_route_with_tool_calls_under_limit(self):
        graph = _make_graph(max_tool_rounds=3)
        state = {
            "messages": [_make_tool_call_msg()],
            "tool_rounds": 1,
        }
        result = graph._route_after_agent(state)
        assert result == "tools"

    def test_route_with_tool_calls_at_limit(self):
        graph = _make_graph(max_tool_rounds=3)
        state = {
            "messages": [_make_tool_call_msg()],
            "tool_rounds": 3,
        }
        result = graph._route_after_agent(state)
        assert result == "end"

    def test_route_without_tool_calls(self):
        graph = _make_graph()
        state = {
            "messages": [AIMessage(content="推荐完了")],
            "tool_rounds": 1,
        }
        result = graph._route_after_agent(state)
        assert result == "end"

    def test_route_empty_messages(self):
        graph = _make_graph()
        state = {"messages": [], "tool_rounds": 0}
        result = graph._route_after_agent(state)
        assert result == "end"


class TestExtractProfileSignals:
    def test_budget_extraction(self):
        from src.agent.langgraph_engine import extract_profile_signals
        mock_store = Mock()
        extract_profile_signals("test_conv", "预算8000左右", mock_store)
        assert mock_store.update.called


def _make_graph(max_tool_rounds=3):
    """Minimal graph for testing routing logic."""
    return ShoppingGuideGraph(
        llm=Mock(),
        tools=[],
        product_retriever=Mock(),
        profile_store=Mock(),
        system_prompt="test",
        stage_classifier_prompt="test",
        max_tool_rounds=max_tool_rounds,
    )
