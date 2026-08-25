"""Regression tests for chat metadata and conversation cleanup contracts."""

import asyncio

import pytest
from fastapi import HTTPException

from backend.schemas import ChatRequest
from backend.routers import chat as chat_router
from backend.routers import conversations as conversation_router


class FakeAgent:
    def __init__(self, error=None):
        self.error = error

    def run(self, **kwargs):
        if self.error:
            raise self.error
        return {
            "answer": "ok",
            "stage": "search",
            "product_context": "context",
            "user_profile": "profile",
            "tool_rounds": 1,
            "agent_rounds": 2,
            "stop_reason": "completed",
        }


def test_chat_response_includes_execution_metadata(monkeypatch):
    monkeypatch.setattr(chat_router, "get_agent", lambda: FakeAgent())
    response = asyncio.run(chat_router.chat(ChatRequest(
        conv_id="conv-1", question="hello", chat_history=[]
    )))
    assert response.agent_rounds == 2
    assert response.stop_reason == "completed"


def test_chat_error_does_not_expose_internal_exception(monkeypatch):
    monkeypatch.setattr(
        chat_router, "get_agent", lambda: FakeAgent(ValueError("secret detail"))
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_router.chat(ChatRequest(
            conv_id="conv-1", question="hello", chat_history=[]
        )))
    assert exc_info.value.detail == "Agent execution failed"
    assert "secret" not in exc_info.value.detail


def test_delete_conversation_clears_runtime_state(monkeypatch):
    deleted = []
    cleared = []

    class FakeStore:
        def get_conversation(self, conv_id):
            return {"id": conv_id}

        def delete_conversation(self, conv_id):
            deleted.append(conv_id)

    monkeypatch.setattr(conversation_router, "get_conv_store", lambda: FakeStore())
    monkeypatch.setattr(
        conversation_router, "clear_conversation_runtime", lambda conv_id: cleared.append(conv_id)
    )
    result = asyncio.run(conversation_router.delete_conversation("conv-1"))
    assert result == {"deleted": "conv-1"}
    assert deleted == ["conv-1"]
    assert cleared == ["conv-1"]


def test_clear_messages_also_clears_runtime_state(monkeypatch):
    cleared_messages = []
    cleared_runtime = []

    class FakeStore:
        def get_conversation(self, conv_id):
            return {"id": conv_id}

        def clear_messages(self, conv_id):
            cleared_messages.append(conv_id)

    monkeypatch.setattr(conversation_router, "get_conv_store", lambda: FakeStore())
    monkeypatch.setattr(
        conversation_router,
        "clear_conversation_runtime",
        lambda conv_id: cleared_runtime.append(conv_id),
    )
    result = asyncio.run(conversation_router.clear_messages("conv-1"))
    assert result == {"cleared": "conv-1"}
    assert cleared_messages == ["conv-1"]
    assert cleared_runtime == ["conv-1"]
