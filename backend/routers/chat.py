"""Chat endpoint — wraps the existing ShoppingGuideAgent as an HTTP API."""

from __future__ import annotations

import asyncio
import json as _json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain.schema import HumanMessage, AIMessage

from backend.dependencies import get_agent
from backend.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _build_chat_history(chat_history: list) -> list:
    """Convert frontend ChatMessage list to LangChain message objects."""
    msgs = []
    for m in chat_history:
        if m.role == "user":
            msgs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            msgs.append(AIMessage(content=m.content))
    return msgs[-20:]


def _format_sse(event: str, data: dict | str) -> str:
    """Format a single SSE message."""
    payload = _json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


_AGENT_TIMEOUT = 60  # seconds


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the shopping guide Agent (non-streaming)."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    history = _build_chat_history(request.chat_history)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent.run(
                    question=request.question,
                    conv_id=request.conv_id,
                    chat_history=history,
                )
            ),
            timeout=_AGENT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Agent timeout for conv_id=%s after %ds", request.conv_id, _AGENT_TIMEOUT)
        raise HTTPException(status_code=504, detail="Agent response timed out, please try again")
    except Exception as e:
        logger.exception("Agent execution failed for conv_id=%s", request.conv_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    return ChatResponse(
        answer=result.get("answer", ""),
        stage=result.get("stage", "discovery"),
        product_context=result.get("product_context", ""),
        user_profile=result.get("user_profile", ""),
        tool_rounds=result.get("tool_rounds", 0),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message to the Agent and stream the response via SSE.

    Events: stage | status | token | done | error
    The client should use EventSource or fetch + ReadableStream to consume.
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    history = _build_chat_history(request.chat_history)

    async def _event_generator():
        try:
            async for sse_msg in agent.run_stream(
                question=request.question,
                conv_id=request.conv_id,
                chat_history=history,
            ):
                yield sse_msg
        except Exception as exc:
            logger.exception("Streaming failed for conv_id=%s", request.conv_id)
            yield _format_sse("error", {"message": str(exc)})

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
