"""Chat endpoint — wraps the existing ShoppingGuideAgent.run() as an HTTP API."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
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
    return msgs[-20:]  # Keep last 20 turns (same as Streamlit)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the shopping guide Agent.

    The agent runs synchronously (LangChain + LLM calls are blocking),
    so we offload it to a thread pool to avoid blocking the event loop.
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    history = _build_chat_history(request.chat_history)

    try:
        # Run blocking agent in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.run(
                question=request.question,
                conv_id=request.conv_id,
                chat_history=history,
            )
        )
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
