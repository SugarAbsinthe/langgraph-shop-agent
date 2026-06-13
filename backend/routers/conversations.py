"""Conversation & message CRUD endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from backend.dependencies import get_conv_store
from backend.schemas import (
    ConversationItem,
    CreateConversationRequest,
    MessageItem,
    AddMessageRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])


# ---- Conversations ----

@router.get("/conversations", response_model=list[ConversationItem])
async def list_conversations():
    """List all conversations, newest first."""
    store = get_conv_store()
    rows = store.list_conversations()
    return [ConversationItem(**r) for r in rows]


@router.post("/conversations", response_model=ConversationItem)
async def create_conversation(body: CreateConversationRequest):
    """Create a new conversation."""
    store = get_conv_store()
    conv_id = store.create_conversation(title=body.title, model=body.model)
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return ConversationItem(**conv)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    store = get_conv_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    store.delete_conversation(conv_id)
    return {"deleted": conv_id}


# ---- Messages ----

@router.get("/conversations/{conv_id}/messages", response_model=list[MessageItem])
async def get_messages(conv_id: str):
    """Get all messages for a conversation, in chronological order."""
    store = get_conv_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = store.get_messages(conv_id)
    return [MessageItem(**r) for r in rows]


@router.post("/conversations/{conv_id}/messages", response_model=MessageItem)
async def add_message(conv_id: str, body: AddMessageRequest):
    """Append a message to a conversation."""
    store = get_conv_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msg_id = store.add_message(conv_id, body.role, body.content, body.details)
    # Fetch back the created message
    msgs = store.get_messages(conv_id)
    for m in msgs:
        if m["id"] == msg_id:
            return MessageItem(**m)
    raise HTTPException(status_code=500, detail="Failed to retrieve created message")


@router.delete("/conversations/{conv_id}/messages")
async def clear_messages(conv_id: str):
    """Clear all messages in a conversation (keep the conversation record)."""
    store = get_conv_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    store.clear_messages(conv_id)
    return {"cleared": conv_id}
