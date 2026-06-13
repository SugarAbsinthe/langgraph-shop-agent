"""Pydantic models for FastAPI request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ---- Chat ----

class ChatMessage(BaseModel):
    """A single chat history message sent from the frontend."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""
    conv_id: str = Field(..., min_length=1, description="Conversation ID")
    question: str = Field(..., min_length=1, description="User's latest message")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation messages (role + content)"
    )


class ChatResponse(BaseModel):
    """Response body from POST /api/chat."""
    answer: str
    stage: str
    product_context: str
    user_profile: str
    tool_rounds: int


# ---- Conversations ----

class ConversationItem(BaseModel):
    """A conversation in the history list."""
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str


class CreateConversationRequest(BaseModel):
    """Request body for POST /api/conversations."""
    title: str = "新对话"
    model: str = ""


class AddMessageRequest(BaseModel):
    """Request body for POST /api/conversations/{id}/messages."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    details: Optional[dict] = None


class MessageItem(BaseModel):
    """A single message in conversation history."""
    id: int
    conv_id: str
    role: str
    content: str
    details: Optional[str] = None
    created_at: str


# ---- Health ----

class ComponentStatus(BaseModel):
    llm: bool = False
    chromadb: bool = False
    redis: bool = False
    database: bool = False


class HealthResponse(BaseModel):
    status: str
    components: ComponentStatus
