from __future__ import annotations

import os
import time
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=1024, ge=1, le=128000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "openai"


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
