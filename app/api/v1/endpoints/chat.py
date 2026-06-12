import os
import time
import uuid
from fastapi import APIRouter, HTTPException, status
from openai import APIStatusError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionMessage,
    Choice,
    UsageInfo,
    ModelInfo,
    ModelsListResponse,
)

router = APIRouter()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(id=_DEFAULT_MODEL, created=1746057600, owned_by="openai"),
    ModelInfo(id="gpt-4.1", created=1746057600, owned_by="openai"),
    ModelInfo(id="gpt-4.1-mini", created=1746057600, owned_by="openai"),
]


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    return ModelsListResponse(data=_AVAILABLE_MODELS)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="messages array must not be empty.",
        )

    create_kwargs: dict = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
    }
    if request.temperature is not None:
        create_kwargs["temperature"] = request.temperature

    try:
        client = OpenAI()
        response = client.chat.completions.create(**create_kwargs)
    except BadRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid OpenAI API key."
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit exceeded. Please try again later.",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc.message}",
        ) from exc

    text_content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason or "stop"
    usage = response.usage

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                message=ChatCompletionMessage(content=text_content),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
    )
