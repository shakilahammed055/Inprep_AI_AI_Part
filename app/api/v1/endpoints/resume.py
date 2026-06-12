import io
import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from openai import APIStatusError, AuthenticationError, BadRequestError, OpenAI, RateLimitError
from pypdf import PdfReader

from app.schemas.resume import IntroductionResponse, ManualIntroductionRequest

router = APIRouter()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

_INTRO_PROMPT = """You are a professional career coach. Based on the candidate information below, write a
confident, natural, first-person self-introduction that the candidate will say OUT LOUD in front
of a camera to introduce themselves.

Requirements:
- Write in first person ("I am...", "I have...", "My experience...")
- 3-5 sentences, roughly 60-90 seconds when spoken aloud
- Start with name and current role/headline (if present)
- Highlight 2-3 strongest skills or experiences
- End with what the person is looking for or what value they bring
- Sound warm and human, not like a formal CV reading
- Do NOT include formatting, bullet points, or headers, just clean spoken prose

Candidate information:
{candidate_text}"""


def _extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract text from the PDF. Make sure it is not a scanned image.",
        )
    return text


def _format_manual_profile(payload: ManualIntroductionRequest) -> str:
    profile_lines: list[str] = [f"Name: {payload.name}"]

    if payload.current_role:
        profile_lines.append(f"Current role: {payload.current_role}")
    if payload.city or payload.country:
        location = ", ".join(part for part in [payload.city, payload.country] if part)
        profile_lines.append(f"Location: {location}")
    if payload.skills:
        profile_lines.append(f"Skills: {', '.join(payload.skills)}")
    if payload.experience_summary:
        profile_lines.append(f"Experience summary: {payload.experience_summary}")
    if payload.experience_years:
        profile_lines.append(f"Experience duration: {payload.experience_years}")
    if payload.education:
        education_text = "; ".join(
            f"{entry.degree} from {entry.institute_name}" for entry in payload.education
        )
        profile_lines.append(f"Education: {education_text}")
    if payload.certificates:
        profile_lines.append(f"Certificates: {', '.join(payload.certificates)}")
    if payload.awards:
        profile_lines.append(f"Awards: {', '.join(payload.awards)}")
    if payload.target_role:
        profile_lines.append(f"Target role: {payload.target_role}")
    if payload.value_proposition:
        profile_lines.append(f"Value proposition: {payload.value_proposition}")

    return "\n".join(profile_lines)


def _generate_introduction(candidate_text: str) -> IntroductionResponse:
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=_DEFAULT_MODEL,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": _INTRO_PROMPT.format(candidate_text=candidate_text[:8000]),
                }
            ],
        )
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

    intro = (response.choices[0].message.content or "").strip()
    return IntroductionResponse(
        introduction=intro,
        word_count=len(intro.split()),
    )


@router.post("/upload", response_model=IntroductionResponse, status_code=status.HTTP_200_OK)
async def upload_resume(file: UploadFile = File(...)) -> IntroductionResponse:
    """
    Accept a PDF resume and return a camera-ready self-introduction script.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are supported.",
            )

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size is 10 MB.",
        )

    resume_text = _extract_text(raw)
    return _generate_introduction(resume_text)


@router.post("/manual-introduction", response_model=IntroductionResponse, status_code=status.HTTP_200_OK)
async def create_manual_introduction(
    payload: ManualIntroductionRequest,
) -> IntroductionResponse:
    """
    Accept manual profile details and return a camera-ready self-introduction script.
    """
    candidate_text = _format_manual_profile(payload)
    return _generate_introduction(candidate_text)
