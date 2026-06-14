import base64
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from openai import APIStatusError, AuthenticationError, BadRequestError, OpenAI, RateLimitError
from pypdf import PdfReader

from app.schemas.resume import (
    ComputerVisionSummary,
    InterviewAnalysisResponse,
    InterviewFeedbackResult,
    IntroductionResponse,
    ManualIntroductionRequest,
)

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in dev until installed
    cv2 = None
    np = None

router = APIRouter()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
_VIDEO_ANALYSIS_MODEL = os.getenv("OPENAI_VIDEO_ANALYSIS_MODEL", _DEFAULT_MODEL)
_MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024
_SUPPORTED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "application/octet-stream",
}
_SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}

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

_VIDEO_ANALYSIS_PROMPT = """You are an expert interview coach reviewing a candidate's recorded interview.
Use the transcript, computer vision summary, and any provided video frames to assess delivery quality.

Write concise but specific feedback for these categories:
- articulation: tone, clarity, pacing, filler words, confidence
- behavioural_cue: posture, eye contact, gestures, facial expression, presence
- problem_solving: quality of reasoning, structure, justification, clarity of examples
- what_can_i_do_better: a practical improvement paragraph the candidate can apply next time

Scoring rules:
- Return a single integer `inprep_score` from 0 to 100.
- Reward confidence, clarity, structure, and calm body language.
- If visual data is limited, mention that naturally inside behavioural feedback instead of inventing details.
- Treat the computer vision summary as measured evidence about visibility, centering, motion, lighting, and sharpness.

Keep the tone supportive, honest, and actionable."""


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


def _write_temp_file(raw: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(raw)
        return temp_file.name


def _validate_video_upload(file: UploadFile, raw: bytes) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in _SUPPORTED_VIDEO_CONTENT_TYPES and suffix not in _SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only MP4, MOV, M4V, and WEBM videos are supported.",
        )
    if len(raw) > _MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Video file too large. Maximum size is 100 MB.",
        )
    if suffix in _SUPPORTED_VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"


def _transcribe_video(video_path: str) -> str:
    try:
        client = OpenAI()
        with open(video_path, "rb") as video_file:
            transcript = client.audio.transcriptions.create(
                model=_TRANSCRIPTION_MODEL,
                file=video_file,
            )
    except BadRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to transcribe this video: {exc}",
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

    text = getattr(transcript, "text", "") or ""
    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The interview audio could not be transcribed.",
        )
    return text


def _extract_video_frames(video_path: str, max_frames: int = 6) -> list[str]:
    if not shutil.which("ffmpeg"):
        return []

    with tempfile.TemporaryDirectory() as frames_dir:
        output_pattern = str(Path(frames_dir) / "frame-%02d.jpg")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            "fps=1/15",
            "-frames:v",
            str(max_frames),
            output_pattern,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []

        encoded_frames: list[str] = []
        for frame_path in sorted(Path(frames_dir).glob("frame-*.jpg")):
            encoded_frames.append(base64.b64encode(frame_path.read_bytes()).decode("utf-8"))
        return encoded_frames


def _load_face_cascade():
    if cv2 is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    return cv2.CascadeClassifier(str(cascade_path))


def _build_posture_note(centered_face_ratio: float, face_visibility_ratio: float) -> str:
    if face_visibility_ratio < 0.4:
        return "The face was not consistently visible, so posture and eye-contact cues were limited."
    if centered_face_ratio >= 0.7:
        return "The speaker stayed mostly centered in frame, which suggests stable posture and camera presence."
    if centered_face_ratio >= 0.4:
        return "The speaker was visible but shifted in frame at times, suggesting posture or framing drift."
    return "The speaker moved away from the center frequently, which may weaken on-camera presence."


def _build_camera_note(average_brightness: float, average_sharpness: float) -> str:
    lighting = "lighting looked usable"
    if average_brightness < 70:
        lighting = "lighting looked dim"
    elif average_brightness > 190:
        lighting = "lighting looked quite bright"

    focus = "video clarity looked acceptable"
    if average_sharpness < 40:
        focus = "the video looked soft or slightly blurry"
    elif average_sharpness > 120:
        focus = "the video looked sharp"

    return f"{lighting}, and {focus}."


def _compute_computer_vision_summary(video_path: str, max_frames: int = 24) -> ComputerVisionSummary | None:
    if cv2 is None or np is None:
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    face_cascade = _load_face_cascade()
    if face_cascade is None:
        capture.release()
        return None

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(total_frames // max_frames, 1) if total_frames else 15

    analyzed_frames = 0
    face_detected_in_frames = 0
    centered_face_frames = 0
    brightness_values: list[float] = []
    sharpness_values: list[float] = []
    motion_values: list[float] = []
    previous_gray = None
    frame_index = 0

    while analyzed_frames < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % step != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        brightness_values.append(float(gray.mean()))
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        if previous_gray is not None:
            frame_delta = cv2.absdiff(gray, previous_gray)
            motion_values.append(float(frame_delta.mean()))
        previous_gray = gray

        if len(faces) > 0:
            face_detected_in_frames += 1
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            face_center_x = x + (w / 2)
            frame_center_x = frame.shape[1] / 2
            offset_ratio = abs(face_center_x - frame_center_x) / max(frame.shape[1], 1)
            if offset_ratio <= 0.15:
                centered_face_frames += 1

        analyzed_frames += 1
        frame_index += 1

    capture.release()

    if analyzed_frames == 0:
        return None

    face_visibility_ratio = face_detected_in_frames / analyzed_frames
    centered_face_ratio = centered_face_frames / analyzed_frames
    average_brightness = sum(brightness_values) / len(brightness_values)
    average_sharpness = sum(sharpness_values) / len(sharpness_values)
    motion_score = sum(motion_values) / len(motion_values) if motion_values else 0.0

    return ComputerVisionSummary(
        face_detected_in_frames=face_detected_in_frames,
        analyzed_frames=analyzed_frames,
        face_visibility_ratio=round(face_visibility_ratio, 3),
        centered_face_ratio=round(centered_face_ratio, 3),
        average_brightness=round(average_brightness, 2),
        average_sharpness=round(average_sharpness, 2),
        motion_score=round(motion_score, 2),
        posture_note=_build_posture_note(centered_face_ratio, face_visibility_ratio),
        camera_note=_build_camera_note(average_brightness, average_sharpness),
    )


def _build_video_analysis_input(
    transcript: str,
    introduction_script: str | None,
    target_role: str | None,
    interview_question: str | None,
    additional_context: str | None,
    encoded_frames: list[str],
    computer_vision_summary: ComputerVisionSummary | None,
) -> list[dict]:
    context_parts = [f"Interview transcript:\n{transcript}"]
    if introduction_script:
        context_parts.append(f"Expected introduction or profile summary:\n{introduction_script}")
    if target_role:
        context_parts.append(f"Target role:\n{target_role}")
    if interview_question:
        context_parts.append(f"Interview prompt or question answered:\n{interview_question}")
    if additional_context:
        context_parts.append(f"Additional context:\n{additional_context}")
    if computer_vision_summary:
        context_parts.append(
            "Computer vision summary:\n"
            f"- Face visible in {computer_vision_summary.face_detected_in_frames}/"
            f"{computer_vision_summary.analyzed_frames} analyzed frames\n"
            f"- Face visibility ratio: {computer_vision_summary.face_visibility_ratio}\n"
            f"- Centered face ratio: {computer_vision_summary.centered_face_ratio}\n"
            f"- Average brightness: {computer_vision_summary.average_brightness}\n"
            f"- Average sharpness: {computer_vision_summary.average_sharpness}\n"
            f"- Motion score: {computer_vision_summary.motion_score}\n"
            f"- Posture note: {computer_vision_summary.posture_note}\n"
            f"- Camera note: {computer_vision_summary.camera_note}"
        )

    content: list[dict] = [{"type": "input_text", "text": "\n\n".join(context_parts)}]
    for encoded_frame in encoded_frames:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded_frame}",
            }
        )
    return [{"role": "user", "content": content}]


def _analyze_interview_video(
    transcript: str,
    introduction_script: str | None,
    target_role: str | None,
    interview_question: str | None,
    additional_context: str | None,
    encoded_frames: list[str],
    computer_vision_summary: ComputerVisionSummary | None,
) -> InterviewAnalysisResponse:
    try:
        client = OpenAI()
        response = client.responses.parse(
            model=_VIDEO_ANALYSIS_MODEL,
            instructions=_VIDEO_ANALYSIS_PROMPT,
            input=_build_video_analysis_input(
                transcript=transcript,
                introduction_script=introduction_script,
                target_role=target_role,
                interview_question=interview_question,
                additional_context=additional_context,
                encoded_frames=encoded_frames,
                computer_vision_summary=computer_vision_summary,
            ),
            max_output_tokens=600,
            text_format=InterviewFeedbackResult,
        )
    except BadRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to analyze this interview video: {exc}",
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

    if response.output_parsed is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI response could not be parsed into interview feedback.",
        )

    return InterviewAnalysisResponse(
        feedback=response.output_parsed,
        transcript=transcript,
        visual_analysis_used=bool(encoded_frames),
        analyzed_frame_count=len(encoded_frames),
        computer_vision_summary=computer_vision_summary,
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


@router.post("/video-analysis", response_model=InterviewAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_video_interview(
    file: UploadFile = File(...),
    introduction_script: str | None = Form(default=None),
    target_role: str | None = Form(default=None),
    interview_question: str | None = Form(default=None),
    additional_context: str | None = Form(default=None),
) -> InterviewAnalysisResponse:
    """
    Accept an interview video and return AI feedback on articulation, behaviour, and problem solving.
    """
    raw = await file.read()
    suffix = _validate_video_upload(file, raw)
    video_path = _write_temp_file(raw, suffix)

    try:
        transcript = _transcribe_video(video_path)
        encoded_frames = _extract_video_frames(video_path)
        computer_vision_summary = _compute_computer_vision_summary(video_path)
        return _analyze_interview_video(
            transcript=transcript,
            introduction_script=introduction_script,
            target_role=target_role,
            interview_question=interview_question,
            additional_context=additional_context,
            encoded_frames=encoded_frames,
            computer_vision_summary=computer_vision_summary,
        )
    finally:
        Path(video_path).unlink(missing_ok=True)
