from __future__ import annotations

from pydantic import BaseModel, Field


class EducationEntry(BaseModel):
    institute_name: str = Field(..., min_length=1)
    degree: str = Field(..., min_length=1)


class ManualIntroductionRequest(BaseModel):
    name: str = Field(..., min_length=1)
    current_role: str | None = None
    city: str | None = None
    country: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience_summary: str | None = None
    experience_years: str | None = None
    education: list[EducationEntry] = Field(default_factory=list)
    certificates: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    target_role: str | None = None
    value_proposition: str | None = None


class IntroductionResponse(BaseModel):
    introduction: str
    word_count: int


class InterviewFeedbackRequest(BaseModel):
    introduction_script: str | None = None
    target_role: str | None = None
    interview_question: str | None = None
    additional_context: str | None = None


class InterviewFeedbackResult(BaseModel):
    articulation: str = Field(..., min_length=1)
    behavioural_cue: str = Field(..., min_length=1)
    problem_solving: str = Field(..., min_length=1)
    inprep_score: int = Field(..., ge=0, le=100)
    what_can_i_do_better: str = Field(..., min_length=1)


class ComputerVisionSummary(BaseModel):
    face_detected_in_frames: int = Field(..., ge=0)
    analyzed_frames: int = Field(..., ge=0)
    face_visibility_ratio: float = Field(..., ge=0.0, le=1.0)
    centered_face_ratio: float = Field(..., ge=0.0, le=1.0)
    average_brightness: float = Field(..., ge=0.0)
    average_sharpness: float = Field(..., ge=0.0)
    motion_score: float = Field(..., ge=0.0)
    posture_note: str = Field(..., min_length=1)
    camera_note: str = Field(..., min_length=1)


class InterviewAnalysisResponse(BaseModel):
    feedback: InterviewFeedbackResult
    transcript: str
    visual_analysis_used: bool
    analyzed_frame_count: int = Field(..., ge=0)
    computer_vision_summary: ComputerVisionSummary | None = None
