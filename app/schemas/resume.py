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
