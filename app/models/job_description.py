"""Data model for structured job descriptions."""

from typing import Dict,List, Optional
from pydantic import BaseModel, Field, field_validator


class JobDescription(BaseModel):
    """Structured and validated job description."""

    title: str = Field(
        default="Unknown Position", description="Job position / role title"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Mandatory technical and professional skills",
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Nice-to-have / preferred skills",
    )
    minimum_experience_years: Optional[float] = Field(
        default=None,
        description="Minimum years of professional experience required",
    )
    education_requirements: List[str] = Field(
        default_factory=list,
        description="Required or preferred educational degrees / disciplines",
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Core role responsibilities and duties",
    )
    raw_text: str = Field(
        default="", description="Original raw text extracted from the document"
    )

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Strip whitespace and provide default if empty."""
        if not v or not v.strip():
            return "Unknown Position"
        return v.strip()

    @field_validator("required_skills", "preferred_skills")
    @classmethod
    def deduplicate_skills(cls, v: List[str]) -> List[str]:
        """Deduplicate skills case-insensitively while preserving formatting."""
        seen = set()
        cleaned = []
        for skill in v:
            trimmed = skill.strip()
            if not trimmed:
                continue
            normalized_key = trimmed.lower()
            if normalized_key not in seen:
                seen.add(normalized_key)
                cleaned.append(trimmed)
        return cleaned
