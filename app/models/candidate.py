"""Data models for candidate profiles, education, and work experience."""

from typing import Dict,List, Optional
from pydantic import BaseModel, Field, field_validator


class Education(BaseModel):
    """Structured education history entry."""

    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    raw_text: Optional[str] = None


class Experience(BaseModel):
    """Structured work or research experience entry."""

    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    raw_text: Optional[str] = None


class Candidate(BaseModel):
    """Structured and validated candidate profile."""

    name: str = Field(default="Unknown Candidate", description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Contact email address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    skills: List[str] = Field(
        default_factory=list, description="Extracted and normalized skills"
    )
    education: List[Education] = Field(
        default_factory=list, description="Educational background entries"
    )
    experience: List[Experience] = Field(
        default_factory=list, description="Work and research experience entries"
    )
    raw_text: str = Field(
        default="", description="Original raw text extracted from the resume"
    )

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        """Strip whitespace from name and provide a fallback if blank."""
        if not v or not v.strip():
            return "Unknown Candidate"
        return v.strip()

    @field_validator("skills")
    @classmethod
    def deduplicate_skills(cls, v: List[str]) -> List[str]:
        """Deduplicate skills while preserving case and order."""
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
