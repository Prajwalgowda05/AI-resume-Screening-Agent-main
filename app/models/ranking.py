"""Data models for candidate matching results, score breakdowns, rankings, and batch screening."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ScoringWeights(BaseModel):
    """Configurable weights for multi-factor candidate scoring."""

    required_skills: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Weight for required skills match"
    )
    preferred_skills: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Weight for preferred skills match"
    )
    semantic_similarity: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Weight for semantic embedding similarity"
    )
    experience: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for experience match"
    )
    education: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Weight for education match"
    )

    @field_validator("education")
    @classmethod
    def validate_total_weights(cls, v: float, info) -> float:
        """Validate that non-zero configured weights sum to approximately 1.0."""
        data = info.data
        req = data.get("required_skills", 0.40)
        pref = data.get("preferred_skills", 0.20)
        sem = data.get("semantic_similarity", 0.20)
        exp = data.get("experience", 0.15)
        total = req + pref + sem + exp + v
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total:.2f})")
        return v


class ScoreBreakdown(BaseModel):
    """Granular breakdown of candidate scoring metrics."""

    required_skill_score: float = Field(
        ..., ge=0.0, le=100.0, description="Required skill match score (0-100)"
    )
    preferred_skill_score: float = Field(
        ..., ge=0.0, le=100.0, description="Preferred skill match score (0-100)"
    )
    semantic_similarity_score: float = Field(
        ..., ge=0.0, le=100.0, description="Semantic cosine similarity score (0-100)"
    )
    experience_score: float = Field(
        ..., ge=0.0, le=100.0, description="Experience match score (0-100)"
    )
    education_score: float = Field(
        ..., ge=0.0, le=100.0, description="Education match score (0-100)"
    )

    matched_required_skills: List[str] = Field(
        default_factory=list, description="List of matched mandatory skills"
    )
    missing_required_skills: List[str] = Field(
        default_factory=list, description="List of missing mandatory skills"
    )
    matched_preferred_skills: List[str] = Field(
        default_factory=list, description="List of matched preferred skills"
    )

    candidate_experience_years: float = Field(
        default=0.0, description="Total estimated years of candidate experience"
    )
    required_experience_years: Optional[float] = Field(
        default=None, description="Required minimum years of experience from JD"
    )


class MatchResult(BaseModel):
    """Comprehensive evaluation result for a candidate against a job description."""

    candidate_name: str = Field(..., description="Candidate's full name")
    job_title: str = Field(..., description="Target job title")
    final_score: float = Field(
        ..., ge=0.0, le=100.0, description="Final composite relevance score (0-100)"
    )
    breakdown: ScoreBreakdown = Field(
        ..., description="Granular component score breakdown and explanations"
    )
    summary: str = Field(
        default="", description="Human-readable evaluation summary"
    )
    explanation: str = Field(
        default="", description="Structured grounded narrative explanation"
    )


class RankedCandidate(BaseModel):
    """Ranked candidate entry with assigned rank and complete evaluation metrics."""

    rank: int = Field(..., ge=1, description="1-indexed rank position")
    candidate_name: str = Field(..., description="Candidate's full name")
    job_title: str = Field(..., description="Target job title")
    final_score: float = Field(
        ..., ge=0.0, le=100.0, description="Final composite relevance score (0-100)"
    )
    breakdown: ScoreBreakdown = Field(
        ..., description="Granular component score breakdown and explanations"
    )
    summary: str = Field(
        default="", description="Human-readable evaluation summary"
    )
    explanation: str = Field(
        default="", description="Structured grounded narrative explanation"
    )

    @property
    def matched_required_skills(self) -> List[str]:
        """Convenience property for matched required skills."""
        return self.breakdown.matched_required_skills

    @property
    def missing_required_skills(self) -> List[str]:
        """Convenience property for missing required skills."""
        return self.breakdown.missing_required_skills

    @property
    def matched_preferred_skills(self) -> List[str]:
        """Convenience property for matched preferred skills."""
        return self.breakdown.matched_preferred_skills


class RankingResult(BaseModel):
    """Collection of ranked candidates for a target job description."""

    job_title: str = Field(..., description="Target job title")
    total_candidates: int = Field(
        ..., ge=0, description="Total number of candidates evaluated"
    )
    rankings: List[RankedCandidate] = Field(
        default_factory=list, description="Ordered list of ranked candidates"
    )


class BatchScreeningResult(BaseModel):
    """Comprehensive output container for a batch resume screening run."""

    job_title: str = Field(..., description="Target job title")
    total_files: int = Field(
        ..., ge=0, description="Total resume files discovered"
    )
    successful_count: int = Field(
        ..., ge=0, description="Number of successfully processed resumes"
    )
    failed_count: int = Field(
        ..., ge=0, description="Number of failed/corrupted resumes"
    )
    rankings: List[RankedCandidate] = Field(
        default_factory=list, description="Ranked list of processed candidates"
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of file processing errors (filename, error message)",
    )
    csv_path: Optional[str] = Field(
        default=None, description="Path to exported CSV ranking file"
    )
    json_path: Optional[str] = Field(
        default=None, description="Path to exported JSON ranking file"
    )
