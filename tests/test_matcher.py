"""Unit tests for semantic representation builders and explicit matcher utilities."""

import pytest
from app.engine.matcher import (
    build_candidate_semantic_text,
    build_jd_semantic_text,
    calculate_candidate_experience_years,
    match_explicit_skills,
)
from app.models.candidate import Candidate, Education, Experience
from app.models.job_description import JobDescription


@pytest.fixture
def sample_candidate() -> Candidate:
    """Fixture for a structured candidate."""
    return Candidate(
        name="Elena Rostova",
        email="elena@example.com",
        skills=["Python", "PyTorch", "Docker", "SQL"],
        education=[
            Education(
                degree="Master's Degree",
                field_of_study="Computer Science",
                institution="ETH Zurich",
                end_year=2023,
            )
        ],
        experience=[
            Experience(
                title="AI Research Intern",
                company="Inria",
                start_date="Jan 2022",
                end_date="Dec 2023",
                description="Trained transformer models for vision-language tasks.",
            )
        ],
        raw_text="PROJECTS\nAutonomous Drone Vision System\nBuilt object detection pipeline.",
    )


@pytest.fixture
def sample_jd() -> JobDescription:
    """Fixture for a structured job description."""
    return JobDescription(
        title="Machine Learning Engineer",
        required_skills=["Python", "PyTorch", "SQL"],
        preferred_skills=["Docker", "Kubernetes"],
        minimum_experience_years=2.0,
        education_requirements=["Master's Degree", "Degree in Computer Science"],
        responsibilities=[
            "Design and deploy production ML models.",
            "Maintain scalable inference endpoints.",
        ],
    )


class TestMatcherUtilities:
    """Test suite for semantic representation and skill matching."""

    def test_build_candidate_semantic_text(self, sample_candidate: Candidate):
        """Test synthesizing structured candidate into semantic text profile."""
        text = build_candidate_semantic_text(sample_candidate)
        assert "Candidate: Elena Rostova" in text
        assert "Core Skills: Python, PyTorch, Docker, SQL" in text
        assert "AI Research Intern at Inria" in text
        assert "Master's Degree in Computer Science from ETH Zurich" in text
        assert "Autonomous Drone Vision System" in text

    def test_build_jd_semantic_text(self, sample_jd: JobDescription):
        """Test synthesizing structured JD into semantic text profile."""
        text = build_jd_semantic_text(sample_jd)
        assert "Target Role: Machine Learning Engineer" in text
        assert "Required Skills: Python, PyTorch, SQL" in text
        assert "Preferred Skills: Docker, Kubernetes" in text
        assert "Experience Requirement: Minimum 2.0 years" in text
        assert "Design and deploy production ML models." in text

    def test_match_explicit_skills(self):
        """Test skill intersection and missing skill identification."""
        candidate_skills = ["Python", "pytorch", "FastAPI", "GIT"]
        target_skills = ["Python", "PyTorch", "Docker", "Git"]

        matched, missing = match_explicit_skills(candidate_skills, target_skills)
        assert len(matched) == 3
        assert "Docker" in missing
        assert "Docker" not in matched

    def test_match_explicit_skills_empty(self):
        """Test matching when candidate or JD has no skills."""
        matched, missing = match_explicit_skills([], ["Python", "PyTorch"])
        assert matched == []
        assert missing == ["Python", "PyTorch"]

        matched_2, missing_2 = match_explicit_skills(["Python"], [])
        assert matched_2 == []
        assert missing_2 == []

    def test_calculate_experience_years(self, sample_candidate: Candidate):
        """Test estimating total experience duration from date ranges."""
        years = calculate_candidate_experience_years(sample_candidate)
        # Jan 2022 to Dec 2023 is ~2.0 years
        assert 1.8 <= years <= 2.2

    def test_calculate_experience_empty(self):
        """Test experience calculation for empty candidate profile."""
        empty_candidate = Candidate(name="No Exp Candidate")
        years = calculate_candidate_experience_years(empty_candidate)
        assert years == 0.0
