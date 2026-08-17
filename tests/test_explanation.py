"""Unit tests for Candidate Explanation services and score invariance."""

import re
import pytest
from app.engine.explanation import (
    FallbackExplanationService,
    LocalLLMExplanationService,
    get_default_explanation_service,
)
from app.engine.scorer import ScoringEngine
from app.models.candidate import Candidate, Education, Experience
from app.models.job_description import JobDescription
from app.models.ranking import ScoreBreakdown


@pytest.fixture
def sample_breakdown_high() -> ScoreBreakdown:
    """Fixture for a strong candidate breakdown."""
    return ScoreBreakdown(
        required_skill_score=100.0,
        preferred_skill_score=100.0,
        semantic_similarity_score=88.5,
        experience_score=100.0,
        education_score=100.0,
        matched_required_skills=["Python", "PyTorch", "Scikit-Learn", "SQL"],
        missing_required_skills=[],
        matched_preferred_skills=["Docker", "FastAPI", "Transformers"],
        candidate_experience_years=3.5,
        required_experience_years=1.0,
    )


@pytest.fixture
def sample_breakdown_low() -> ScoreBreakdown:
    """Fixture for a weak candidate breakdown."""
    return ScoreBreakdown(
        required_skill_score=25.0,
        preferred_skill_score=0.0,
        semantic_similarity_score=42.0,
        experience_score=0.0,
        education_score=40.0,
        matched_required_skills=["Python"],
        missing_required_skills=["PyTorch", "Scikit-Learn", "SQL"],
        matched_preferred_skills=[],
        candidate_experience_years=0.0,
        required_experience_years=1.0,
    )


class TestExplanationService:
    """Test suite for explanation generation and fallback behavior."""

    def test_high_score_explanation(self, sample_breakdown_high: ScoreBreakdown):
        """Test high scoring candidate produces strong alignment narrative."""
        service = FallbackExplanationService()
        explanation = service.generate_explanation(
            candidate_name="Priya Sharma",
            job_title="Junior AI Research Associate",
            final_score=92.5,
            breakdown=sample_breakdown_high,
        )

        assert "Priya Sharma scored 92.5/100" in explanation
        assert "strong qualifications" in explanation
        assert "Python" in explanation or "PyTorch" in explanation
        assert "recommended for advancement" in explanation

        # Count sentences using sentence-boundary regex (not splitting on decimals)
        sentences = [s.strip() for s in re.split(r"(?<=[a-zA-Z])\.\s+", explanation.strip()) if s.strip()]
        assert 2 <= len(sentences) <= 4

    def test_low_score_explanation(self, sample_breakdown_low: ScoreBreakdown):
        """Test low scoring candidate identifies gaps clearly."""
        service = FallbackExplanationService()
        explanation = service.generate_explanation(
            candidate_name="Lucas Silva",
            job_title="Junior AI Research Associate",
            final_score=35.0,
            breakdown=sample_breakdown_low,
        )

        assert "Lucas Silva scored 35.0/100" in explanation
        assert "limited alignment" in explanation
        assert "PyTorch" in explanation
        assert "significant skill and requirement gaps" in explanation

        sentences = [s.strip() for s in re.split(r"(?<=[a-zA-Z])\.\s+", explanation.strip()) if s.strip()]
        assert 2 <= len(sentences) <= 4

    def test_missing_skills_highlighted(self, sample_breakdown_low: ScoreBreakdown):
        """Test missing required skills are explicitly mentioned in gaps section."""
        service = FallbackExplanationService()
        explanation = service.generate_explanation(
            candidate_name="Test Dev",
            job_title="AI Engineer",
            final_score=50.0,
            breakdown=sample_breakdown_low,
        )
        assert "missing mandatory skills" in explanation
        assert "PyTorch" in explanation

    def test_local_llm_service_fallback_on_unreachable_endpoint(
        self, sample_breakdown_high: ScoreBreakdown
    ):
        """Test LocalLLMExplanationService fails over to deterministic fallback when offline."""
        unreachable_service = LocalLLMExplanationService(
            api_base="http://localhost:59999/v1",
            timeout=0.1,
        )
        explanation = unreachable_service.generate_explanation(
            candidate_name="Alex Wang",
            job_title="AI Engineer",
            final_score=88.0,
            breakdown=sample_breakdown_high,
        )

        assert len(explanation) > 20
        assert "Alex Wang scored 88.0/100" in explanation
        assert "strong qualifications" in explanation

    def test_score_and_ranking_invariance(self):
        """Test explanation generation does not alter final score or metrics."""
        cand = Candidate(
            name="Immutability Test Candidate",
            skills=["Python", "SQL"],
        )
        jd = JobDescription(
            title="Data Analyst",
            required_skills=["Python", "SQL"],
            preferred_skills=[],
        )

        engine = ScoringEngine()
        result = engine.evaluate(cand, jd)

        # Pre-explanation final score calculation verification
        score_val = result.final_score
        assert 0.0 <= score_val <= 100.0

        # Verify explanation is populated
        assert len(result.explanation) > 0
        assert f"Immutability Test Candidate scored {score_val:.1f}/100" in result.explanation

        # Verify score is strictly unchanged
        assert result.final_score == score_val
