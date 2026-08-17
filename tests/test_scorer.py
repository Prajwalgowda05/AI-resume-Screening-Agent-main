"""Unit and integration tests for ScoringEngine and multi-factor evaluation."""

import pytest
from app.engine.embeddings import MockEmbeddingService
from app.engine.scorer import ScoringEngine, evaluate_candidate
from app.models.candidate import Candidate, Education, Experience
from app.models.job_description import JobDescription
from app.models.ranking import MatchResult, ScoringWeights


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Provide deterministic mock embedding service for fast, isolated tests."""
    return MockEmbeddingService()


@pytest.fixture
def scoring_engine(mock_embedding_service: MockEmbeddingService) -> ScoringEngine:
    """Provide ScoringEngine with mock embedding service."""
    return ScoringEngine(embedding_service=mock_embedding_service)


@pytest.fixture
def sample_candidate() -> Candidate:
    """Fixture for sample candidate."""
    return Candidate(
        name="Aditya Sharma",
        email="aditya@example.com",
        phone="+91 9876543210",
        skills=["Python", "PyTorch", "Scikit-Learn", "SQL", "FastAPI", "Docker"],
        education=[
            Education(
                degree="Bachelor's Degree",
                field_of_study="Computer Science",
                institution="ABC Tech",
                start_year=2022,
                end_year=2026,
            )
        ],
        experience=[
            Experience(
                title="Software Engineering Intern",
                company="Tech Solutions",
                start_date="Jan 2025",
                end_date="Jun 2025",
                description="Built REST APIs using Python and FastAPI. Worked with Docker.",
            )
        ],
    )


@pytest.fixture
def sample_jd() -> JobDescription:
    """Fixture for sample job description."""
    return JobDescription(
        title="Junior AI Research Associate",
        required_skills=["Python", "PyTorch", "Scikit-Learn", "SQL"],
        preferred_skills=["Docker", "FastAPI", "Kubernetes"],
        minimum_experience_years=0.5,
        education_requirements=["Bachelor's Degree", "Degree in Computer Science"],
        responsibilities=[
            "Train and evaluate machine learning models.",
            "Deploy backend REST APIs.",
        ],
    )


class TestScoringEngine:
    """Test suite for multi-factor scoring calculations and edge cases."""

    def test_required_skills_score(
        self, scoring_engine: ScoringEngine, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test exact match on all 4 required skills yields 100%."""
        score, matched, missing = scoring_engine.compute_required_skills_score(
            sample_candidate, sample_jd
        )
        assert score == 100.0
        assert len(matched) == 4
        assert missing == []

    def test_partial_required_skills_score(
        self, scoring_engine: ScoringEngine, sample_jd: JobDescription
    ):
        """Test partial match yields proportional score."""
        candidate = Candidate(name="Partial Skills", skills=["Python", "SQL"])
        score, matched, missing = scoring_engine.compute_required_skills_score(
            candidate, sample_jd
        )
        assert score == 50.0
        assert len(matched) == 2
        assert set(missing) == {"PyTorch", "Scikit-Learn"}

    def test_preferred_skills_score(
        self, scoring_engine: ScoringEngine, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test matching 2 out of 3 preferred skills yields 66.67%."""
        score, matched = scoring_engine.compute_preferred_skills_score(
            sample_candidate, sample_jd
        )
        assert round(score, 1) == 66.7
        assert "Docker" in matched
        assert "FastAPI" in matched

    def test_experience_score_calculation(
        self, scoring_engine: ScoringEngine, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test experience score reaches 100% when meeting/exceeding minimum required years."""
        score, cand_years = scoring_engine.compute_experience_score(
            sample_candidate, sample_jd
        )
        assert score == 100.0
        assert cand_years >= 0.4

    def test_experience_score_capping(
        self, scoring_engine: ScoringEngine, sample_jd: JobDescription
    ):
        """Test experience score does not exceed 100% when candidate has excess experience."""
        experienced_candidate = Candidate(
            name="Senior Dev",
            experience=[
                Experience(
                    title="Engineer",
                    start_date="2018",
                    end_date="2024",
                )
            ],
        )
        score, cand_years = scoring_engine.compute_experience_score(
            experienced_candidate, sample_jd
        )
        assert score == 100.0
        assert cand_years >= 5.0

    def test_education_score_matching(
        self, scoring_engine: ScoringEngine, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test education scoring with degree level and discipline match."""
        score = scoring_engine.compute_education_score(
            sample_candidate, sample_jd
        )
        assert score == 100.0

    def test_education_score_partial_credit(self, scoring_engine: ScoringEngine):
        """Test partial credit when candidate holds Bachelor's but JD requests Master's."""
        jd_master = JobDescription(
            title="Senior Scientist",
            education_requirements=["Master's Degree", "Degree in Computer Science"],
        )
        cand_bachelor = Candidate(
            name="Bachelor Candidate",
            education=[
                Education(
                    degree="Bachelor's Degree",
                    field_of_study="Computer Science",
                )
            ],
        )
        score = scoring_engine.compute_education_score(
            cand_bachelor, jd_master
        )
        assert score == 70.0

    def test_dynamic_weight_redistribution_no_preferred_skills(
        self, scoring_engine: ScoringEngine, sample_candidate: Candidate
    ):
        """Test that when JD has no preferred skills, weight redistributes to required skills."""
        jd_no_pref = JobDescription(
            title="Python Dev",
            required_skills=["Python", "SQL"],
            preferred_skills=[],
            minimum_experience_years=0.5,
            education_requirements=["Bachelor's Degree"],
        )
        result = scoring_engine.evaluate(sample_candidate, jd_no_pref)
        assert result.breakdown.preferred_skill_score == 100.0
        assert result.final_score > 0.0

    def test_custom_scoring_weights(
        self, mock_embedding_service: MockEmbeddingService, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test overriding weights via ScoringWeights."""
        custom_weights = ScoringWeights(
            required_skills=0.50,
            preferred_skills=0.10,
            semantic_similarity=0.20,
            experience=0.15,
            education=0.05,
        )
        engine = ScoringEngine(
            weights=custom_weights, embedding_service=mock_embedding_service
        )
        result = engine.evaluate(sample_candidate, sample_jd)
        assert isinstance(result, MatchResult)
        assert 0.0 <= result.final_score <= 100.0

    def test_invalid_scoring_weights_raise_validation_error(self):
        """Test that weights not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError):
            ScoringWeights(
                required_skills=0.70,
                preferred_skills=0.70,
                semantic_similarity=0.20,
                experience=0.15,
                education=0.05,
            )

    def test_edge_case_empty_candidate(
        self, scoring_engine: ScoringEngine, sample_jd: JobDescription
    ):
        """Test evaluating a completely blank candidate profile without crashing."""
        blank_candidate = Candidate(name="Blank Candidate")
        result = scoring_engine.evaluate(blank_candidate, sample_jd)

        assert result.breakdown.required_skill_score == 0.0
        assert result.breakdown.missing_required_skills == sample_jd.required_skills
        assert result.breakdown.experience_score == 0.0
        assert result.breakdown.education_score == 0.0
        assert 0.0 <= result.final_score <= 100.0

    def test_end_to_end_evaluate_candidate_helper(
        self, sample_candidate: Candidate, sample_jd: JobDescription
    ):
        """Test convenience helper evaluate_candidate."""
        result = evaluate_candidate(sample_candidate, sample_jd)
        assert isinstance(result, MatchResult)
        assert result.candidate_name == "Aditya Sharma"
        assert result.job_title == "Junior AI Research Associate"
        assert result.final_score > 70.0
        assert "Aditya Sharma scored" in result.summary
