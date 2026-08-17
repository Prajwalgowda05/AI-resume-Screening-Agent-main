"""Unit and integration tests for CandidateRanker and ranking models."""

import pytest
from app.engine.embeddings import MockEmbeddingService
from app.engine.ranker import CandidateRanker, create_ranking_result, rank_candidates
from app.engine.scorer import ScoringEngine
from app.models.candidate import Candidate, Education, Experience
from app.models.job_description import JobDescription
from app.models.ranking import MatchResult, RankedCandidate, RankingResult, ScoreBreakdown


def create_sample_match_result(
    candidate_name: str,
    final_score: float,
    required_skill_score: float = 80.0,
    semantic_similarity_score: float = 75.0,
    job_title: str = "Junior AI Research Associate",
) -> MatchResult:
    """Helper to create a MatchResult with specific scores for testing."""
    breakdown = ScoreBreakdown(
        required_skill_score=required_skill_score,
        preferred_skill_score=50.0,
        semantic_similarity_score=semantic_similarity_score,
        experience_score=80.0,
        education_score=100.0,
        matched_required_skills=["Python", "PyTorch"],
        missing_required_skills=["SQL"],
        matched_preferred_skills=["Docker"],
        candidate_experience_years=2.0,
        required_experience_years=1.0,
    )
    return MatchResult(
        candidate_name=candidate_name,
        job_title=job_title,
        final_score=final_score,
        breakdown=breakdown,
        summary=f"{candidate_name} scored {final_score:.1f}/100.",
    )


class TestCandidateRanker:
    """Test suite for ranking logic, ordering, tie-breaking, and top-N filtering."""

    def test_rank_multiple_candidates_descending_order(self):
        """Test candidates are ordered strictly from highest final score to lowest."""
        res_a = create_sample_match_result("Candidate A", 87.4)
        res_b = create_sample_match_result("Candidate B", 93.1)
        res_c = create_sample_match_result("Candidate C", 72.8)

        ranked = rank_candidates([res_a, res_b, res_c])

        assert len(ranked) == 3
        assert [r.candidate_name for r in ranked] == [
            "Candidate B",
            "Candidate A",
            "Candidate C",
        ]
        assert [r.final_score for r in ranked] == [93.1, 87.4, 72.8]
        assert [r.rank for r in ranked] == [1, 2, 3]

    def test_rank_numbers_are_one_indexed(self):
        """Test rank positions start at 1 and increment continuously."""
        results = [
            create_sample_match_result(f"Candidate {i}", float(i * 10))
            for i in range(1, 6)
        ]
        ranked = rank_candidates(results)
        assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]
        assert ranked[0].final_score == 50.0

    def test_tie_breaking_required_skill_score(self):
        """Test tie-breaking on identical final score favors higher required skill score."""
        cand_1 = create_sample_match_result(
            "Cand 1", final_score=85.0, required_skill_score=90.0
        )
        cand_2 = create_sample_match_result(
            "Cand 2", final_score=85.0, required_skill_score=80.0
        )

        ranked = rank_candidates([cand_2, cand_1])
        assert ranked[0].candidate_name == "Cand 1"
        assert ranked[1].candidate_name == "Cand 2"
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_tie_breaking_semantic_similarity(self):
        """Test tie-breaking on equal score and required skills favors higher semantic similarity."""
        cand_1 = create_sample_match_result(
            "Cand 1",
            final_score=85.0,
            required_skill_score=80.0,
            semantic_similarity_score=88.0,
        )
        cand_2 = create_sample_match_result(
            "Cand 2",
            final_score=85.0,
            required_skill_score=80.0,
            semantic_similarity_score=75.0,
        )

        ranked = rank_candidates([cand_2, cand_1])
        assert ranked[0].candidate_name == "Cand 1"
        assert ranked[1].candidate_name == "Cand 2"

    def test_tie_breaking_alphabetical_name(self):
        """Test final tie-breaker sorts candidate names alphabetically."""
        cand_bob = create_sample_match_result(
            "Bob",
            final_score=85.0,
            required_skill_score=80.0,
            semantic_similarity_score=75.0,
        )
        cand_alice = create_sample_match_result(
            "Alice",
            final_score=85.0,
            required_skill_score=80.0,
            semantic_similarity_score=75.0,
        )

        ranked = rank_candidates([cand_bob, cand_alice])
        assert ranked[0].candidate_name == "Alice"
        assert ranked[1].candidate_name == "Bob"

    def test_top_n_filtering(self):
        """Test top_n limit returns only the requested number of top candidates."""
        results = [
            create_sample_match_result(f"Candidate {i}", float(i * 10))
            for i in range(1, 6)
        ]
        ranked_top_2 = rank_candidates(results, top_n=2)
        assert len(ranked_top_2) == 2
        assert ranked_top_2[0].rank == 1
        assert ranked_top_2[0].final_score == 50.0
        assert ranked_top_2[1].rank == 2
        assert ranked_top_2[1].final_score == 40.0

    def test_top_n_exceeding_length(self):
        """Test top_n larger than candidate count returns all candidates without error."""
        results = [
            create_sample_match_result("A", 80.0),
            create_sample_match_result("B", 90.0),
        ]
        ranked = rank_candidates(results, top_n=10)
        assert len(ranked) == 2
        assert ranked[0].candidate_name == "B"
        assert ranked[1].candidate_name == "A"

    def test_invalid_top_n_raises_error(self):
        """Test non-positive top_n raises ValueError."""
        results = [create_sample_match_result("A", 80.0)]
        with pytest.raises(ValueError):
            rank_candidates(results, top_n=0)
        with pytest.raises(ValueError):
            rank_candidates(results, top_n=-5)

    def test_empty_candidate_list(self):
        """Test ranking an empty candidate list returns empty list."""
        ranked = rank_candidates([])
        assert ranked == []

    def test_single_candidate(self):
        """Test ranking a single candidate returns 1-item list with rank 1."""
        res = create_sample_match_result("Solo Candidate", 82.5)
        ranked = rank_candidates([res])
        assert len(ranked) == 1
        assert ranked[0].rank == 1
        assert ranked[0].candidate_name == "Solo Candidate"
        assert ranked[0].final_score == 82.5

    def test_preservation_of_match_details(self):
        """Test all MatchResult fields and shortcuts are preserved on RankedCandidate."""
        res = create_sample_match_result("Jane Doe", 91.0)
        ranked = rank_candidates([res])[0]

        assert ranked.candidate_name == "Jane Doe"
        assert ranked.final_score == 91.0
        assert ranked.matched_required_skills == ["Python", "PyTorch"]
        assert ranked.missing_required_skills == ["SQL"]
        assert ranked.matched_preferred_skills == ["Docker"]
        assert "Jane Doe scored" in ranked.summary

    def test_create_ranking_result_container(self):
        """Test create_ranking_result helper returns valid RankingResult container."""
        results = [
            create_sample_match_result("Alice", 85.0),
            create_sample_match_result("Bob", 92.0),
        ]
        result_container = create_ranking_result(
            results, job_title="AI Researcher", top_n=1
        )
        assert isinstance(result_container, RankingResult)
        assert result_container.job_title == "AI Researcher"
        assert result_container.total_candidates == 2
        assert len(result_container.rankings) == 1
        assert result_container.rankings[0].candidate_name == "Bob"

    def test_integration_rank_evaluated_candidates(self):
        """End-to-end integration test evaluating and ranking multiple realistic candidates."""
        jd = JobDescription(
            title="Junior AI Research Associate",
            required_skills=["Python", "PyTorch", "Scikit-Learn", "SQL"],
            preferred_skills=["Docker", "FastAPI"],
            minimum_experience_years=1.0,
            education_requirements=["Bachelor's Degree", "Degree in Computer Science"],
        )

        cand_strong = Candidate(
            name="Strong Candidate",
            skills=["Python", "PyTorch", "Scikit-Learn", "SQL", "Docker", "FastAPI"],
            education=[
                Education(
                    degree="Master's Degree",
                    field_of_study="Computer Science",
                    institution="Stanford",
                    end_year=2023,
                )
            ],
            experience=[
                Experience(
                    title="ML Engineer",
                    company="Tech Corp",
                    start_date="2021",
                    end_date="2023",
                    description="Trained deep learning models with PyTorch.",
                )
            ],
        )

        cand_moderate = Candidate(
            name="Moderate Candidate",
            skills=["Python", "SQL", "Docker"],
            education=[
                Education(
                    degree="Bachelor's Degree",
                    field_of_study="Computer Science",
                    institution="State University",
                    end_year=2024,
                )
            ],
            experience=[
                Experience(
                    title="Intern",
                    company="Startup",
                    start_date="Jan 2024",
                    end_date="Jun 2024",
                    description="Built SQL queries and Python scripts.",
                )
            ],
        )

        cand_weak = Candidate(
            name="Weak Candidate",
            skills=["Java", "C++"],
            education=[],
            experience=[],
        )

        engine = ScoringEngine(embedding_service=MockEmbeddingService())
        results = [
            engine.evaluate(cand_moderate, jd),
            engine.evaluate(cand_strong, jd),
            engine.evaluate(cand_weak, jd),
        ]

        ranked = rank_candidates(results)

        assert len(ranked) == 3
        assert ranked[0].candidate_name == "Strong Candidate"
        assert ranked[0].rank == 1
        assert ranked[1].candidate_name == "Moderate Candidate"
        assert ranked[1].rank == 2
        assert ranked[2].candidate_name == "Weak Candidate"
        assert ranked[2].rank == 3
        assert ranked[0].final_score > ranked[1].final_score > ranked[2].final_score
