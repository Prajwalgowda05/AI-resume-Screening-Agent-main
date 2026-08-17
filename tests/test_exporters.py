"""Unit tests for CSV and JSON report exporters."""

import csv
import json
from pathlib import Path
import pytest

from app.exporters import export_ranking_to_csv, export_ranking_to_json
from app.models.ranking import (
    BatchScreeningResult,
    RankedCandidate,
    RankingResult,
    ScoreBreakdown,
)


@pytest.fixture
def sample_ranked_candidates() -> list[RankedCandidate]:
    """Fixture with two sample ranked candidates."""
    bd1 = ScoreBreakdown(
        required_skill_score=100.0,
        preferred_skill_score=66.67,
        semantic_similarity_score=85.0,
        experience_score=100.0,
        education_score=100.0,
        matched_required_skills=["Python", "PyTorch", "SQL"],
        missing_required_skills=[],
        matched_preferred_skills=["Docker", "FastAPI"],
        candidate_experience_years=3.0,
        required_experience_years=1.0,
    )
    c1 = RankedCandidate(
        rank=1,
        candidate_name="Alice Wang",
        job_title="Junior AI Research Associate",
        final_score=92.5,
        breakdown=bd1,
        summary="Alice Wang scored 92.5/100.",
        explanation="Alice Wang scored 92.5/100 demonstrating strong alignment. Key strengths include Python and PyTorch.",
    )

    bd2 = ScoreBreakdown(
        required_skill_score=66.67,
        preferred_skill_score=33.33,
        semantic_similarity_score=72.0,
        experience_score=50.0,
        education_score=70.0,
        matched_required_skills=["Python", "SQL"],
        missing_required_skills=["PyTorch"],
        matched_preferred_skills=["Docker"],
        candidate_experience_years=0.5,
        required_experience_years=1.0,
    )
    c2 = RankedCandidate(
        rank=2,
        candidate_name="Bob Smith",
        job_title="Junior AI Research Associate",
        final_score=64.8,
        breakdown=bd2,
        summary="Bob Smith scored 64.8/100.",
        explanation="Bob Smith achieved a moderate score of 64.8/100. Missing required skill in PyTorch.",
    )
    return [c1, c2]


class TestExporters:
    """Test suite for CSV and JSON export routines."""

    def test_export_ranking_to_csv(
        self, tmp_path: Path, sample_ranked_candidates: list[RankedCandidate]
    ):
        """Test exporting ranked candidates to CSV and verify columns and rows."""
        csv_file = tmp_path / "outputs" / "ranking.csv"
        result_path = export_ranking_to_csv(sample_ranked_candidates, csv_file)

        assert result_path.exists()
        with open(result_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # Check headers
        headers = reader[0]
        assert "rank" in headers
        assert "candidate_name" in headers
        assert "final_score" in headers
        assert "matched_required_skills" in headers
        assert "explanation" in headers

        # Check row data
        assert len(reader) == 3  # header + 2 candidates
        assert reader[1][0] == "1"
        assert reader[1][1] == "Alice Wang"
        assert reader[1][2] == "92.50"
        assert "Python; PyTorch; SQL" in reader[1][8]
        assert "demonstrating strong alignment" in reader[1][11]

        assert reader[2][0] == "2"
        assert reader[2][1] == "Bob Smith"

    def test_export_ranking_to_json(
        self, tmp_path: Path, sample_ranked_candidates: list[RankedCandidate]
    ):
        """Test exporting BatchScreeningResult to JSON and verify structure."""
        json_file = tmp_path / "outputs" / "ranking.json"
        batch_result = BatchScreeningResult(
            job_title="Junior AI Research Associate",
            total_files=2,
            successful_count=2,
            failed_count=0,
            rankings=sample_ranked_candidates,
            errors=[],
        )

        result_path = export_ranking_to_json(batch_result, json_file)
        assert result_path.exists()

        with open(result_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["job_title"] == "Junior AI Research Associate"
        assert data["total_files"] == 2
        assert len(data["rankings"]) == 2
        assert data["rankings"][0]["candidate_name"] == "Alice Wang"
        assert data["rankings"][0]["final_score"] == 92.5
        assert "demonstrating strong alignment" in data["rankings"][0]["explanation"]
