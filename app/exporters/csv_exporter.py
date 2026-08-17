"""CSV report exporter for ranked candidate results."""

import csv
from pathlib import Path
from typing import List, Union

from app.models.ranking import RankedCandidate


def export_ranking_to_csv(
    rankings: List[RankedCandidate],
    output_path: Union[str, Path] = "outputs/ranking.csv",
) -> Path:
    """Export ranked candidate list to a structured CSV report.

    Args:
        rankings: List of RankedCandidate objects.
        output_path: Target destination path for the CSV file.

    Returns:
        Resolved Path to the created CSV file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "rank",
        "candidate_name",
        "final_score",
        "required_skill_score",
        "preferred_skill_score",
        "semantic_similarity_score",
        "experience_score",
        "education_score",
        "matched_required_skills",
        "missing_required_skills",
        "matched_preferred_skills",
        "explanation",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for candidate in rankings:
            bd = candidate.breakdown
            row = [
                candidate.rank,
                candidate.candidate_name,
                f"{candidate.final_score:.2f}",
                f"{bd.required_skill_score:.2f}",
                f"{bd.preferred_skill_score:.2f}",
                f"{bd.semantic_similarity_score:.2f}",
                f"{bd.experience_score:.2f}",
                f"{bd.education_score:.2f}",
                "; ".join(bd.matched_required_skills),
                "; ".join(bd.missing_required_skills),
                "; ".join(bd.matched_preferred_skills),
                candidate.explanation,
            ]
            writer.writerow(row)

    return path
