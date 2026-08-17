"""JSON report exporter for ranked candidate results and batch screening reports."""

import json
from pathlib import Path
from typing import List, Union

from app.models.ranking import BatchScreeningResult, RankedCandidate, RankingResult


def export_ranking_to_json(
    data: Union[BatchScreeningResult, RankingResult, List[RankedCandidate]],
    output_path: Union[str, Path] = "outputs/ranking.json",
) -> Path:
    """Export ranking results or batch screening reports to structured JSON.

    Args:
        data: BatchScreeningResult, RankingResult, or list of RankedCandidate objects.
        output_path: Target destination path for the JSON file.

    Returns:
        Resolved Path to the created JSON file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, (BatchScreeningResult, RankingResult)):
        json_content = data.model_dump_json(indent=2)
    elif isinstance(data, list):
        # List of RankedCandidate models
        dict_list = [c.model_dump() for c in data]
        json_content = json.dumps(dict_list, indent=2)
    else:
        json_content = json.dumps(data, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_content)

    return path
