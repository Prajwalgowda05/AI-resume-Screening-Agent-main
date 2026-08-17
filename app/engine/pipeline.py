"""Batch screening pipeline orchestrating document parsing, extraction, scoring, ranking, and export."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from app.engine.embeddings import BaseEmbeddingService
from app.engine.ranker import CandidateRanker
from app.engine.scorer import ScoringEngine
from app.extractors.jd_extractor import extract_job_description
from app.extractors.resume_extractor import extract_resume
from app.exporters.csv_exporter import export_ranking_to_csv
from app.exporters.json_exporter import export_ranking_to_json
from app.models.ranking import (
    BatchScreeningResult,
    MatchResult,
    RankedCandidate,
    ScoringWeights,
)
from app.parsers import PARSER_REGISTRY


class BatchScreeningPipeline:
    """End-to-end batch screening service evaluating multiple resumes against a target job description."""

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
    ):
        """Initialize the batch screening pipeline."""
        self.scoring_engine = ScoringEngine(
            weights=weights, embedding_service=embedding_service
        )
        self.ranker = CandidateRanker()

    def discover_resume_files(
        self, resumes_dir: Union[str, Path]
    ) -> List[Path]:
        """Discover all supported resume documents (PDF, DOCX, TXT) in a directory.

        Args:
            resumes_dir: Directory containing candidate resumes.

        Returns:
            Sorted list of resolved Path objects.
        """
        dir_path = Path(resumes_dir).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

        supported_extensions: Set[str] = set(PARSER_REGISTRY.keys())
        discovered_files: List[Path] = []

        for item in dir_path.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                discovered_files.append(item)

        # Sort for deterministic processing order
        return sorted(discovered_files, key=lambda p: p.name.lower())

    def run(
        self,
        job_path: Union[str, Path],
        resumes_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = "outputs",
        top_n: Optional[int] = None,
    ) -> BatchScreeningResult:
        """Execute batch resume screening against a job description.

        Args:
            job_path: Path to the target Job Description document.
            resumes_dir: Directory containing candidate resumes.
            output_dir: Optional output directory for CSV and JSON exports.
            top_n: Optional limit to top N candidates.

        Returns:
            BatchScreeningResult container.
        """
        # 1. Parse and extract target Job Description
        jd_file = Path(job_path).resolve()
        if not jd_file.exists():
            raise FileNotFoundError(f"Job description file not found: {job_path}")

        jd = extract_job_description(jd_file)

        # 2. Discover resume files
        resume_files = self.discover_resume_files(resumes_dir)

        match_results: List[MatchResult] = []
        errors: List[Dict[str, str]] = []

        # 3. Process each resume with fault tolerance
        for resume_path in resume_files:
            try:
                candidate = extract_resume(resume_path)
                match_result = self.scoring_engine.evaluate(candidate, jd)
                match_results.append(match_result)
            except Exception as e:
                errors.append(
                    {"file": resume_path.name, "error": str(e)}
                )

        # 4. Rank candidates deterministically
        ranked_candidates: List[RankedCandidate] = self.ranker.rank(
            match_results, top_n=top_n
        )

        csv_path_str: Optional[str] = None
        json_path_str: Optional[str] = None

        # 5. Export results if output directory is configured
        if output_dir:
            out_dir = Path(output_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            csv_file = out_dir / "ranking.csv"
            json_file = out_dir / "ranking.json"

            export_ranking_to_csv(ranked_candidates, csv_file)

            batch_result = BatchScreeningResult(
                job_title=jd.title,
                total_files=len(resume_files),
                successful_count=len(match_results),
                failed_count=len(errors),
                rankings=ranked_candidates,
                errors=errors,
                csv_path=str(csv_file),
                json_path=str(json_file),
            )

            export_ranking_to_json(batch_result, json_file)
            return batch_result

        return BatchScreeningResult(
            job_title=jd.title,
            total_files=len(resume_files),
            successful_count=len(match_results),
            failed_count=len(errors),
            rankings=ranked_candidates,
            errors=errors,
        )


def run_batch_screening(
    job_path: Union[str, Path],
    resumes_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = "outputs",
    top_n: Optional[int] = None,
    weights: Optional[ScoringWeights] = None,
    embedding_service: Optional[BaseEmbeddingService] = None,
) -> BatchScreeningResult:
    """Convenience function to run the complete batch screening pipeline.

    Args:
        job_path: Path to target Job Description file.
        resumes_dir: Directory containing candidate resumes.
        output_dir: Output directory for ranking exports.
        top_n: Optional top N candidate limit.
        weights: Optional custom ScoringWeights.
        embedding_service: Optional custom embedding service.

    Returns:
        BatchScreeningResult container.
    """
    pipeline = BatchScreeningPipeline(
        weights=weights, embedding_service=embedding_service
    )
    return pipeline.run(
        job_path=job_path,
        resumes_dir=resumes_dir,
        output_dir=output_dir,
        top_n=top_n,
    )
