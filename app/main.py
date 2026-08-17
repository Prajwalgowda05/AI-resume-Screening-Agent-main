"""Main CLI entry point for AI Resume Screening Agent."""

import argparse
from pathlib import Path
import sys
from typing import Optional

from app.config import (
    APP_ENV,
    DATA_DIR,
    JOB_DESCRIPTIONS_DIR,
    LOG_LEVEL,
    OUTPUTS_DIR,
    RESUMES_DIR,
)
from app.engine.pipeline import run_batch_screening


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Resume Screening Agent - Automated multi-factor candidate screening and ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default=str(JOB_DESCRIPTIONS_DIR / "sample_job_description.txt"),
        help="Path to the target Job Description file (.pdf, .docx, .txt)",
    )
    parser.add_argument(
        "--resumes",
        "-r",
        type=str,
        default=str(RESUMES_DIR),
        help="Directory containing candidate resumes (.pdf, .docx, .txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Output directory for generated CSV and JSON ranking reports",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=None,
        help="Optional limit to top N ranked candidates",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main execution entry point."""
    if args is None:
        args = parse_args()

    job_file = Path(args.job)
    resumes_path = Path(args.resumes)
    output_path = Path(args.output)

    print("=" * 65)
    print("       AI Resume Screening Agent - Batch Pipeline")
    print("=" * 65)
    print(f"Target Job  : {job_file.name if job_file.exists() else str(job_file)}")
    print(f"Resumes Dir : {resumes_path}")
    print(f"Outputs Dir : {output_path}")
    print("-" * 65)

    if not job_file.exists():
        print("Status      : Ready. System foundation initialized.")
        print(f"Note        : Job description file '{job_file}' not found.")
        print("              Provide a valid JD path via --job <path> to run screening.")
        print("=" * 65)
        return 0

    if not resumes_path.exists():
        print(f"Status      : Ready. Resumes directory '{resumes_path}' not found.")
        print("=" * 65)
        return 0

    try:
        batch_result = run_batch_screening(
            job_path=job_file,
            resumes_dir=resumes_path,
            output_dir=output_path,
            top_n=args.top_n,
        )

        print(f"Processed   : {batch_result.total_files} file(s)")
        print(f"Successful  : {batch_result.successful_count}")
        print(f"Failed      : {batch_result.failed_count}")
        print("-" * 65)

        if batch_result.rankings:
            print("Top Ranked Candidates:")
            for cand in batch_result.rankings:
                bd = cand.breakdown
                print(
                    f"  {cand.rank:2d}. {cand.candidate_name:<25} - {cand.final_score:5.1f}/100 "
                    f"[Skills: {bd.required_skill_score:4.1f}%, Semantic: {bd.semantic_similarity_score:4.1f}%, "
                    f"Exp: {bd.experience_score:4.1f}%]"
                )
        else:
            print("No candidate resumes were successfully ranked.")

        if batch_result.errors:
            print("-" * 65)
            print("Processing Errors:")
            for err in batch_result.errors:
                print(f"  - {err.get('file', 'Unknown')}: {err.get('error', 'Error')}")

        print("-" * 65)
        print("Generated Reports:")
        if batch_result.csv_path:
            print(f"  CSV : {batch_result.csv_path}")
        if batch_result.json_path:
            print(f"  JSON: {batch_result.json_path}")
        print("=" * 65)
        return 0

    except Exception as e:
        print(f"Pipeline Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
