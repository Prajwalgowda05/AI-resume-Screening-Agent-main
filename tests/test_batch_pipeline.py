"""Integration tests for BatchScreeningPipeline with 10+ resumes across PDF, DOCX, and TXT."""

from pathlib import Path
import docx
import pytest

from app.engine.embeddings import MockEmbeddingService
from app.engine.pipeline import BatchScreeningPipeline, run_batch_screening
from app.models.ranking import BatchScreeningResult
from tests.test_parsers import create_pdf_with_text


@pytest.fixture
def sample_jd_file(tmp_path: Path) -> Path:
    """Create a temporary job description file."""
    jd_path = tmp_path / "target_job.txt"
    jd_content = """
    Job Title: Junior AI Research Associate

    RESPONSIBILITIES
    - Design and evaluate machine learning models and NLP algorithms.
    - Deploy backend services using Python and FastAPI.

    REQUIREMENTS
    - Bachelor's Degree or Master's Degree in Computer Science or Data Science.
    - Minimum 1+ years of experience in ML and software development.
    - Strong proficiency in Python, PyTorch, and Scikit-Learn.
    - Solid understanding of SQL and Git.

    PREFERRED QUALIFICATIONS
    - Experience with Docker and FastAPI.
    - Familiarity with Large Language Models and Transformers.
    """
    jd_path.write_text(jd_content, encoding="utf-8")
    return jd_path


@pytest.fixture
def ten_plus_resumes_dir(tmp_path: Path) -> Path:
    """Generate 12 diverse candidate resumes in TXT, DOCX, and PDF formats."""
    resumes_dir = tmp_path / "resumes_batch"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    candidates_data = [
        ("Candidate 01 - Priya Sharma", "Python, PyTorch, Scikit-Learn, SQL, Docker, FastAPI", "Master's in Computer Science", "2021 - 2024", "txt"),
        ("Candidate 02 - David Miller", "Python, PyTorch, Scikit-Learn, SQL, Docker", "Bachelor's in Computer Science", "2022 - 2024", "docx"),
        ("Candidate 03 - Chen Wei", "Python, PyTorch, SQL, Machine Learning", "Master's in Data Science", "2023 - 2024", "pdf"),
        ("Candidate 04 - Sarah Connor", "Python, Scikit-Learn, SQL, Git", "Bachelor's in Computer Science", "2020 - 2023", "txt"),
        ("Candidate 05 - Alex Johnson", "Python, PyTorch, FastAPI, Git", "Bachelor's in Artificial Intelligence", "2023 - 2024", "docx"),
        ("Candidate 06 - Elena Rostova", "Python, SQL, Git, Linux", "Master's in Mathematics", "2022 - 2024", "pdf"),
        ("Candidate 07 - Marcus Aurelius", "Python, Machine Learning, Git", "Bachelor's in Software Engineering", "2024 - 2025", "txt"),
        ("Candidate 08 - Fatima Al-Mansoor", "Java, C++, SQL", "Bachelor's in Information Technology", "2021 - 2023", "docx"),
        ("Candidate 09 - Lucas Silva", "JavaScript, React, HTML, CSS", "Bachelor's in Computer Science", "2022 - 2024", "pdf"),
        ("Candidate 10 - Emily Watson", "Python, Docker, Kubernetes", "Master's in Electrical Engineering", "2021 - 2023", "txt"),
        ("Candidate 11 - Rajesh Kumar", "Python, PyTorch, Transformers, BERT", "Ph.D. in Computer Science", "2020 - 2024", "docx"),
        ("Candidate 12 - Chloe Bennett", "Python, Scikit-Learn, Pandas, NumPy", "Bachelor's in Statistics", "2023 - 2025", "pdf"),
    ]

    for name, skills, edu, exp_dates, fmt in candidates_data:
        content = f"""
        {name}
        Email: {name.lower().replace(' ', '_')}@example.com
        Phone: 555-010-9988

        SKILLS
        {skills}

        EDUCATION
        {edu}
        University of Technology, 2024

        EXPERIENCE
        Software Engineer
        Tech Company
        {exp_dates}
        - Built machine learning features and data pipelines.
        """

        if fmt == "txt":
            f_path = resumes_dir / f"{name}.txt"
            f_path.write_text(content, encoding="utf-8")
        elif fmt == "docx":
            f_path = resumes_dir / f"{name}.docx"
            doc = docx.Document()
            for line in content.strip().splitlines():
                if line.strip():
                    doc.add_paragraph(line.strip())
            doc.save(str(f_path))
        elif fmt == "pdf":
            f_path = resumes_dir / f"{name}.pdf"
            create_pdf_with_text(f_path, f"{name} Skills: {skills} Education: {edu}")

    return resumes_dir


class TestBatchScreeningPipeline:
    """Test suite for the batch screening pipeline."""

    def test_batch_pipeline_with_12_resumes(
        self, tmp_path: Path, sample_jd_file: Path, ten_plus_resumes_dir: Path
    ):
        """Test processing 12 mixed-format resumes produces valid ranking outputs."""
        output_dir = tmp_path / "test_outputs"
        pipeline = BatchScreeningPipeline(
            embedding_service=MockEmbeddingService()
        )

        result = pipeline.run(
            job_path=sample_jd_file,
            resumes_dir=ten_plus_resumes_dir,
            output_dir=output_dir,
        )

        assert isinstance(result, BatchScreeningResult)
        assert result.total_files == 12
        assert result.successful_count == 12
        assert result.failed_count == 0
        assert len(result.rankings) == 12

        # Check ranks are 1-12 in strictly descending score order
        ranks = [r.rank for r in result.rankings]
        assert ranks == list(range(1, 13))

        scores = [r.final_score for r in result.rankings]
        assert scores == sorted(scores, reverse=True)

        # Check export files exist
        assert Path(result.csv_path).exists()
        assert Path(result.json_path).exists()

    def test_batch_pipeline_fault_tolerance(
        self, tmp_path: Path, sample_jd_file: Path
    ):
        """Test pipeline gracefully records corrupted files without crashing the remaining batch."""
        resumes_dir = tmp_path / "faulty_resumes"
        resumes_dir.mkdir(parents=True, exist_ok=True)

        # 1 Valid resume
        valid_file = resumes_dir / "valid_cand.txt"
        valid_file.write_text("Candidate One\nSKILLS\nPython, PyTorch", encoding="utf-8")

        # 1 Corrupt PDF file
        corrupt_file = resumes_dir / "corrupted.pdf"
        corrupt_file.write_bytes(b"NOT A VALID PDF CONTENT")

        # 1 Unsupported image file (should be ignored by discover)
        img_file = resumes_dir / "photo.png"
        img_file.write_bytes(b"\x89PNG")

        output_dir = tmp_path / "fault_outputs"
        pipeline = BatchScreeningPipeline(
            embedding_service=MockEmbeddingService()
        )

        result = pipeline.run(
            job_path=sample_jd_file,
            resumes_dir=resumes_dir,
            output_dir=output_dir,
        )

        assert result.total_files == 2  # valid + corrupt (img is skipped by extension filter)
        assert result.successful_count == 1
        assert result.failed_count == 1
        assert len(result.errors) == 1
        assert result.errors[0]["file"] == "corrupted.pdf"
        assert len(result.rankings) == 1

    def test_batch_pipeline_empty_directory(
        self, tmp_path: Path, sample_jd_file: Path
    ):
        """Test pipeline handles empty resumes folder cleanly."""
        empty_dir = tmp_path / "empty_resumes"
        empty_dir.mkdir()

        pipeline = BatchScreeningPipeline(
            embedding_service=MockEmbeddingService()
        )
        result = pipeline.run(
            job_path=sample_jd_file,
            resumes_dir=empty_dir,
            output_dir=tmp_path / "empty_out",
        )

        assert result.total_files == 0
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert result.rankings == []

    def test_batch_pipeline_deterministic_reproducibility(
        self, tmp_path: Path, sample_jd_file: Path, ten_plus_resumes_dir: Path
    ):
        """Test two identical batch runs produce 100% identical rankings and scores."""
        pipeline = BatchScreeningPipeline(
            embedding_service=MockEmbeddingService()
        )

        run_1 = pipeline.run(
            job_path=sample_jd_file,
            resumes_dir=ten_plus_resumes_dir,
            output_dir=tmp_path / "run_1",
        )

        run_2 = pipeline.run(
            job_path=sample_jd_file,
            resumes_dir=ten_plus_resumes_dir,
            output_dir=tmp_path / "run_2",
        )

        scores_1 = [r.final_score for r in run_1.rankings]
        scores_2 = [r.final_score for r in run_2.rankings]
        names_1 = [r.candidate_name for r in run_1.rankings]
        names_2 = [r.candidate_name for r in run_2.rankings]

        assert scores_1 == scores_2
        assert names_1 == names_2
