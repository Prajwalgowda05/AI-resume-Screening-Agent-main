# AI Resume Screening Agent

An automated, explainable resume screening system that screens batches of candidate resumes against a job description and produces a deterministic, ranked shortlist with grounded explanations. Outputs are exported as CSV and JSON reports.

---

## Problem

Hiring pipelines require screening large volumes of resumes against specific job requirements. Manual screening is time-consuming and inconsistent. This agent automates multi-factor candidate evaluation by combining explicit skill matching with semantic similarity, producing transparent and reproducible rankings.

---

## Pipeline

```
Job Description + 10+ Resumes
        │
        ▼
Document Parsing (PDF, DOCX, TXT)
        │
        ▼
Resume / JD Structured Extraction
        │
        ▼
Explicit Skill Matching
        │
        ▼
Semantic Similarity (all-MiniLM-L6-v2)
        │
        ▼
Deterministic Composite Scoring
        │
        ▼
Candidate Ranking
        │
        ▼
Grounded Explanation Generation
        │
        ▼
CSV + JSON Export
```

---

## Supported Document Formats

- **PDF** — via `pypdf`
- **DOCX** — via `python-docx`
- **TXT** — plain text with multi-encoding fallback

---

## Scoring Methodology

Each candidate receives a composite score (0–100) computed from five weighted factors:

| Factor              | Weight |
| :------------------ | -----: |
| Required Skills     |    40% |
| Preferred Skills    |    20% |
| Semantic Similarity |    20% |
| Experience          |    15% |
| Education           |     5% |

**Key properties:**

- **Numerical scoring is deterministic.** Given the same inputs, the same scores are always produced.
- **Ranking is deterministic.** Tie-breaking uses required skill score, semantic similarity, then alphabetical candidate name.
- **The explanation layer does not modify scores or rankings.** Explanations are generated after scoring is complete and are strictly read-only with respect to evaluation results.

If a job description specifies no preferred skills, the preferred skills weight (20%) is dynamically redistributed to required skills.

---

## NLP / Embeddings

The agent uses [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~80 MB) to compute dense 384-dimensional sentence embeddings. Cosine similarity between structured candidate and job description semantic profiles provides a measure of contextual alignment beyond exact keyword matching.

This model runs locally with no external API calls, rate limits, or costs.

---

## Explanation Layer

The system generates concise, evidence-grounded 2–4 sentence narrative explanations for every candidate. Explanations reference only structured evidence from the evaluation: matched skills, missing skills, experience duration, and component scores.

- **Deterministic fallback:** A rule-based template engine always produces explanations, with no external dependencies.
- **Optional local LLM enhancement:** If a local LLM endpoint (e.g., Ollama) is configured via `LLM_API_BASE`, the system will attempt to use it for richer explanations. If the endpoint is unavailable, the deterministic fallback is used automatically.

An LLM is **not** required for the core screening pipeline. All scoring, matching, ranking, and explanation generation work fully offline.

---

## Installation

### Prerequisites

- Python 3.10+

### Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment (Optional)

```bash
cp .env.example .env
```

---

## Usage

### Run Batch Screening

```bash
python -m app.main --job data/job_descriptions/sample_job_description.txt --resumes data/resumes --output outputs
```

**CLI Options:**

| Option           | Description                                          | Default                                             |
| :--------------- | :--------------------------------------------------- | :-------------------------------------------------- |
| `--job` / `-j`   | Path to Job Description file (`.pdf`, `.docx`, `.txt`) | `data/job_descriptions/sample_job_description.txt` |
| `--resumes` / `-r` | Directory containing candidate resumes             | `data/resumes`                                      |
| `--output` / `-o`  | Output directory for CSV and JSON reports           | `outputs`                                           |
| `--top-n` / `-n`   | Limit output to top N ranked candidates             | All candidates                                      |

### Run Tests

```bash
python -m pytest
```

Current verified result: **79 tests passing**.

---

## Example Result

Running the demo pipeline against 11 synthetic resumes:

```
 1. Priya Sharma    — 91.5/100
 2. Marcus Vance    — 79.7/100
 3. Rajesh Kumar    — 77.6/100
 4. David Miller    — 73.0/100
 5. Alex Chen       — 62.2/100
 6. Sarah Connor    — 59.7/100
 7. Elena Rostova   — 52.1/100
 8. Emily Watson    — 51.9/100
 9. Aditya Sharma   — 46.5/100
10. Fatima Mansoor  — 46.3/100
11. Lucas Silva     — 36.8/100
```

The generated `ranking.csv` and `ranking.json` contain full score breakdowns, matched and missing skills, and narrative explanations for each candidate.

---

## Architecture

```
ai-resume-screening-agent/
├── app/
│   ├── __init__.py            # Package version
│   ├── config.py              # Environment config (python-dotenv)
│   ├── main.py                # CLI entry point (argparse)
│   ├── parsers/               # Document parsers
│   │   ├── base.py            #   Abstract parser + exceptions
│   │   ├── pdf_parser.py      #   PDF text extraction (pypdf)
│   │   ├── docx_parser.py     #   DOCX text extraction (python-docx)
│   │   └── text_parser.py     #   Plain text reader
│   ├── extractors/            # Structured information extraction
│   │   ├── resume_extractor.py    # Candidate profile extraction
│   │   └── jd_extractor.py        # Job description extraction
│   ├── models/                # Pydantic data schemas
│   │   ├── candidate.py       #   Candidate, Education, Experience
│   │   ├── job_description.py #   JobDescription
│   │   └── ranking.py         #   MatchResult, ScoreBreakdown, RankedCandidate, BatchScreeningResult
│   ├── engine/                # Core screening engine
│   │   ├── embeddings.py      #   Embedding services (SentenceTransformer + mock)
│   │   ├── matcher.py         #   Explicit skill matching + semantic text builders
│   │   ├── scorer.py          #   Multi-factor composite scoring
│   │   ├── ranker.py          #   Deterministic ranking + tie-breaking
│   │   ├── explanation.py     #   Grounded explanation generation
│   │   └── pipeline.py        #   Batch screening orchestration
│   └── exporters/             # Report generation
│       ├── csv_exporter.py    #   CSV export
│       └── json_exporter.py   #   JSON export
├── data/
│   ├── job_descriptions/      # Target job descriptions
│   └── resumes/               # Candidate resumes (11 synthetic samples)
├── tests/                     # 79 automated tests
├── outputs/                   # Generated ranking reports (gitignored)
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variable template
```

---

## Limitations

- **Rule-based extraction** can struggle with highly unconventional resume layouts, non-standard section headers, or multi-column formatting.
- **Semantic similarity** captures contextual alignment but is not a replacement for human judgment in nuanced hiring decisions.
- **Synthetic demo data** is used for validation. Real-world resume diversity may surface edge cases in extraction.
- **Explanations are grounded in structured evidence** (matched skills, experience, scores) and do not introduce external information, but they summarize rather than deeply analyze candidate fit.
- The system does not claim perfect accuracy. It is a screening aid, not an autonomous hiring decision-maker.
