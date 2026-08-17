"""Core screening, matching, scoring, ranking, explanation, and batch pipeline package."""

from app.engine.embeddings import (
    BaseEmbeddingService,
    MockEmbeddingService,
    SentenceTransformerEmbeddingService,
    get_default_embedding_service,
)
from app.engine.explanation import (
    BaseExplanationService,
    FallbackExplanationService,
    LocalLLMExplanationService,
    get_default_explanation_service,
)
from app.engine.matcher import (
    build_candidate_semantic_text,
    build_jd_semantic_text,
    calculate_candidate_experience_years,
    match_explicit_skills,
)
from app.engine.pipeline import (
    BatchScreeningPipeline,
    run_batch_screening,
)
from app.engine.ranker import (
    CandidateRanker,
    create_ranking_result,
    rank_candidates,
)
from app.engine.scorer import ScoringEngine, evaluate_candidate

__all__ = [
    "BaseEmbeddingService",
    "SentenceTransformerEmbeddingService",
    "MockEmbeddingService",
    "get_default_embedding_service",
    "BaseExplanationService",
    "FallbackExplanationService",
    "LocalLLMExplanationService",
    "get_default_explanation_service",
    "build_candidate_semantic_text",
    "build_jd_semantic_text",
    "calculate_candidate_experience_years",
    "match_explicit_skills",
    "ScoringEngine",
    "evaluate_candidate",
    "CandidateRanker",
    "rank_candidates",
    "create_ranking_result",
    "BatchScreeningPipeline",
    "run_batch_screening",
]
