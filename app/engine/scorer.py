"""Scoring engine combining explicit matching, semantic embeddings, experience, and education."""

from typing import List, Optional, Tuple

from app.engine.embeddings import (
    BaseEmbeddingService,
    get_default_embedding_service,
)
from app.engine.explanation import (
    BaseExplanationService,
    get_default_explanation_service,
)
from app.engine.matcher import (
    build_candidate_semantic_text,
    build_jd_semantic_text,
    calculate_candidate_experience_years,
    match_explicit_skills,
)
from app.models.candidate import Candidate
from app.models.job_description import JobDescription
from app.models.ranking import MatchResult, ScoreBreakdown, ScoringWeights


class ScoringEngine:
    """Multi-factor candidate evaluation and scoring engine."""

    DEGREE_RANKS = {
        "bachelor": 1,
        "bachelor's degree": 1,
        "b.tech": 1,
        "b.e.": 1,
        "b.s.": 1,
        "master": 2,
        "master's degree": 2,
        "m.tech": 2,
        "m.s.": 2,
        "m.e.": 2,
        "ph.d.": 3,
        "doctorate": 3,
    }

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        explanation_service: Optional[BaseExplanationService] = None,
    ):
        """Initialize the scoring engine with configurable weights, embedding service, and explanation service."""
        self.weights = weights or ScoringWeights()
        self.embedding_service = (
            embedding_service or get_default_embedding_service()
        )
        self.explanation_service = (
            explanation_service or get_default_explanation_service()
        )

    def compute_required_skills_score(
        self, candidate: Candidate, jd: JobDescription
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate score for mandatory/required skills match."""
        if not jd.required_skills:
            return (100.0, [], [])

        matched, missing = match_explicit_skills(
            candidate.skills, jd.required_skills
        )
        score = (len(matched) / len(jd.required_skills)) * 100.0
        return (round(score, 2), matched, missing)

    def compute_preferred_skills_score(
        self, candidate: Candidate, jd: JobDescription
    ) -> Tuple[float, List[str]]:
        """Calculate score for optional/preferred skills match."""
        if not jd.preferred_skills:
            return (100.0, [])

        matched, _ = match_explicit_skills(
            candidate.skills, jd.preferred_skills
        )
        score = (len(matched) / len(jd.preferred_skills)) * 100.0
        return (round(score, 2), matched)

    def compute_semantic_similarity_score(
        self, candidate: Candidate, jd: JobDescription
    ) -> float:
        """Calculate dense vector semantic similarity between Candidate and JD profiles."""
        cand_text = build_candidate_semantic_text(candidate)
        jd_text = build_jd_semantic_text(jd)

        if not cand_text.strip() or not jd_text.strip():
            return 0.0

        raw_sim = self.embedding_service.compute_similarity(cand_text, jd_text)
        # Scale cosine similarity from [-1.0, 1.0] to [0.0, 100.0]
        normalized_score = max(0.0, min(100.0, raw_sim * 100.0))
        return round(normalized_score, 2)

    def compute_experience_score(
        self, candidate: Candidate, jd: JobDescription
    ) -> Tuple[float, float]:
        """Calculate experience score comparing candidate duration against minimum required years."""
        cand_years = calculate_candidate_experience_years(candidate)

        if (
            jd.minimum_experience_years is None
            or jd.minimum_experience_years <= 0
        ):
            return (100.0, cand_years)

        if cand_years <= 0.0:
            return (0.0, cand_years)

        ratio = cand_years / jd.minimum_experience_years
        score = min(100.0, ratio * 100.0)
        return (round(score, 2), cand_years)

    def compute_education_score(
        self, candidate: Candidate, jd: JobDescription
    ) -> float:
        """Calculate education score comparing degree tiers and field of study."""
        if not jd.education_requirements:
            return 100.0

        if not candidate.education:
            return 0.0

        # Determine target degree tier from JD
        target_rank = 1
        target_fields = []

        for req in jd.education_requirements:
            req_lower = req.lower()
            if "ph.d" in req_lower or "doctorate" in req_lower:
                target_rank = max(target_rank, 3)
            elif "master" in req_lower or "m.s" in req_lower or "m.tech" in req_lower:
                target_rank = max(target_rank, 2)

            if "in " in req_lower:
                field = req_lower.split("in ")[-1].strip()
                target_fields.append(field)

        # Determine candidate's highest degree tier and discipline matches
        cand_highest_rank = 0
        cand_field_matched = False

        for edu in candidate.education:
            deg_lower = (edu.degree or "").lower()
            for key, rank in self.DEGREE_RANKS.items():
                if key in deg_lower:
                    cand_highest_rank = max(cand_highest_rank, rank)

            if edu.field_of_study:
                field_lower = edu.field_of_study.lower()
                for target_field in target_fields:
                    if target_field in field_lower or field_lower in target_field:
                        cand_field_matched = True

        if cand_highest_rank >= target_rank:
            base_score = 100.0
        elif cand_highest_rank == target_rank - 1:
            base_score = 70.0
        elif cand_highest_rank > 0:
            base_score = 40.0
        else:
            base_score = 30.0

        if target_fields and not cand_field_matched:
            base_score = max(20.0, base_score - 15.0)

        return round(min(100.0, base_score), 2)

    def evaluate(
        self, candidate: Candidate, jd: JobDescription
    ) -> MatchResult:
        """Evaluate a candidate against a job description and return comprehensive MatchResult."""
        req_score, matched_req, missing_req = (
            self.compute_required_skills_score(candidate, jd)
        )
        pref_score, matched_pref = self.compute_preferred_skills_score(
            candidate, jd
        )
        sem_score = self.compute_semantic_similarity_score(candidate, jd)
        exp_score, cand_years = self.compute_experience_score(candidate, jd)
        edu_score = self.compute_education_score(candidate, jd)

        # Dynamic weight redistribution when JD has no preferred skills
        if not jd.preferred_skills:
            w_req = self.weights.required_skills + self.weights.preferred_skills
            w_pref = 0.0
        else:
            w_req = self.weights.required_skills
            w_pref = self.weights.preferred_skills

        w_sem = self.weights.semantic_similarity
        w_exp = self.weights.experience
        w_edu = self.weights.education

        final_score = (
            w_req * req_score
            + w_pref * pref_score
            + w_sem * sem_score
            + w_exp * exp_score
            + w_edu * edu_score
        )
        final_score = round(max(0.0, min(100.0, final_score)), 2)

        breakdown = ScoreBreakdown(
            required_skill_score=req_score,
            preferred_skill_score=pref_score,
            semantic_similarity_score=sem_score,
            experience_score=exp_score,
            education_score=edu_score,
            matched_required_skills=matched_req,
            missing_required_skills=missing_req,
            matched_preferred_skills=matched_pref,
            candidate_experience_years=cand_years,
            required_experience_years=jd.minimum_experience_years,
        )

        summary_parts = [
            f"{candidate.name} scored {final_score:.1f}/100 for the {jd.title} position.",
            f"Skills Match: {len(matched_req)}/{len(jd.required_skills)} required skills matched"
            + (f" (missing: {', '.join(missing_req)})" if missing_req else "")
            + f", {len(matched_pref)} preferred skills matched.",
            f"Semantic Relevance: {sem_score:.1f}%.",
            f"Experience: {cand_years:.1f} years (required: {jd.minimum_experience_years or 0} years).",
            f"Education Score: {edu_score:.1f}%.",
        ]
        summary = " ".join(summary_parts)

        # Generate structured grounded narrative explanation
        explanation = self.explanation_service.generate_explanation(
            candidate_name=candidate.name,
            job_title=jd.title,
            final_score=final_score,
            breakdown=breakdown,
        )

        return MatchResult(
            candidate_name=candidate.name,
            job_title=jd.title,
            final_score=final_score,
            breakdown=breakdown,
            summary=summary,
            explanation=explanation,
        )


def evaluate_candidate(
    candidate: Candidate,
    jd: JobDescription,
    weights: Optional[ScoringWeights] = None,
    embedding_service: Optional[BaseEmbeddingService] = None,
    explanation_service: Optional[BaseExplanationService] = None,
) -> MatchResult:
    """Convenience function to evaluate a candidate against a job description.

    Args:
        candidate: Structured Candidate profile.
        jd: Structured JobDescription criteria.
        weights: Optional custom ScoringWeights.
        embedding_service: Optional custom embedding service.
        explanation_service: Optional custom explanation service.

    Returns:
        MatchResult object.
    """
    engine = ScoringEngine(
        weights=weights,
        embedding_service=embedding_service,
        explanation_service=explanation_service,
    )
    return engine.evaluate(candidate, jd)
