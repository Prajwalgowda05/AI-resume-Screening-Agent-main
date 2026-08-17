"""Modular explanation services generating grounded candidate evaluations."""

from abc import ABC, abstractmethod
import json
import os
from typing import Optional
import urllib.request
import urllib.error

from app.models.ranking import ScoreBreakdown


class BaseExplanationService(ABC):
    """Abstract base class for explanation generation services."""

    @abstractmethod
    def generate_explanation(
        self,
        candidate_name: str,
        job_title: str,
        final_score: float,
        breakdown: ScoreBreakdown,
    ) -> str:
        """Generate a concise 2-4 sentence explanation grounded strictly in structured evidence.

        Args:
            candidate_name: Candidate's full name.
            job_title: Target role title.
            final_score: Composite score (0-100).
            breakdown: Granular score breakdown with skill and experience metrics.

        Returns:
            Concise, grounded narrative explanation string.
        """
        pass


class FallbackExplanationService(BaseExplanationService):
    """Deterministic, rule-based explanation generator grounded purely in structured evaluation data."""

    def generate_explanation(
        self,
        candidate_name: str,
        job_title: str,
        final_score: float,
        breakdown: ScoreBreakdown,
    ) -> str:
        """Construct a structured, factual narrative explanation."""
        sentences = []

        # 1. Performance Level Sentence
        if final_score >= 80.0:
            sentences.append(
                f"{candidate_name} scored {final_score:.1f}/100, demonstrating strong qualifications "
                f"and close alignment with the {job_title} role."
            )
        elif final_score >= 60.0:
            sentences.append(
                f"{candidate_name} achieved a score of {final_score:.1f}/100, showing moderate alignment "
                f"with foundational requirements for the {job_title} position."
            )
        else:
            sentences.append(
                f"{candidate_name} scored {final_score:.1f}/100, reflecting limited alignment "
                f"with the core criteria for the {job_title} role."
            )

        # 2. Key Strengths Sentence
        matched_req = breakdown.matched_required_skills
        matched_pref = breakdown.matched_preferred_skills
        exp_years = breakdown.candidate_experience_years

        strength_parts = []
        if matched_req:
            top_skills = ", ".join(matched_req[:4])
            strength_parts.append(f"matching {len(matched_req)} required skills ({top_skills})")
        if matched_pref:
            top_pref = ", ".join(matched_pref[:3])
            strength_parts.append(f"possessing preferred skills ({top_pref})")
        if exp_years > 0:
            strength_parts.append(f"{exp_years:.1f} years of relevant experience")

        if strength_parts:
            sentences.append(f"Key strengths include {', and '.join(strength_parts)}.")
        else:
            sentences.append("The profile displays early-stage background in software and technology.")

        # 3. Missing Requirements & Gaps Sentence
        missing_req = breakdown.missing_required_skills
        req_exp = breakdown.required_experience_years or 0.0

        gap_parts = []
        if missing_req:
            gap_parts.append(f"missing mandatory skills in {', '.join(missing_req[:4])}")
        if req_exp > 0 and exp_years < req_exp:
            gap_parts.append(
                f"an experience shortfall ({exp_years:.1f} years vs. {req_exp:.1f} years required)"
            )

        if gap_parts:
            sentences.append(f"Primary gaps include {', and '.join(gap_parts)}.")
        else:
            sentences.append("No critical mandatory skill gaps were identified against the criteria.")

        # 4. Overall Assessment Sentence
        if final_score >= 80.0:
            sentences.append(
                "Overall, the candidate is a strong fit recommended for advancement to technical interviews."
            )
        elif final_score >= 60.0:
            sentences.append(
                "Overall, the candidate is a viable candidate but would benefit from evaluation in gap areas."
            )
        else:
            sentences.append(
                "Overall, significant skill and requirement gaps remain for this specific position."
            )

        return " ".join(sentences)


class LocalLLMExplanationService(BaseExplanationService):
    """Local LLM explanation service connecting to local Ollama or OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model_name: str = "llama3.2",
        timeout: float = 3.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.fallback = FallbackExplanationService()

    def generate_explanation(
        self,
        candidate_name: str,
        job_title: str,
        final_score: float,
        breakdown: ScoreBreakdown,
    ) -> str:
        """Generate explanation using local LLM endpoint, falling back to deterministic template on failure."""
        prompt = (
            f"You are an objective hiring assistant. Write a concise 2-4 sentence evaluation explaining why "
            f"candidate {candidate_name} scored {final_score:.1f}/100 for the role '{job_title}'.\n"
            f"Structured Evidence:\n"
            f"- Matched Required Skills: {', '.join(breakdown.matched_required_skills) or 'None'}\n"
            f"- Missing Required Skills: {', '.join(breakdown.missing_required_skills) or 'None'}\n"
            f"- Matched Preferred Skills: {', '.join(breakdown.matched_preferred_skills) or 'None'}\n"
            f"- Candidate Experience: {breakdown.candidate_experience_years:.1f} years (Required: {breakdown.required_experience_years or 0} years)\n"
            f"- Education Score: {breakdown.education_score:.1f}%\n"
            f"- Semantic Similarity: {breakdown.semantic_similarity_score:.1f}%\n\n"
            f"Rules:\n"
            f"1. Strictly use ONLY the evidence above. Never invent unlisted skills or credentials.\n"
            f"2. Keep length to exactly 2 to 4 sentences.\n"
            f"3. Do not modify or recalculate the score."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a grounded resume evaluation explanation assistant. Be factual, concise, and professional.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 160,
        }

        try:
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    res_data = json.loads(response.read().decode("utf-8"))
                    text = res_data["choices"][0]["message"]["content"].strip()
                    if text:
                        return text
        except Exception:
            # Seamlessly fallback if local server is unreachable or timed out
            pass

        return self.fallback.generate_explanation(
            candidate_name, job_title, final_score, breakdown
        )


def get_default_explanation_service(
    prefer_llm: bool = False,
) -> BaseExplanationService:
    """Retrieve active explanation service instance."""
    llm_api_base = os.getenv("LLM_API_BASE") or os.getenv("OLLAMA_HOST")
    if prefer_llm or llm_api_base:
        return LocalLLMExplanationService(
            api_base=llm_api_base or "http://localhost:11434/v1"
        )
    return FallbackExplanationService()
