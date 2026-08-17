"""Semantic representation builders and explicit criteria matching utilities."""

from datetime import datetime
import re
from typing import List, Optional, Tuple

from app.models.candidate import Candidate
from app.models.job_description import JobDescription


def build_candidate_semantic_text(candidate: Candidate) -> str:
    """Construct a coherent semantic profile text from structured candidate data."""
    parts: List[str] = [f"Candidate: {candidate.name}"]

    if candidate.skills:
        parts.append(f"Core Skills: {', '.join(candidate.skills)}")

    if candidate.experience:
        exp_lines = ["Work and Research Experience:"]
        for exp in candidate.experience:
            title = exp.title or "Professional Role"
            company = f" at {exp.company}" if exp.company else ""
            dates = (
                f" ({exp.start_date} - {exp.end_date})"
                if exp.start_date and exp.end_date
                else ""
            )
            desc = f": {exp.description}" if exp.description else ""
            exp_lines.append(f"- {title}{company}{dates}{desc}")
        parts.append("\n".join(exp_lines))

    if candidate.education:
        edu_lines = ["Education Background:"]
        for edu in candidate.education:
            degree = edu.degree or "Degree"
            field = f" in {edu.field_of_study}" if edu.field_of_study else ""
            inst = f" from {edu.institution}" if edu.institution else ""
            year = f" ({edu.end_year})" if edu.end_year else ""
            edu_lines.append(f"- {degree}{field}{inst}{year}")
        parts.append("\n".join(edu_lines))

    # Incorporate any extra summary / project context if available
    if candidate.raw_text:
        proj_match = re.search(
            r"(?:PROJECTS|KEY PROJECTS)\s*\n(.*?)(?=\n[A-Z\s]{3,}|\Z)",
            candidate.raw_text,
            re.IGNORECASE | re.DOTALL,
        )
        if proj_match:
            parts.append(f"Projects:\n{proj_match.group(1).strip()}")

    return "\n\n".join(parts)


def build_jd_semantic_text(jd: JobDescription) -> str:
    """Construct a coherent semantic profile text from structured job description data."""
    parts: List[str] = [f"Target Role: {jd.title}"]

    if jd.required_skills:
        parts.append(f"Required Skills: {', '.join(jd.required_skills)}")

    if jd.preferred_skills:
        parts.append(f"Preferred Skills: {', '.join(jd.preferred_skills)}")

    if jd.minimum_experience_years is not None:
        parts.append(
            f"Experience Requirement: Minimum {jd.minimum_experience_years} years"
        )

    if jd.education_requirements:
        parts.append(
            f"Education Criteria: {', '.join(jd.education_requirements)}"
        )

    if jd.responsibilities:
        resp_lines = ["Key Responsibilities:"]
        for resp in jd.responsibilities:
            resp_lines.append(f"- {resp}")
        parts.append("\n".join(resp_lines))

    return "\n\n".join(parts)


def match_explicit_skills(
    candidate_skills: List[str], target_skills: List[str]
) -> Tuple[List[str], List[str]]:
    """Match candidate skills against target skills using case-insensitive canonical matching.

    Args:
        candidate_skills: List of skills held by the candidate.
        target_skills: List of required or preferred skills from the JD.

    Returns:
        Tuple of (matched_skills, missing_skills).
    """
    candidate_skill_map = {s.lower().strip(): s for s in candidate_skills if s.strip()}
    matched: List[str] = []
    missing: List[str] = []

    for target in target_skills:
        cleaned_target = target.strip()
        if not cleaned_target:
            continue
        key = cleaned_target.lower()
        if key in candidate_skill_map:
            matched.append(candidate_skill_map[key])
        else:
            missing.append(cleaned_target)

    return (matched, missing)


def calculate_candidate_experience_years(candidate: Candidate) -> float:
    """Estimate total candidate professional experience in years from experience entries."""
    if not candidate.experience:
        return 0.0

    current_year = datetime.now().year
    total_years: float = 0.0

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    def parse_date_token(token: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract (year, month) from a date token."""
        token = token.strip().lower()
        if token in {"present", "current", "now"}:
            return (current_year, 12)

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", token)
        if not year_match:
            return (None, None)
        year = int(year_match.group(1))

        month = None
        for m_str, m_num in month_map.items():
            if m_str in token:
                month = m_num
                break

        return (year, month or 1)

    for exp in candidate.experience:
        if exp.start_date and exp.end_date:
            s_year, s_month = parse_date_token(exp.start_date)
            e_year, e_month = parse_date_token(exp.end_date)

            if s_year and e_year and e_year >= s_year:
                # Calculate inclusive months
                months_diff = (e_year - s_year) * 12 + (e_month - s_month) + 1
                years_diff = max(0.08, months_diff / 12.0)
                total_years += years_diff
                continue

        if exp.title:
            total_years += 0.5

    return round(total_years, 2)
