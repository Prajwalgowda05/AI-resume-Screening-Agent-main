"""Job description extraction module providing structured JD profiling."""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from app.extractors.resume_extractor import RuleBasedResumeExtractor
from app.models.job_description import JobDescription
from app.parsers import parse_document


class BaseJobDescriptionExtractor(ABC):
    """Abstract interface for job description extractors."""

    @abstractmethod
    def extract_from_text(self, raw_text: str) -> JobDescription:
        """Extract structured information from raw job description text.

        Args:
            raw_text: Raw text of the job description.

        Returns:
            Validated JobDescription object.
        """
        pass

    def extract_from_file(self, file_path: Union[str, Path]) -> JobDescription:
        """Parse a job description document and extract structured criteria.

        Args:
            file_path: Path to job description document (PDF, DOCX, TXT).

        Returns:
            Validated JobDescription object.
        """
        raw_text = parse_document(file_path)
        return self.extract_from_text(raw_text)


class RuleBasedJobDescriptionExtractor(BaseJobDescriptionExtractor):
    """Deterministic, rule-based job description extractor."""

    # Reuse canonical skill dictionary from RuleBasedResumeExtractor
    CANONICAL_SKILLS: Dict[str, str] = RuleBasedResumeExtractor.CANONICAL_SKILLS

    # Non-skill stop words and phrases to ignore when scanning custom tokens
    SKILL_STOPWORDS = {
        "experience",
        "knowledge",
        "ability",
        "proficient",
        "proficiency",
        "understanding",
        "strong",
        "solid",
        "basic",
        "minimum",
        "field",
        "related",
        "degree",
        "year",
        "years",
        "plus",
        "bonus",
        "required",
        "preferred",
        "candidate",
        "candidates",
        "team",
        "tools",
        "skills",
    }

    # Section Headers Patterns
    REQUIRED_SECTION_PATTERN = re.compile(
        r"^(?:minimum\s+)?(?:requirements|qualifications|required\s+(?:skills|qualifications)|must\s+haves?|what\s+you(?:\'ll)?\s+need|basic\s+qualifications)$",
        re.IGNORECASE,
    )

    PREFERRED_SECTION_PATTERN = re.compile(
        r"^(?:preferred\s+(?:qualifications|skills)|nice\s+to\s+haves?|bonus(?:\s+points?)?|good\s+to\s+have|plus(?:\s+points?)?|desired\s+(?:skills|qualifications))$",
        re.IGNORECASE,
    )

    RESPONSIBILITIES_SECTION_PATTERN = re.compile(
        r"^(?:responsibilities|key\s+responsibilities|what\s+you(?:\'ll)?\s+do|duties|role\s+responsibilities|core\s+responsibilities)$",
        re.IGNORECASE,
    )

    DEGREE_PATTERNS = [
        (re.compile(r"\b(?:Ph\.?D\.?|Doctorate|Doctor\s+of\s+Philosophy)\b", re.IGNORECASE), "Ph.D."),
        (
            re.compile(
                r"\b(?:M\.?S\.?|Master(?:\'s)?(?:\s+degree)?|M\.?Tech\.?|M\.?E\.?)\b",
                re.IGNORECASE,
            ),
            "Master's Degree",
        ),
        (
            re.compile(
                r"\b(?:B\.?S\.?|Bachelor(?:\'s)?(?:\s+degree)?|B\.?Tech\.?|B\.?E\.?|B\.?A\.?)\b",
                re.IGNORECASE,
            ),
            "Bachelor's Degree",
        ),
    ]

    FIELD_OF_STUDY_KEYWORDS = [
        "Computer Science",
        "Artificial Intelligence",
        "Data Science",
        "Machine Learning",
        "Electrical Engineering",
        "Software Engineering",
        "Information Technology",
        "Mathematics",
        "Statistics",
        "Physics",
        "Robotics",
    ]

    EXPERIENCE_PATTERNS = [
        re.compile(
            r"(?:at\s+least|minimum|min\.?|minimum\s+of)?\s*(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:relevant|professional|software|hands-on|industry|work)?\s*experience",
            re.IGNORECASE,
        ),
        re.compile(
            r"(\d+(?:\.\d+)?)\+\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience)?",
            re.IGNORECASE,
        ),
    ]

    def extract_from_text(self, raw_text: str) -> JobDescription:
        """Extract structured criteria from raw job description text."""
        if not raw_text or not raw_text.strip():
            return JobDescription(raw_text="")

        normalized_text = raw_text.strip()
        lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]

        title = self._extract_title(lines)
        sections = self._segment_sections(lines)

        required_skills = self._extract_skills_from_lines(
            sections.get("required", [])
        )
        preferred_skills = self._extract_skills_from_lines(
            sections.get("preferred", [])
        )

        # If no explicit sections found, perform conservative full-text matching for required
        if not required_skills and not preferred_skills:
            required_skills = self._extract_skills_from_text(normalized_text)

        # Avoid duplicating required skills in preferred skills
        preferred_skills = [
            skill for skill in preferred_skills if skill not in required_skills
        ]

        min_exp = self._extract_experience_years(
            sections.get("required", []) + lines
        )
        education_reqs = self._extract_education_requirements(
            sections.get("required", []) + lines
        )
        responsibilities = self._extract_responsibilities(
            sections.get("responsibilities", [])
        )

        return JobDescription(
            title=title,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            minimum_experience_years=min_exp,
            education_requirements=education_reqs,
            responsibilities=responsibilities,
            raw_text=raw_text,
        )

    def _extract_title(self, lines: List[str]) -> str:
        """Extract job title from the header lines."""
        if not lines:
            return "Unknown Position"

        for line in lines[:5]:
            # Look for explicit label
            match = re.match(
                r"^(?:job\s+title|role|position|title)\s*:\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted

        # Heuristic: First clean line that isn't a section header
        for line in lines[:3]:
            if (
                not self.REQUIRED_SECTION_PATTERN.match(line)
                and not self.PREFERRED_SECTION_PATTERN.match(line)
                and not self.RESPONSIBILITIES_SECTION_PATTERN.match(line)
                and len(line.split()) <= 8
                and len(line) >= 3
            ):
                return line.strip()

        return "Unknown Position"

    def _segment_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """Group job description lines into identified sections."""
        sections: Dict[str, List[str]] = {
            "required": [],
            "preferred": [],
            "responsibilities": [],
            "other": [],
        }
        current_section = "other"

        for line in lines:
            if self.REQUIRED_SECTION_PATTERN.match(line):
                current_section = "required"
                continue
            elif self.PREFERRED_SECTION_PATTERN.match(line):
                current_section = "preferred"
                continue
            elif self.RESPONSIBILITIES_SECTION_PATTERN.match(line):
                current_section = "responsibilities"
                continue

            sections[current_section].append(line)

        return sections

    def _extract_skills_from_lines(self, lines: List[str]) -> List[str]:
        """Extract and normalize skills from a specific section's lines."""
        if not lines:
            return []

        found_skills: Set[str] = set()
        section_text = "\n".join(lines)
        lower_section_text = section_text.lower()

        # 1. Match known canonical skills
        for alias, canonical in self.CANONICAL_SKILLS.items():
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, lower_section_text):
                found_skills.add(canonical)

        # 2. Extract delimited custom tokens from lines
        for line in lines:
            cleaned_line = re.sub(r"^[-•*–\d\.]+\s*", "", line)
            tokens = re.split(r"[,|/;\n]+", cleaned_line)
            for token in tokens:
                cleaned_token = token.strip()
                # Strip leading conjunctions / prepositions
                cleaned_token = re.sub(
                    r"^(?:and|or|with|in|plus|including|such\s+as)\s+",
                    "",
                    cleaned_token,
                    flags=re.IGNORECASE,
                ).strip()
                # Strip trailing punctuation
                cleaned_token = re.sub(r"[\.,;:!\?]+$", "", cleaned_token).strip()

                if not cleaned_token or len(cleaned_token) > 35 or len(cleaned_token) < 2:
                    continue

                lower_token = cleaned_token.lower()
                if lower_token in self.CANONICAL_SKILLS:
                    found_skills.add(self.CANONICAL_SKILLS[lower_token])
                elif (
                    len(cleaned_token.split()) <= 3
                    and re.match(r"^[A-Za-z0-9\+\#\.\s\-]+$", cleaned_token)
                    and not any(sw in lower_token.split() for sw in self.SKILL_STOPWORDS)
                ):
                    found_skills.add(cleaned_token.title())

        return sorted(list(found_skills), key=lambda x: x.lower())

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Conservative fallback for matching skills when no sections exist."""
        found_skills: Set[str] = set()
        lower_text = text.lower()

        for alias, canonical in self.CANONICAL_SKILLS.items():
            if len(alias) < 3 and alias not in {"c++", "c#", "r", "ml", "dl", "ai"}:
                continue
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, lower_text):
                found_skills.add(canonical)

        return sorted(list(found_skills), key=lambda x: x.lower())

    def _extract_experience_years(self, lines: List[str]) -> Optional[float]:
        """Extract required minimum years of experience."""
        for line in lines:
            for pattern in self.EXPERIENCE_PATTERNS:
                match = pattern.search(line)
                if match:
                    try:
                        years = float(match.group(1))
                        return years
                    except (ValueError, IndexError):
                        continue
        return None

    def _extract_education_requirements(self, lines: List[str]) -> List[str]:
        """Extract degree and discipline requirements."""
        education_reqs: Set[str] = set()

        for line in lines:
            # Check for degrees
            for pattern, canonical_degree in self.DEGREE_PATTERNS:
                if pattern.search(line):
                    education_reqs.add(canonical_degree)

            # Check for field of study mentions
            for field in self.FIELD_OF_STUDY_KEYWORDS:
                if re.search(r"\b" + re.escape(field) + r"\b", line, re.IGNORECASE):
                    education_reqs.add(f"Degree in {field}")

        return sorted(list(education_reqs))

    def _extract_responsibilities(self, lines: List[str]) -> List[str]:
        """Extract responsibility items and bullet points."""
        responsibilities: List[str] = []

        for line in lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            bullet_cleaned = re.sub(r"^[-•*–\d\.]+\s*", "", cleaned).strip()
            if len(bullet_cleaned) > 5:
                responsibilities.append(bullet_cleaned)

        return responsibilities


def extract_job_description(
    file_path: Union[str, Path],
    extractor: Optional[BaseJobDescriptionExtractor] = None,
) -> JobDescription:
    """Extract structured job description criteria from a document.

    Args:
        file_path: Path to the job description document (PDF, DOCX, or TXT).
        extractor: Optional custom extractor. Defaults to RuleBasedJobDescriptionExtractor.

    Returns:
        Validated JobDescription object.
    """
    if extractor is None:
        extractor = RuleBasedJobDescriptionExtractor()
    return extractor.extract_from_file(file_path)
