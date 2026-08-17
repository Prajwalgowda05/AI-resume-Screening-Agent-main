"""Resume information extraction module providing structured candidate profiling."""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from app.models.candidate import Candidate, Education, Experience
from app.parsers import parse_document


class BaseResumeExtractor(ABC):
    """Abstract interface for resume information extractors."""

    @abstractmethod
    def extract_from_text(self, raw_text: str) -> Candidate:
        """Extract candidate information from raw resume text.

        Args:
            raw_text: Raw extracted text from resume.

        Returns:
            Validated Candidate profile.
        """
        pass

    def extract_from_file(self, file_path: Union[str, Path]) -> Candidate:
        """Parse a resume file and extract candidate profile.

        Args:
            file_path: Path to resume document.

        Returns:
            Validated Candidate profile.
        """
        raw_text = parse_document(file_path)
        return self.extract_from_text(raw_text)


class RuleBasedResumeExtractor(BaseResumeExtractor):
    """Deterministic, rule-based resume information extractor."""

    # Pre-compiled regex patterns for contact details
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    )

    # Standard canonical skill dictionary with aliases (conservative, no ambiguous single-letters)
    CANONICAL_SKILLS: Dict[str, str] = {
        "python": "Python",
        "python3": "Python",
        "python 3": "Python",
        "pytorch": "PyTorch",
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "tf": "TensorFlow",
        "keras": "Keras",
        "scikit-learn": "Scikit-Learn",
        "sklearn": "Scikit-Learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "c++": "C++",
        "cpp": "C++",
        "c#": "C#",
        "java": "Java",
        "sql": "SQL",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "sqlite": "SQLite",
        "mongodb": "MongoDB",
        "machine learning": "Machine Learning",
        "ml": "Machine Learning",
        "deep learning": "Deep Learning",
        "dl": "Deep Learning",
        "nlp": "Natural Language Processing",
        "natural language processing": "Natural Language Processing",
        "computer vision": "Computer Vision",
        "cv": "Computer Vision",
        "large language models": "Large Language Models",
        "llm": "Large Language Models",
        "llms": "Large Language Models",
        "transformers": "Transformers",
        "bert": "BERT",
        "gpt": "GPT",
        "reinforcement learning": "Reinforcement Learning",
        "rl": "Reinforcement Learning",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "aws": "AWS",
        "gcp": "GCP",
        "azure": "Azure",
        "linux": "Linux",
        "bash": "Bash",
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "react": "React",
        "reactjs": "React",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "html": "HTML",
        "css": "CSS",
    }

    # Section Headers Patterns
    SECTION_PATTERNS = {
        "skills": re.compile(
            r"^(?:technical\s+)?skills(?:\s*&\s*(?:tools|technologies|competencies))?$",
            re.IGNORECASE,
        ),
        "education": re.compile(
            r"^(?:education|academic\s+(?:background|qualifications)|qualifications)$",
            re.IGNORECASE,
        ),
        "experience": re.compile(
            r"^(?:(?:work|professional|research)\s+)?experience|employment\s+history$",
            re.IGNORECASE,
        ),
        "projects": re.compile(r"^(?:key\s+)?projects|academic\s+projects$", re.IGNORECASE),
    }

    DEGREE_PATTERNS = [
        (re.compile(r"\b(?:Ph\.?D\.?|Doctor\s+of\s+Philosophy)\b", re.IGNORECASE), "Ph.D."),
        (
            re.compile(
                r"\b(?:M\.?S\.?|Master(?:\'s)?(?:\s+of\s+Science)?|M\.?Tech\.?|M\.?E\.?)\b",
                re.IGNORECASE,
            ),
            "Master's Degree",
        ),
        (
            re.compile(
                r"\b(?:B\.?S\.?|Bachelor(?:\'s)?(?:\s+of\s+Science)?|B\.?Tech\.?|B\.?E\.?|B\.?A\.?)\b",
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

    JOB_TITLE_KEYWORDS = [
        "Engineer",
        "Scientist",
        "Researcher",
        "Developer",
        "Intern",
        "Fellow",
        "Analyst",
        "Assistant",
        "Lead",
        "Manager",
        "Consultant",
        "Architect",
    ]

    DATE_PATTERN = re.compile(
        r"((?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?,?\s*)?\d{4})\s*[-–—to]+\s*((?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?,?\s*)?\d{4}|Present|Current)",
        re.IGNORECASE,
    )

    YEAR_RANGE_PATTERN = re.compile(
        r"\b(19\d{2}|20\d{2})\s*[-–—to]+\s*(19\d{2}|20\d{2})\b"
    )
    SINGLE_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

    def extract_from_text(self, raw_text: str) -> Candidate:
        """Extract candidate information deterministically from raw resume text."""
        if not raw_text or not raw_text.strip():
            return Candidate(raw_text="")

        normalized_text = raw_text.strip()
        lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]

        email = self._extract_email(normalized_text)
        phone = self._extract_phone(normalized_text)
        name = self._extract_name(lines, email, phone)

        sections = self._segment_sections(lines)

        skills = self._extract_skills(normalized_text, sections.get("skills", []))
        education = self._extract_education(sections.get("education", []), lines)
        experience = self._extract_experience(sections.get("experience", []), lines)

        return Candidate(
            name=name,
            email=email,
            phone=phone,
            skills=skills,
            education=education,
            experience=experience,
            raw_text=raw_text,
        )

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract first valid email address."""
        match = self.EMAIL_PATTERN.search(text)
        return match.group(0).lower() if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract first valid phone number."""
        match = self.PHONE_PATTERN.search(text)
        if match:
            phone_str = match.group(0).strip()
            digits_only = re.sub(r"\D", "", phone_str)
            if len(digits_only) >= 10:
                return phone_str
        return None

    def _extract_name(
        self, lines: List[str], email: Optional[str], phone: Optional[str]
    ) -> str:
        """Extract candidate name using top-of-page heuristics."""
        if not lines:
            return "Unknown Candidate"

        for line in lines[:5]:
            if re.match(r"^name\s*:\s*(.+)$", line, re.IGNORECASE):
                extracted = re.sub(r"^name\s*:\s*", "", line, flags=re.IGNORECASE).strip()
                if extracted:
                    return extracted

        for line in lines[:5]:
            if email and email in line.lower():
                continue
            if phone and phone in line:
                continue
            if re.search(r"https?://|github\.com|linkedin\.com|@|\.com", line, re.IGNORECASE):
                continue
            if any(p.match(line) for p in self.SECTION_PATTERNS.values()):
                continue
            if len(line.split()) > 6 or len(line) < 2:
                continue
            if re.match(r"^[A-Za-z\s\.\,\-\'\"]+$", line):
                cleaned = re.sub(r"[^A-Za-z\s\.\-]", "", line).strip()
                if len(cleaned.split()) >= 1:
                    return cleaned

        return "Unknown Candidate"

    def _segment_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """Group resume lines into identified section blocks."""
        sections: Dict[str, List[str]] = {}
        current_section: Optional[str] = None

        for line in lines:
            matched_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.match(line):
                    matched_section = section_name
                    break

            if matched_section:
                current_section = matched_section
                if current_section not in sections:
                    sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        return sections

    def _extract_skills(
        self, full_text: str, skill_section_lines: List[str]
    ) -> List[str]:
        """Extract and normalize candidate skills, prioritizing the explicit SKILLS section."""
        found_skills: Set[str] = set()

        if skill_section_lines:
            skill_section_text = "\n".join(skill_section_lines)
            lower_section_text = skill_section_text.lower()
            for alias, canonical in self.CANONICAL_SKILLS.items():
                pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                if re.search(pattern, lower_section_text):
                    found_skills.add(canonical)

            for line in skill_section_lines:
                tokens = re.split(r"[,|•·/;\n]+", line)
                for token in tokens:
                    cleaned_token = token.strip()
                    if not cleaned_token or len(cleaned_token) > 40:
                        continue
                    lower_token = cleaned_token.lower()
                    if lower_token in self.CANONICAL_SKILLS:
                        found_skills.add(self.CANONICAL_SKILLS[lower_token])
                    elif (
                        len(cleaned_token.split()) <= 4
                        and re.match(r"^[A-Za-z0-9\+\#\.\s\-]+$", cleaned_token)
                        and len(cleaned_token) >= 2
                    ):
                        found_skills.add(cleaned_token.title())
        else:
            lower_full_text = full_text.lower()
            for alias, canonical in self.CANONICAL_SKILLS.items():
                if len(alias) < 3 and alias not in {"c++", "c#", "r", "ml", "dl", "ai"}:
                    continue
                pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
                if re.search(pattern, lower_full_text):
                    found_skills.add(canonical)

        return sorted(list(found_skills), key=lambda x: x.lower())

    def _extract_education(
        self, edu_lines: List[str], all_lines: List[str]
    ) -> List[Education]:
        """Extract structured education entries across multi-line blocks."""
        lines_to_search = edu_lines if edu_lines else all_lines
        education_entries: List[Education] = []
        current_edu: Optional[Education] = None

        for line in lines_to_search:
            degree_found = None
            for pattern, canonical_degree in self.DEGREE_PATTERNS:
                if pattern.search(line):
                    degree_found = canonical_degree
                    break

            # Search for field of study
            field_found = None
            for field in self.FIELD_OF_STUDY_KEYWORDS:
                if re.search(r"\b" + re.escape(field) + r"\b", line, re.IGNORECASE):
                    field_found = field
                    break

            # Search for year range or single year
            start_year = None
            end_year = None
            range_match = self.YEAR_RANGE_PATTERN.search(line)
            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
            else:
                single_match = self.SINGLE_YEAR_PATTERN.search(line)
                if single_match:
                    end_year = int(single_match.group(1))

            # Search for institution
            inst_match = re.search(
                r"([A-Za-z\s]+(?:University|College|Institute|School|Academy)[A-Za-z\s]*)",
                line,
                re.IGNORECASE,
            )
            institution_found = inst_match.group(1).strip() if inst_match else None

            if degree_found:
                if current_edu:
                    education_entries.append(current_edu)
                current_edu = Education(
                    degree=degree_found,
                    field_of_study=field_found,
                    institution=institution_found,
                    start_year=start_year,
                    end_year=end_year,
                    raw_text=line,
                )
            elif current_edu:
                if not current_edu.field_of_study and field_found:
                    current_edu.field_of_study = field_found
                if start_year is not None and current_edu.start_year is None:
                    current_edu.start_year = start_year
                if end_year is not None and current_edu.end_year is None:
                    current_edu.end_year = end_year
                if not current_edu.institution and institution_found:
                    current_edu.institution = institution_found
                if current_edu.raw_text:
                    current_edu.raw_text += f"\n{line}"

        if current_edu:
            education_entries.append(current_edu)

        return education_entries

    def _extract_experience(
        self, exp_lines: List[str], all_lines: List[str]
    ) -> List[Experience]:
        """Extract structured experience entries across multi-line blocks."""
        lines_to_search = exp_lines if exp_lines else []
        if not lines_to_search:
            return []

        experience_entries: List[Experience] = []
        current_exp: Optional[Experience] = None

        for line in lines_to_search:
            date_match = self.DATE_PATTERN.search(line)
            title_found = None
            inline_company = None

            for title_kw in self.JOB_TITLE_KEYWORDS:
                if re.search(r"\b" + re.escape(title_kw) + r"\b", line, re.IGNORECASE):
                    # Check if line has title | company format
                    if "|" in line:
                        parts = [p.strip() for p in line.split("|")]
                        title_found = parts[0]
                        if len(parts) > 1 and not self.DATE_PATTERN.search(parts[1]):
                            inline_company = parts[1]
                    elif " at " in line:
                        parts = line.split(" at ")
                        title_found = parts[0].strip()
                        if len(parts) > 1:
                            inline_company = parts[1].strip()
                    else:
                        title_found = line.strip()
                    break

            # If this line introduces a NEW title and we already have an active entry with a title, push current
            if title_found and current_exp and current_exp.title:
                experience_entries.append(current_exp)
                current_exp = None

            if not current_exp:
                start_d = date_match.group(1).strip() if date_match else None
                end_d = date_match.group(2).strip() if date_match else None
                current_exp = Experience(
                    title=title_found,
                    company=inline_company,
                    start_date=start_d,
                    end_date=end_d,
                    raw_text=line,
                )
            else:
                # Supplement current entry
                if not current_exp.title and title_found:
                    current_exp.title = title_found
                if inline_company and not current_exp.company:
                    current_exp.company = inline_company

                if date_match:
                    current_exp.start_date = date_match.group(1).strip()
                    current_exp.end_date = date_match.group(2).strip()
                elif (
                    current_exp.company is None
                    and current_exp.start_date is None
                    and not line.startswith(("-", "•", "*"))
                    and len(line.split()) <= 7
                ):
                    # Intermediate line between title and date is the company name
                    current_exp.company = line.strip()
                elif line.startswith(("-", "•", "*")) or current_exp.start_date is not None:
                    # Description bullet points or details after date
                    if current_exp.description:
                        current_exp.description += f" {line.strip()}"
                    else:
                        current_exp.description = line.strip()

                if current_exp.raw_text:
                    current_exp.raw_text += f"\n{line}"

        if current_exp:
            experience_entries.append(current_exp)

        return experience_entries


def extract_resume(
    file_path: Union[str, Path],
    extractor: Optional[BaseResumeExtractor] = None,
) -> Candidate:
    """Extract candidate profile from a resume file.

    Args:
        file_path: Path to the candidate's resume (PDF, DOCX, or TXT).
        extractor: Optional custom extractor. Defaults to RuleBasedResumeExtractor.

    Returns:
        Validated Candidate object.
    """
    if extractor is None:
        extractor = RuleBasedResumeExtractor()
    return extractor.extract_from_file(file_path)
