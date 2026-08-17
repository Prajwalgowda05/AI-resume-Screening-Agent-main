"""Candidate ranking engine providing deterministic ordering and top-N filtering."""

from typing import List, Optional

from app.models.ranking import MatchResult, RankedCandidate, RankingResult


class CandidateRanker:
    """Deterministic ranking engine for scored candidate match results."""

    def rank(
        self, results: List[MatchResult], top_n: Optional[int] = None
    ) -> List[RankedCandidate]:
        """Order match results deterministically and assign 1-indexed ranks.

        Args:
            results: List of evaluated MatchResult objects.
            top_n: Optional positive integer to limit the returned rankings.

        Returns:
            List of RankedCandidate objects ordered from highest to lowest score.

        Raises:
            ValueError: If top_n is not a positive integer (<= 0).
        """
        if not results:
            return []

        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError(
                    f"top_n must be a positive integer greater than 0 (got {top_n})"
                )

        # Deterministic sort key:
        # 1. Higher final score (descending)
        # 2. Higher required skill score (descending)
        # 3. Higher semantic similarity score (descending)
        # 4. Candidate name alphabetically (ascending case-insensitive)
        sorted_results = sorted(
            results,
            key=lambda r: (
                -r.final_score,
                -r.breakdown.required_skill_score,
                -r.breakdown.semantic_similarity_score,
                r.candidate_name.lower(),
            ),
        )

        ranked_list: List[RankedCandidate] = []
        for idx, result in enumerate(sorted_results):
            ranked_candidate = RankedCandidate(
                rank=idx + 1,
                candidate_name=result.candidate_name,
                job_title=result.job_title,
                final_score=result.final_score,
                breakdown=result.breakdown,
                summary=result.summary,
                explanation=result.explanation,
            )
            ranked_list.append(ranked_candidate)

        if top_n is not None:
            return ranked_list[:top_n]

        return ranked_list

    def rank_as_result(
        self,
        results: List[MatchResult],
        job_title: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> RankingResult:
        """Rank match results and wrap into a RankingResult container.

        Args:
            results: List of evaluated MatchResult objects.
            job_title: Optional target job title (defaults to first match result's job title).
            top_n: Optional positive integer to limit the returned rankings.

        Returns:
            RankingResult container.
        """
        ranked = self.rank(results, top_n=top_n)
        resolved_title = (
            job_title
            or (results[0].job_title if results else "Candidate Ranking")
        )
        return RankingResult(
            job_title=resolved_title,
            total_candidates=len(results),
            rankings=ranked,
        )


def rank_candidates(
    results: List[MatchResult], top_n: Optional[int] = None
) -> List[RankedCandidate]:
    """Convenience function to rank evaluated candidates deterministically.

    Args:
        results: List of MatchResult objects.
        top_n: Optional positive integer limit.

    Returns:
        List of RankedCandidate objects.
    """
    ranker = CandidateRanker()
    return ranker.rank(results, top_n=top_n)


def create_ranking_result(
    results: List[MatchResult],
    job_title: Optional[str] = None,
    top_n: Optional[int] = None,
) -> RankingResult:
    """Convenience function to rank evaluated candidates and return a RankingResult.

    Args:
        results: List of MatchResult objects.
        job_title: Optional target job title.
        top_n: Optional positive integer limit.

    Returns:
        RankingResult container.
    """
    ranker = CandidateRanker()
    return ranker.rank_as_result(results, job_title=job_title, top_n=top_n)
