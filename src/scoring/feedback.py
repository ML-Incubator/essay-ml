"""Feedback generation module."""

from typing import Any, Dict, List


class FeedbackGenerator:
    """Generates actionable feedback for essays."""

    def __init__(self) -> None:
        """Initialize the feedback generator."""
        pass

    def generate(self, scores: Dict[str, float], essay: str) -> Dict[str, Any]:
        """Generate feedback based on scores."""
        raise NotImplementedError("To be implemented in Phase 2")

    def get_suggestions(self, category: str, score: float) -> List[str]:
        """Get improvement suggestions for a category."""
        raise NotImplementedError("To be implemented in Phase 2")
