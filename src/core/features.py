"""Feature extraction module (TF-IDF, linguistic features)."""

from typing import Any, Dict

import numpy as np


class FeatureExtractor:
    """Extracts features from essay text."""

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        pass

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract all features from text."""
        raise NotImplementedError("To be implemented in Phase 2")

    def extract_tfidf(self, text: str) -> np.ndarray:
        """Extract TF-IDF features."""
        raise NotImplementedError("To be implemented in Phase 2")

    def extract_linguistic(self, text: str) -> Dict[str, float]:
        """Extract linguistic features."""
        raise NotImplementedError("To be implemented in Phase 2")
