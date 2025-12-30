"""ML model wrapper (RandomForest)."""

from pathlib import Path
from typing import Any, Optional

import numpy as np


class EssayModel:
    """Wrapper for the essay scoring ML model."""

    def __init__(self) -> None:
        """Initialize the model."""
        self.model: Optional[Any] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        raise NotImplementedError("To be implemented in Phase 2")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("To be implemented in Phase 2")

    def save(self, path: Path) -> None:
        """Save the model to disk."""
        raise NotImplementedError("To be implemented in Phase 2")

    def load(self, path: Path) -> None:
        """Load the model from disk."""
        raise NotImplementedError("To be implemented in Phase 2")
