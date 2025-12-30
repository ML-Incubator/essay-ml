"""
Configuration module for essay-ml.
Defines all constants, model parameters, and scoring thresholds.
"""

from pathlib import Path
from typing import Dict, Tuple

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """ML model configuration."""

    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    random_state: int = 42
    max_features: str = "sqrt"


class FeatureConfig(BaseModel):
    """Feature extraction configuration."""

    tfidf_max_features: int = 500
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95


class ScoringConfig(BaseModel):
    """Scoring thresholds and weights."""

    # Category weights (must sum to 1.0)
    weights: Dict[str, float] = {
        "grammar": 0.25,
        "vocabulary": 0.25,
        "structure": 0.25,
        "argument": 0.25,
    }

    # Score thresholds
    excellent_threshold: float = 85.0
    good_threshold: float = 70.0
    fair_threshold: float = 55.0
    # Below fair_threshold is "needs improvement"


class PathConfig(BaseModel):
    """Project paths."""

    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"

    model_config = {"arbitrary_types_allowed": True}

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for path in [self.raw_data_dir, self.processed_data_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)


# Global configuration instances
MODEL_CONFIG = ModelConfig()
FEATURE_CONFIG = FeatureConfig()
SCORING_CONFIG = ScoringConfig()
PATH_CONFIG = PathConfig()

# Ensure directories exist
PATH_CONFIG.ensure_dirs()
