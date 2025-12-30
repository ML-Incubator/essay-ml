"""Tests for the features module."""

from src.core.features import FeatureExtractor


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_feature_extractor_init(self) -> None:
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
