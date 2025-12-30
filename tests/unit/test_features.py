"""Tests for the features module."""

import numpy as np

from src.core.features import FeatureExtractor


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_feature_extractor_init(self) -> None:
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert extractor.is_fitted is False

    def test_extract_linguistic_features(self) -> None:
        """Test linguistic feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_linguistic_features("This is a test sentence.")
        assert "avg_word_length" in features
        assert "lexical_diversity" in features
        assert "sentence_count" in features
        assert features["sentence_count"] == 1.0

    def test_extract_readability_features(self) -> None:
        """Test readability feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_readability_features("Simple text here.")
        assert "flesch_reading_ease" in features
        assert "avg_syllables_per_word" in features
        assert "complex_word_ratio" in features

    def test_extract_structural_features(self) -> None:
        """Test structural feature extraction."""
        extractor = FeatureExtractor()
        text = "First paragraph.\n\nSecond paragraph."
        features = extractor.extract_structural_features(text)
        assert "paragraph_count" in features
        assert features["paragraph_count"] == 2.0

    def test_fit_tfidf(self) -> None:
        """Test TF-IDF fitting."""
        extractor = FeatureExtractor()
        # Need enough documents for min_df=2 to work
        texts = [
            "Sample essay one about education and learning.",
            "Sample essay two about education and teaching.",
            "Sample essay three about education and students.",
        ]
        extractor.fit_tfidf(texts)
        assert extractor.is_fitted is True

    def test_extract_tfidf_features(self) -> None:
        """Test TF-IDF feature extraction."""
        extractor = FeatureExtractor()
        texts = [
            "Sample essay text about writing and learning.",
            "Another sample about writing and teaching.",
            "More sample text about writing and reading.",
        ]
        extractor.fit_tfidf(texts)
        tfidf = extractor.extract_tfidf_features(["Sample essay text about writing."])
        assert isinstance(tfidf, np.ndarray)
        assert tfidf.shape[0] == 1

    def test_extract_tfidf_not_fitted_raises(self) -> None:
        """Test that extracting TF-IDF before fitting raises error."""
        extractor = FeatureExtractor()
        try:
            extractor.extract_tfidf_features(["Test"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "fit_tfidf" in str(e)

    def test_extract_all_features(self) -> None:
        """Test extracting all features combined."""
        extractor = FeatureExtractor()
        texts = [
            "Sample text for fitting the vectorizer properly.",
            "Another text for fitting with similar words.",
            "Third text for fitting to meet min_df requirement.",
        ]
        extractor.fit_tfidf(texts)
        features = extractor.extract_all_features("Test essay for features extraction.")
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_get_feature_names(self) -> None:
        """Test getting feature names."""
        extractor = FeatureExtractor()
        texts = [
            "Sample text for feature names.",
            "Another sample text for testing.",
            "Third sample text for validation.",
        ]
        extractor.fit_tfidf(texts)
        names = extractor.get_feature_names()
        assert "avg_word_length" in names
        assert "flesch_reading_ease" in names
        assert "paragraph_count" in names
