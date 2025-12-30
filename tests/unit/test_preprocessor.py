"""Tests for the preprocessor module."""

from src.core.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_preprocessor_init(self) -> None:
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
