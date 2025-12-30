"""Tests for the preprocessor module."""

from src.core.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_preprocessor_init(self) -> None:
        """Test preprocessor initialization."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
        assert preprocessor.remove_stopwords_flag is False

    def test_preprocessor_init_with_stopwords(self) -> None:
        """Test preprocessor initialization with stopword removal."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        assert preprocessor.remove_stopwords_flag is True
        assert len(preprocessor.stop_words) > 0

    def test_clean_text_removes_urls(self) -> None:
        """Test that URLs are removed."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Check http://example.com here")
        assert "http" not in result
        assert "example.com" not in result

    def test_clean_text_removes_emails(self) -> None:
        """Test that emails are removed."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Contact test@email.com today")
        assert "@" not in result

    def test_clean_text_normalizes_whitespace(self) -> None:
        """Test that multiple spaces are normalized."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Too    many   spaces")
        assert "  " not in result

    def test_clean_text_preserves_punctuation(self) -> None:
        """Test that essential punctuation is preserved."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Hello! How are you?")
        assert "!" in result
        assert "?" in result

    def test_tokenize_sentences(self) -> None:
        """Test sentence tokenization."""
        preprocessor = TextPreprocessor()
        sentences = preprocessor.tokenize_sentences("First sentence. Second one!")
        assert len(sentences) == 2

    def test_tokenize_words(self) -> None:
        """Test word tokenization."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize_words("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_words_removes_stopwords(self) -> None:
        """Test that stopwords are removed when enabled."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        tokens = preprocessor.tokenize_words("This is a test")
        assert "is" not in tokens
        assert "a" not in tokens
        assert "test" in tokens

    def test_get_pos_tags(self) -> None:
        """Test POS tagging."""
        preprocessor = TextPreprocessor()
        tags = preprocessor.get_pos_tags(["this", "is", "great"])
        assert len(tags) == 3
        assert all(isinstance(t, tuple) for t in tags)

    def test_preprocess_full_pipeline(self) -> None:
        """Test full preprocessing pipeline."""
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Hello world! This is a test.")
        assert "original" in result
        assert "cleaned" in result
        assert "sentences" in result
        assert "tokens" in result
        assert "pos_tags" in result
        assert len(result["sentences"]) == 2
