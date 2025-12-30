"""
Text preprocessing module.
Handles cleaning, tokenization, and normalization of essay text.
"""

import re
from typing import Any, Dict, List, Tuple

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class TextPreprocessor:
    """
    Preprocesses essay text for feature extraction and model input.

    Responsibilities:
    - Remove special characters and normalize whitespace
    - Tokenize into sentences and words
    - Remove stopwords (optional)
    - Preserve essential punctuation for grammar analysis
    """

    def __init__(self, remove_stopwords: bool = False) -> None:
        """
        Initialize preprocessor.

        Args:
            remove_stopwords: Whether to remove common English stopwords
        """
        self.remove_stopwords_flag = remove_stopwords
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text while preserving structure.

        Steps:
        1. Remove URLs (http://, https://, www.)
        2. Remove email addresses
        3. Normalize whitespace (multiple spaces -> single space)
        4. Remove special characters but keep: . , ! ? ' " -
        5. Preserve sentence structure

        Args:
            text: Raw essay text

        Returns:
            Cleaned text ready for analysis

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_text("Check   this!! http://example.com")
            'Check this!!'
        """
        if not text:
            return ""

        # Remove URLs (http://, https://, www.)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove special characters but keep: . , ! ? ' " - and alphanumeric
        text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"\-]", "", text)

        # Normalize whitespace (multiple spaces -> single space)
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses NLTK's sent_tokenize which handles common abbreviations.

        Args:
            text: Cleaned text

        Returns:
            List of sentences

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.tokenize_sentences("Hello world. How are you?")
            ['Hello world.', 'How are you?']
        """
        if not text:
            return []

        result: List[str] = sent_tokenize(text)
        return result

    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into words/tokens.

        Args:
            text: Text to tokenize

        Returns:
            List of word tokens (lowercase, optionally without stopwords)

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.tokenize_words("Hello World!")
            ['hello', 'world', '!']
        """
        if not text:
            return []

        # Tokenize using NLTK
        tokens = word_tokenize(text)

        # Convert to lowercase
        tokens = [token.lower() for token in tokens]

        # Optionally remove stopwords
        if self.remove_stopwords_flag:
            tokens = [token for token in tokens if token not in self.stop_words]

        result: List[str] = tokens
        return result

    def get_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for tokens.

        Args:
            tokens: List of word tokens

        Returns:
            List of (word, POS_tag) tuples

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.get_pos_tags(['this', 'is', 'great'])
            [('this', 'DT'), ('is', 'VBZ'), ('great', 'JJ')]
        """
        if not tokens:
            return []

        result: List[Tuple[str, str]] = pos_tag(tokens)
        return result

    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Full preprocessing pipeline.

        Returns dictionary with all processed forms for downstream use.

        Args:
            text: Raw essay text

        Returns:
            Dictionary containing:
                - 'original': original text
                - 'cleaned': cleaned text
                - 'sentences': list of sentences
                - 'tokens': list of word tokens
                - 'pos_tags': list of POS tags

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> result = preprocessor.preprocess("Hello world!")
            >>> result['cleaned']
            'Hello world!'
        """
        cleaned = self.clean_text(text)
        sentences = self.tokenize_sentences(cleaned)
        tokens = self.tokenize_words(cleaned)
        pos_tags_list = self.get_pos_tags(tokens)

        return {
            "original": text,
            "cleaned": cleaned,
            "sentences": sentences,
            "tokens": tokens,
            "pos_tags": pos_tags_list,
        }
