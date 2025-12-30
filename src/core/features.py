"""
Feature extraction module.
Extracts numerical features from essays for ML model input.
"""

import re
from typing import Dict, List

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config import FEATURE_CONFIG
from src.core.preprocessor import TextPreprocessor


class FeatureExtractor:
    """
    Extracts multiple feature types from essay text.

    Feature Categories:
    1. TF-IDF features (content representation)
    2. Linguistic features (grammar, vocabulary, structure)
    3. Statistical features (length, complexity metrics)
    """

    def __init__(self) -> None:
        """Initialize feature extractor with required models."""
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=FEATURE_CONFIG.tfidf_max_features,
            ngram_range=FEATURE_CONFIG.tfidf_ngram_range,
            min_df=FEATURE_CONFIG.tfidf_min_df,
            max_df=FEATURE_CONFIG.tfidf_max_df,
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.preprocessor = TextPreprocessor()
        self.is_fitted = False

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using a simple heuristic.

        Args:
            word: A single word

        Returns:
            Estimated syllable count
        """
        word = word.lower()
        if len(word) <= 3:
            return 1

        # Remove trailing 'e' (silent e)
        word = re.sub(r"e$", "", word)

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        return max(1, count)

    def _get_max_dependency_depth(self, doc: spacy.tokens.Doc) -> int:
        """
        Calculate maximum dependency tree depth.

        Args:
            doc: spaCy Doc object

        Returns:
            Maximum depth of dependency tree
        """

        def get_depth(token: spacy.tokens.Token) -> int:
            if not list(token.children):
                return 0
            return 1 + max(get_depth(child) for child in token.children)

        if not doc:
            return 0

        max_depth = 0
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    depth = get_depth(token)
                    max_depth = max(max_depth, depth)

        return max_depth

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic quality features.

        Features extracted:
        1. avg_word_length: Average word length (vocabulary sophistication)
        2. lexical_diversity: Unique words / total words
        3. avg_sentence_length: Average sentence length (structure complexity)
        4. sentence_count: Number of sentences
        5. pos_diversity: Unique POS tags / total tokens
        6. named_entity_count: Number of named entities
        7. max_dependency_depth: Maximum dependency tree depth
        8. spelling_error_ratio: Words not in spaCy vocab / total words

        Args:
            text: Essay text

        Returns:
            Dictionary of feature_name -> value

        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_linguistic_features("This is a test.")
            >>> 'avg_word_length' in features
            True
        """
        doc = self.nlp(text)
        processed = self.preprocessor.preprocess(text)

        tokens: List[str] = processed["tokens"]
        sentences: List[str] = processed["sentences"]
        pos_tags: List[tuple[str, str]] = processed["pos_tags"]

        # Filter to only alphabetic tokens for word-based metrics
        word_tokens = [t for t in tokens if t.isalpha()]

        features: Dict[str, float] = {}

        # Average word length
        if word_tokens:
            features["avg_word_length"] = sum(len(w) for w in word_tokens) / len(
                word_tokens
            )
        else:
            features["avg_word_length"] = 0.0

        # Lexical diversity (type-token ratio)
        if word_tokens:
            features["lexical_diversity"] = len(set(word_tokens)) / len(word_tokens)
        else:
            features["lexical_diversity"] = 0.0

        # Average sentence length (in words)
        if sentences:
            sentence_lengths = [
                len(self.preprocessor.tokenize_words(s)) for s in sentences
            ]
            features["avg_sentence_length"] = sum(sentence_lengths) / len(sentences)
        else:
            features["avg_sentence_length"] = 0.0

        # Sentence count
        features["sentence_count"] = float(len(sentences))

        # POS diversity
        if pos_tags:
            unique_pos = set(tag for _, tag in pos_tags)
            features["pos_diversity"] = len(unique_pos) / len(pos_tags)
        else:
            features["pos_diversity"] = 0.0

        # Named entity count
        features["named_entity_count"] = float(len(doc.ents))

        # Max dependency tree depth
        features["max_dependency_depth"] = float(self._get_max_dependency_depth(doc))

        # Spelling error ratio (words not in spaCy vocab)
        if word_tokens:
            oov_count = sum(1 for token in doc if token.is_oov and token.is_alpha)
            total_alpha = sum(1 for token in doc if token.is_alpha)
            features["spelling_error_ratio"] = (
                oov_count / total_alpha if total_alpha > 0 else 0.0
            )
        else:
            features["spelling_error_ratio"] = 0.0

        return features

    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """
        Extract readability metrics.

        Features:
        1. flesch_reading_ease: Flesch Reading Ease score
        2. avg_syllables_per_word: Average syllables per word
        3. complex_word_ratio: Words with 3+ syllables / total words

        Args:
            text: Essay text

        Returns:
            Dictionary of readability features

        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_readability_features("Simple text.")
            >>> 'flesch_reading_ease' in features
            True
        """
        processed = self.preprocessor.preprocess(text)
        tokens: List[str] = processed["tokens"]
        sentences: List[str] = processed["sentences"]

        # Filter to only alphabetic tokens
        word_tokens = [t for t in tokens if t.isalpha()]

        features: Dict[str, float] = {}

        if not word_tokens or not sentences:
            return {
                "flesch_reading_ease": 0.0,
                "avg_syllables_per_word": 0.0,
                "complex_word_ratio": 0.0,
            }

        # Count syllables
        syllable_counts = [self._count_syllables(w) for w in word_tokens]
        total_syllables = sum(syllable_counts)
        total_words = len(word_tokens)
        total_sentences = len(sentences)

        # Average syllables per word
        features["avg_syllables_per_word"] = total_syllables / total_words

        # Flesch Reading Ease: 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
        words_per_sentence = total_words / total_sentences
        syllables_per_word = total_syllables / total_words
        features["flesch_reading_ease"] = (
            206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        )

        # Complex word ratio (words with 3+ syllables)
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        features["complex_word_ratio"] = complex_words / total_words

        return features

    def extract_structural_features(self, text: str) -> Dict[str, float]:
        """
        Extract essay structure features.

        Features:
        1. paragraph_count: Number of paragraphs
        2. avg_paragraph_length: Average paragraph length in words
        3. intro_conclusion_ratio: First paragraph / last paragraph length ratio
        4. transition_word_count: Count of transition words

        Args:
            text: Essay text

        Returns:
            Dictionary of structural features

        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_structural_features("Para 1.\\n\\nPara 2.")
            >>> 'paragraph_count' in features
            True
        """
        transition_words = {
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "consequently",
            "nevertheless",
            "thus",
            "hence",
            "additionally",
            "similarly",
            "conversely",
            "meanwhile",
            "although",
            "whereas",
            "despite",
            "finally",
            "firstly",
            "secondly",
            "lastly",
        }

        features: Dict[str, float] = {}

        # Split into paragraphs (by double newlines or multiple newlines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        if not paragraphs:
            return {
                "paragraph_count": 0.0,
                "avg_paragraph_length": 0.0,
                "intro_conclusion_ratio": 0.0,
                "transition_word_count": 0.0,
            }

        # Paragraph count
        features["paragraph_count"] = float(len(paragraphs))

        # Average paragraph length (in words)
        para_lengths = [len(self.preprocessor.tokenize_words(p)) for p in paragraphs]
        features["avg_paragraph_length"] = sum(para_lengths) / len(paragraphs)

        # Intro/conclusion ratio
        if len(paragraphs) >= 2 and para_lengths[-1] > 0:
            features["intro_conclusion_ratio"] = para_lengths[0] / para_lengths[-1]
        else:
            features["intro_conclusion_ratio"] = 1.0

        # Transition word count
        tokens = self.preprocessor.tokenize_words(text)
        transition_count = sum(1 for t in tokens if t in transition_words)
        features["transition_word_count"] = float(transition_count)

        return features

    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on corpus.

        Must be called during training before extract_tfidf_features.
        Automatically adjusts min_df for small corpora.

        Args:
            texts: List of essay texts for fitting vocabulary

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.fit_tfidf(["Essay one.", "Essay two."])
            >>> extractor.is_fitted
            True
        """
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]

        # Adjust min_df for small corpora to avoid empty vocabulary
        n_docs = len(cleaned_texts)
        min_df = FEATURE_CONFIG.tfidf_min_df
        if n_docs < 5:
            min_df = 1  # Use min_df=1 for very small corpora

        # Recreate vectorizer with adjusted parameters
        self.vectorizer = TfidfVectorizer(
            max_features=FEATURE_CONFIG.tfidf_max_features,
            ngram_range=FEATURE_CONFIG.tfidf_ngram_range,
            min_df=min_df,
            max_df=FEATURE_CONFIG.tfidf_max_df,
        )

        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True

    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features.

        Args:
            texts: List of essay texts

        Returns:
            Matrix of shape (n_essays, n_features)

        Raises:
            ValueError: If vectorizer not fitted

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.fit_tfidf(["Sample essay text."])
            >>> tfidf = extractor.extract_tfidf_features(["Sample essay text."])
            >>> tfidf.shape[0]
            1
        """
        if not self.is_fitted:
            raise ValueError("Call fit_tfidf() first during training")

        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        result: np.ndarray = self.vectorizer.transform(cleaned_texts).toarray()
        return result

    def extract_all_features(self, text: str) -> np.ndarray:
        """
        Extract all feature types and concatenate.

        For single essay scoring (after training).

        Args:
            text: Essay text

        Returns:
            Feature vector combining all feature types

        Raises:
            ValueError: If TF-IDF vectorizer not fitted

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.fit_tfidf(["Sample text for fitting."])
            >>> features = extractor.extract_all_features("Test essay.")
            >>> isinstance(features, np.ndarray)
            True
        """
        if not self.is_fitted:
            raise ValueError("Call fit_tfidf() first during training")

        # Extract all feature types
        linguistic = self.extract_linguistic_features(text)
        readability = self.extract_readability_features(text)
        structural = self.extract_structural_features(text)
        tfidf = self.extract_tfidf_features([text])[0]

        # Combine non-tfidf features into array
        non_tfidf_features = np.array(
            list(linguistic.values())
            + list(readability.values())
            + list(structural.values())
        )

        # Concatenate all features
        all_features = np.concatenate([non_tfidf_features, tfidf])

        return all_features

    def get_feature_names(self) -> List[str]:
        """
        Get names of all features in order.

        Returns:
            List of feature names

        Example:
            >>> extractor = FeatureExtractor()
            >>> extractor.fit_tfidf(["Sample."])
            >>> names = extractor.get_feature_names()
            >>> 'avg_word_length' in names
            True
        """
        linguistic_names = [
            "avg_word_length",
            "lexical_diversity",
            "avg_sentence_length",
            "sentence_count",
            "pos_diversity",
            "named_entity_count",
            "max_dependency_depth",
            "spelling_error_ratio",
        ]
        readability_names = [
            "flesch_reading_ease",
            "avg_syllables_per_word",
            "complex_word_ratio",
        ]
        structural_names = [
            "paragraph_count",
            "avg_paragraph_length",
            "intro_conclusion_ratio",
            "transition_word_count",
        ]

        tfidf_names = []
        if self.is_fitted:
            tfidf_names = [
                f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()
            ]

        return linguistic_names + readability_names + structural_names + tfidf_names
