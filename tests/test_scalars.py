"""Tests for the scalar NLP functions (compute() array-in / array-out)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.harness import fasttext_available, spacy_model_available
from vgi_nlp.scalars import (
    DetectLang,
    DetectLangConf,
    Lemmatize,
    Normalize,
    Sentiment,
    SentimentLabel,
    StripStopwords,
)

needs_fasttext = pytest.mark.skipif(not fasttext_available(), reason="fastText lid.176 model not installed")
needs_spacy = pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_sm not installed")


# --- normalize (no model, always runs) -------------------------------------


class TestNormalize:
    def test_lowercase_and_whitespace_collapse(self) -> None:
        out = Normalize.compute(pa.array(["  Héllo   WORLD  "])).to_pylist()
        assert out == ["héllo world"]

    def test_nfkc(self) -> None:
        # Fullwidth 'Ａ' normalises to ASCII 'a' under NFKC + casefold.
        out = Normalize.compute(pa.array(["Ａ"])).to_pylist()
        assert out == ["a"]

    def test_null_passthrough(self) -> None:
        out = Normalize.compute(pa.array([None, "X"])).to_pylist()
        assert out == [None, "x"]


# --- language detection -----------------------------------------------------


@needs_fasttext
class TestDetectLang:
    def test_english_and_french(self) -> None:
        out = DetectLang.compute(
            pa.array(["I really love this wonderful product.", "Ceci est un texte en français."])
        ).to_pylist()
        assert out == ["en", "fr"]

    def test_null_and_empty(self) -> None:
        out = DetectLang.compute(pa.array([None, "", "  "])).to_pylist()
        assert out == [None, None, None]

    def test_confidence_in_unit_range(self) -> None:
        conf = DetectLangConf.compute(pa.array(["I really love this wonderful product."])).to_pylist()
        assert 0.0 < conf[0] <= 1.0


# --- sentiment --------------------------------------------------------------


class TestSentiment:
    def test_positive_higher_than_negative(self) -> None:
        scores = Sentiment.compute(
            pa.array(["I love this, it is wonderful!", "This is terrible and awful."])
        ).to_pylist()
        assert scores[0] > 0.3
        assert scores[1] < -0.3

    def test_null(self) -> None:
        assert Sentiment.compute(pa.array([None])).to_pylist() == [None]

    def test_labels(self) -> None:
        out = SentimentLabel.compute(
            pa.array(["I love this!", "This is terrible.", "a plain book", None])
        ).to_pylist()
        assert out == ["pos", "neg", "neu", None]


# --- spaCy-backed cleaning --------------------------------------------------


@needs_spacy
class TestCleaning:
    def test_lemmatize(self) -> None:
        out = Lemmatize.compute(pa.array(["The cats were running quickly."]), "en", "").to_pylist()
        assert "cat" in out[0]
        assert "run" in out[0]

    def test_strip_stopwords(self) -> None:
        out = StripStopwords.compute(pa.array(["This is a very good book about cats."]), "en", "").to_pylist()
        # stop-words ("this", "is", "a", "about") and punctuation are removed.
        assert "good" in out[0]
        assert "book" in out[0]
        assert "is" not in out[0].split()
