"""Tests for the scalar NLP functions (compute() array-in / array-out).

Covers the happy path plus error/edge cases: NULL/empty input, unknown
languages and models, very long text, unicode/emoji normalization, and the
arity-overload form of the spaCy-backed cleaners (lemmatize / strip_stopwords).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.harness import fasttext_available, spacy_model_available
from vgi_nlp.pipelines import ModelNotAvailableError
from vgi_nlp.scalars import (
    DetectLang,
    DetectLangConf,
    Lemmatize,
    LemmatizeLang,
    LemmatizeModel,
    Normalize,
    Sentiment,
    SentimentLabel,
    StripStopwords,
    StripStopwordsLang,
    StripStopwordsModel,
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

    def test_empty_and_whitespace_only(self) -> None:
        # Empty string stays empty; whitespace-only collapses to empty.
        out = Normalize.compute(pa.array(["", "   \t\n  "])).to_pylist()
        assert out == ["", ""]

    def test_unicode_and_emoji(self) -> None:
        # Emoji are preserved (NFKC leaves them intact); ligatures decompose.
        out = Normalize.compute(pa.array(["CAFÉ ☕ 🎉  Ⅷ", "ﬁle"])).to_pylist()
        # 'Ⅷ' (roman numeral eight) -> 'viii' under NFKC + casefold; 'ﬁ' ligature -> 'fi'.
        assert out[0] == "café ☕ 🎉 viii"
        assert out[1] == "file"

    def test_long_text_normalizes(self) -> None:
        big = ("Hello   World  " * 5000).strip()
        out = Normalize.compute(pa.array([big])).to_pylist()
        # Whitespace collapsed: 10000 tokens joined by single spaces.
        assert out[0].count("  ") == 0
        assert out[0].split() == ["hello", "world"] * 5000


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

    def test_confidence_is_positive_probability(self) -> None:
        # fastText probabilities are near 1 but can marginally exceed 1.0, so we
        # bound loosely rather than asserting an exact <= 1.0 ceiling.
        conf = DetectLangConf.compute(pa.array(["I really love this wonderful product."])).to_pylist()
        assert 0.5 < conf[0] < 1.5

    def test_confidence_zero_for_empty(self) -> None:
        conf = DetectLangConf.compute(pa.array([None, ""])).to_pylist()
        assert conf == [0.0, 0.0]


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

    def test_empty_and_whitespace_yield_null(self) -> None:
        assert Sentiment.compute(pa.array(["", "   "])).to_pylist() == [None, None]

    def test_labels(self) -> None:
        out = SentimentLabel.compute(
            pa.array(["I love this!", "This is terrible.", "a plain book", None])
        ).to_pylist()
        assert out == ["pos", "neg", "neu", None]

    def test_label_empty_is_null(self) -> None:
        assert SentimentLabel.compute(pa.array(["", "  "])).to_pylist() == [None, None]

    def test_long_text_scores(self) -> None:
        big = "I love this wonderful amazing fantastic product. " * 2000
        out = Sentiment.compute(pa.array([big])).to_pylist()
        assert out[0] > 0.3


# --- spaCy-backed cleaning (arity overloads) --------------------------------


@needs_spacy
class TestLemmatize:
    def test_lemmatize_lang_pinned(self) -> None:
        out = LemmatizeLang.compute(pa.array(["The cats were running quickly."]), "en").to_pylist()
        assert "cat" in out[0]
        assert "run" in out[0]

    def test_lemmatize_explicit_model(self) -> None:
        out = LemmatizeModel.compute(
            pa.array(["The cats were running quickly."]), "en", "en_core_web_sm"
        ).to_pylist()
        assert "cat" in out[0]

    @needs_fasttext
    def test_lemmatize_auto_detect(self) -> None:
        # No lang/model: per-row fastText auto-detect routes English to en_core_web_sm.
        out = Lemmatize.compute(pa.array(["The cats were running quickly."])).to_pylist()
        assert "cat" in out[0]

    def test_lemmatize_null_and_empty(self) -> None:
        out = LemmatizeLang.compute(pa.array([None, "", "   "]), "en").to_pylist()
        assert out == [None, None, None]

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ModelNotAvailableError, match="not installed"):
            LemmatizeModel.compute(pa.array(["hello world"]), "en", "no_such_model_xyz").to_pylist()

    @needs_fasttext
    def test_unknown_language_raises(self) -> None:
        # 'xx' has no default spaCy pipeline -> actionable error when pinned.
        with pytest.raises(ModelNotAvailableError, match="No default spaCy pipeline"):
            LemmatizeLang.compute(pa.array(["hello world"]), "xx").to_pylist()


@needs_spacy
class TestStripStopwords:
    def test_strip_lang_pinned(self) -> None:
        out = StripStopwordsLang.compute(pa.array(["This is a very good book about cats."]), "en").to_pylist()
        # stop-words ("this", "is", "a", "about") and punctuation are removed.
        assert "good" in out[0]
        assert "book" in out[0]
        assert "is" not in out[0].split()

    def test_strip_explicit_model(self) -> None:
        out = StripStopwordsModel.compute(
            pa.array(["This is a very good book."]), "en", "en_core_web_sm"
        ).to_pylist()
        assert "good" in out[0]

    @needs_fasttext
    def test_strip_auto_detect(self) -> None:
        # No lang/model: per-row fastText auto-detect routes English to en_core_web_sm.
        out = StripStopwords.compute(pa.array(["This is a very good book about cats."])).to_pylist()
        assert "good" in out[0]
        assert "is" not in out[0].split()

    def test_strip_null_passthrough(self) -> None:
        out = StripStopwordsLang.compute(pa.array([None, ""]), "en").to_pylist()
        assert out == [None, None]

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ModelNotAvailableError, match="not installed"):
            StripStopwordsModel.compute(pa.array(["hello"]), "en", "no_such_model_xyz").to_pylist()

    def test_long_text_no_crash(self) -> None:
        big = "This is a good book about cats and dogs. " * 1000
        out = StripStopwordsLang.compute(pa.array([big]), "en").to_pylist()
        assert "book" in out[0]
        assert "is" not in out[0].split()
