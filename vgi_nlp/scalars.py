"""Per-row NLP enrichment as DuckDB scalar functions.

Each function maps one text value to one output value, so it drops straight into
a ``SELECT`` list or ``WHERE`` clause:

    SELECT nlp.detect_lang(review) AS lang, count(*) FROM reviews GROUP BY 1;
    SELECT id, nlp.sentiment(body) AS score FROM reviews;

Language ID and sentiment are model-backed but cheap per row; the lemmatize /
strip_stopwords helpers run a spaCy pipeline (auto-detected per row unless ``lang``
is pinned) and reduce the doc back to a single string. ``normalize`` is pure
Python (no model) -- Unicode NFKC + casefold + whitespace collapse.

A note on argument syntax
-------------------------
DuckDB *scalar* functions take **positional** arguments and resolve overloads by
arity -- the ``name := value`` named-argument syntax is a property of table
functions and macros, not scalars. So the spaCy-backed cleaners expose their
``lang`` / ``model`` options as positional arity overloads (mirroring how
``vgi-translate`` does ``translate(text, 'es')`` vs ``translate(text, 'es', 'en')``):

    SELECT nlp.lemmatize(body)             FROM reviews;  -- per-row auto-detect
    SELECT nlp.lemmatize(body, 'en')       FROM reviews;  -- pin the language
    SELECT nlp.lemmatize(body, 'en', 'en_core_web_trf') FROM reviews;  -- pin model

The *table* functions (entities / tokens / sentences / noun_chunks) keep the
``id := 'id'``, ``lang := 'en'``, ``model := '...'`` named-argument form, which IS
supported for table functions.

NULL / empty input yields NULL output throughout.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from typing import Annotated, Any, cast

import pyarrow as pa
from vgi import Param, Returns, ScalarFunction
from vgi.arguments import ConstParam
from vgi.metadata import FunctionExample

from . import pipelines


def _ex(sql: str, description: str) -> list[FunctionExample]:
    return [FunctionExample(sql=sql, description=description)]


# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------


class DetectLang(ScalarFunction):
    """ISO-639 language code for each text (fastText lid.176)."""

    class Meta:
        """Function metadata."""

        name = "detect_lang"
        description = "Detect the dominant language of each text (ISO-639 code, fastText lid.176)"
        categories = ["language-id"]
        examples = _ex(
            "SELECT nlp.detect_lang(review) AS lang, count(*) FROM reviews GROUP BY 1",
            "Language histogram over a reviews column",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to identify")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        out = [pipelines.detect_language(t)[0] for t in text.to_pylist()]
        return pa.array(out, type=pa.string())


class DetectLangConf(ScalarFunction):
    """Confidence (0-1) of the detected language for each text."""

    class Meta:
        """Function metadata."""

        name = "detect_lang_conf"
        description = "Confidence (0-1) of the detected language (fastText lid.176)"
        categories = ["language-id"]
        examples = _ex(
            "SELECT body FROM docs WHERE nlp.detect_lang_conf(body) > 0.8",
            "Keep only confidently-identified rows",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to identify")],
    ) -> Annotated[pa.FloatArray, Returns(pa.float32())]:
        """Map each input row to its output value."""
        out = [pipelines.detect_language(t)[1] for t in text.to_pylist()]
        return pa.array(out, type=pa.float32())


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


class Sentiment(ScalarFunction):
    """VADER compound sentiment score in [-1, 1] for each text."""

    class Meta:
        """Function metadata."""

        name = "sentiment"
        description = "Sentiment score in [-1, 1] (VADER lexicon; tuned for English/social text)"
        categories = ["sentiment"]
        examples = _ex(
            "SELECT product, avg(nlp.sentiment(body)) AS mood FROM reviews GROUP BY product",
            "Average sentiment per product",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to score")],
    ) -> Annotated[pa.FloatArray, Returns(pa.float32())]:
        """Map each input row to its output value."""
        out = [pipelines.vader_compound(t) for t in text.to_pylist()]
        return pa.array(out, type=pa.float32())


class SentimentLabel(ScalarFunction):
    """Coarse sentiment label (neg / neu / pos) for each text."""

    class Meta:
        """Function metadata."""

        name = "sentiment_label"
        description = "Coarse sentiment label: neg / neu / pos (VADER thresholds)"
        categories = ["sentiment"]
        examples = _ex(
            "SELECT nlp.sentiment_label(body) AS mood, count(*) FROM reviews GROUP BY 1",
            "Distribution of sentiment labels",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to label")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        out = [pipelines.sentiment_label_from_score(pipelines.vader_compound(t)) for t in text.to_pylist()]
        return pa.array(out, type=pa.string())


# ---------------------------------------------------------------------------
# Cleaning helpers (spaCy-backed)
# ---------------------------------------------------------------------------


# Each spaCy-backed cleaner is a Doc -> str reduction. DuckDB scalars resolve
# overloads by arity (not by name), and ConstParam has no default-value mechanism
# at the catalog level, so every cleaner is exposed as three arity overloads
# sharing one name: (text), (text, lang), and (text, lang, model).
def _lemma_reduce(doc: Any) -> str:
    return " ".join(tok.lemma_ for tok in doc)


def _strip_reduce(doc: Any) -> str:
    return " ".join(tok.text for tok in doc if not tok.is_stop and not tok.is_punct)


class Lemmatize(ScalarFunction):
    """``lemmatize(text)`` -- lemmatize each text, auto-detecting the language per row."""

    class Meta:
        """Function metadata."""

        name = "lemmatize"
        description = "Lemmatize each text (tokens replaced by their dictionary form); language auto-detected"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.lemmatize(body) FROM reviews",
            "Lemmatize a column, auto-detecting each row's language",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to lemmatize")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), None, None, _lemma_reduce)


class LemmatizeLang(ScalarFunction):
    """``lemmatize(text, lang)`` -- lemmatize with the pipeline language pinned."""

    class Meta:
        """Function metadata."""

        name = "lemmatize"
        description = "Lemmatize each text with the pipeline language pinned (ISO-639 code)"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.lemmatize(body, 'en') FROM reviews",
            "Lemmatize an English column",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to lemmatize")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639), e.g. 'en'")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), lang or None, None, _lemma_reduce)


class LemmatizeModel(ScalarFunction):
    """``lemmatize(text, lang, model)`` -- lemmatize with an explicit spaCy model."""

    class Meta:
        """Function metadata."""

        name = "lemmatize"
        description = "Lemmatize each text with an explicit spaCy model (e.g. en_core_web_trf)"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.lemmatize(body, 'en', 'en_core_web_trf') FROM reviews",
            "Lemmatize with a specific spaCy model",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to lemmatize")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639); '' = ignore when model is set")],
        model: Annotated[str, ConstParam(doc="spaCy model name, e.g. 'en_core_web_trf'")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), lang or None, model or None, _lemma_reduce)


class StripStopwords(ScalarFunction):
    """``strip_stopwords(text)`` -- drop stop-words/punctuation, auto-detecting language."""

    class Meta:
        """Function metadata."""

        name = "strip_stopwords"
        description = "Remove stop-words and punctuation, returning the rest joined; language auto-detected"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.strip_stopwords(body) FROM reviews",
            "Strip stop-words, auto-detecting each row's language",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to clean")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), None, None, _strip_reduce)


class StripStopwordsLang(ScalarFunction):
    """``strip_stopwords(text, lang)`` -- strip stop-words with the language pinned."""

    class Meta:
        """Function metadata."""

        name = "strip_stopwords"
        description = "Remove stop-words and punctuation with the pipeline language pinned (ISO-639)"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.strip_stopwords(body, 'en') FROM reviews",
            "Strip English stop-words",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to clean")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639), e.g. 'en'")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), lang or None, None, _strip_reduce)


class StripStopwordsModel(ScalarFunction):
    """``strip_stopwords(text, lang, model)`` -- strip with an explicit spaCy model."""

    class Meta:
        """Function metadata."""

        name = "strip_stopwords"
        description = "Remove stop-words and punctuation with an explicit spaCy model"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.strip_stopwords(body, 'en', 'en_core_web_trf') FROM reviews",
            "Strip stop-words with a specific spaCy model",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to clean")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639); '' = ignore when model is set")],
        model: Annotated[str, ConstParam(doc="spaCy model name, e.g. 'en_core_web_trf'")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        return _spacy_map(text.to_pylist(), lang or None, model or None, _strip_reduce)


class Normalize(ScalarFunction):
    """Unicode NFKC + casefold + whitespace collapse (no model needed)."""

    class Meta:
        """Function metadata."""

        name = "normalize"
        description = "Normalize text: Unicode NFKC, lowercase, and collapse whitespace"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.normalize(body) FROM reviews",
            "Canonicalize text for dedup / matching",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to normalize")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Map each input row to its output value."""
        out: list[str | None] = []
        for t in text.to_pylist():
            if t is None:
                out.append(None)
                continue
            norm = unicodedata.normalize("NFKC", t)
            out.append(" ".join(norm.casefold().split()))
        return pa.array(out, type=pa.string())


# ---------------------------------------------------------------------------
# Shared spaCy scalar helper
# ---------------------------------------------------------------------------


def _spacy_map(
    texts: list[str | None],
    lang: str | None,
    model: str | None,
    reduce: Callable[[Any], str],
) -> pa.StringArray:
    """Run a spaCy pipeline over ``texts`` and reduce each Doc to a string.

    Rows are grouped by pipeline so ``nlp.pipe()`` batches each language once.
    NULL/empty rows (and rows whose auto-detected language has no pipeline) map to
    NULL, preserving row order.
    """
    results: list[str | None] = [None] * len(texts)
    buckets = pipelines.group_by_pipeline(texts, lang=lang, model=model)
    for model_name, idxs in buckets.items():
        pipe = pipelines.load_spacy_by_name(model_name)
        # group_by_pipeline only buckets indices whose text is a non-empty str,
        # so texts[i] is never None here.
        docs = pipe.pipe(
            (cast(str, texts[i]) for i in idxs),
            batch_size=pipelines.batch_size(),
        )
        for i, doc in zip(idxs, docs, strict=False):
            results[i] = reduce(doc)
    return pa.array(results, type=pa.string())


SCALAR_FUNCTIONS: list[type] = [
    DetectLang,
    DetectLangConf,
    Sentiment,
    SentimentLabel,
    Lemmatize,
    LemmatizeLang,
    LemmatizeModel,
    StripStopwords,
    StripStopwordsLang,
    StripStopwordsModel,
    Normalize,
]
