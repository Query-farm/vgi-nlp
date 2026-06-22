"""Per-row NLP enrichment as DuckDB scalar functions.

Each function maps one text value to one output value, so it drops straight into
a ``SELECT`` list or ``WHERE`` clause:

    SELECT nlp.detect_lang(review) AS lang, count(*) FROM reviews GROUP BY 1;
    SELECT id, nlp.sentiment(body) AS score FROM reviews;

Language ID and sentiment are model-backed but cheap per row; the lemmatize /
strip_stopwords helpers run a spaCy pipeline (auto-detected per row unless ``lang``
is pinned) and reduce the doc back to a single string. ``normalize`` is pure
Python (no model) -- Unicode NFKC + casefold + whitespace collapse.

Conventions shared with the table functions:

* ``lang := 'en'`` pins the pipeline language; default is per-row fastText
  auto-detect.
* ``model := 'en_core_web_trf'`` overrides the spaCy pipeline.

NULL / empty input yields NULL output throughout.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated

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
        out = [pipelines.detect_language(t)[0] for t in text.to_pylist()]
        return pa.array(out, type=pa.string())


class DetectLangConf(ScalarFunction):
    """Confidence (0-1) of the detected language for each text."""

    class Meta:
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
        out = [pipelines.detect_language(t)[1] for t in text.to_pylist()]
        return pa.array(out, type=pa.float32())


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


class Sentiment(ScalarFunction):
    """VADER compound sentiment score in [-1, 1] for each text."""

    class Meta:
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
        out = [pipelines.vader_compound(t) for t in text.to_pylist()]
        return pa.array(out, type=pa.float32())


class SentimentLabel(ScalarFunction):
    """Coarse sentiment label (neg / neu / pos) for each text."""

    class Meta:
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
        out = [
            pipelines.sentiment_label_from_score(pipelines.vader_compound(t))
            for t in text.to_pylist()
        ]
        return pa.array(out, type=pa.string())


# ---------------------------------------------------------------------------
# Cleaning helpers (spaCy-backed)
# ---------------------------------------------------------------------------


class Lemmatize(ScalarFunction):
    """Replace every token with its lemma, returning the rejoined string."""

    class Meta:
        name = "lemmatize"
        description = "Lemmatize each text (tokens replaced by their dictionary form)"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.lemmatize(body, lang := 'en') FROM reviews",
            "Lemmatize an English column",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to lemmatize")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639); '' = auto-detect per row")] = "",
        model: Annotated[str, ConstParam(doc="Override spaCy model name; '' = default for lang")] = "",
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        return _spacy_map(
            text.to_pylist(),
            lang or None,
            model or None,
            lambda doc: " ".join(tok.lemma_ for tok in doc),
        )


class StripStopwords(ScalarFunction):
    """Drop stop-words (and pure-punctuation tokens), returning the rest joined."""

    class Meta:
        name = "strip_stopwords"
        description = "Remove stop-words and punctuation, returning the remaining tokens joined"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.strip_stopwords(body, lang := 'en') FROM reviews",
            "Strip English stop-words",
        )

    @classmethod
    def compute(
        cls,
        text: Annotated[pa.StringArray, Param(doc="Text column to clean")],
        lang: Annotated[str, ConstParam(doc="Pipeline language (ISO-639); '' = auto-detect per row")] = "",
        model: Annotated[str, ConstParam(doc="Override spaCy model name; '' = default for lang")] = "",
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        return _spacy_map(
            text.to_pylist(),
            lang or None,
            model or None,
            lambda doc: " ".join(tok.text for tok in doc if not tok.is_stop and not tok.is_punct),
        )


class Normalize(ScalarFunction):
    """Unicode NFKC + casefold + whitespace collapse (no model needed)."""

    class Meta:
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


def _spacy_map(texts, lang, model, reduce) -> pa.StringArray:  # noqa: ANN001
    """Run a spaCy pipeline over ``texts`` and reduce each Doc to a string.

    Rows are grouped by pipeline so ``nlp.pipe()`` batches each language once.
    NULL/empty rows (and rows whose auto-detected language has no pipeline) map to
    NULL, preserving row order.
    """
    results: list[str | None] = [None] * len(texts)
    buckets = pipelines.group_by_pipeline(texts, lang=lang, model=model)
    for model_name, idxs in buckets.items():
        pipe = pipelines.load_spacy_by_name(model_name)
        docs = pipe.pipe(
            (texts[i] for i in idxs),
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
    StripStopwords,
    Normalize,
]
