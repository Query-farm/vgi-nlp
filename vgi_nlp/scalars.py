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
from .meta import object_tags

_SRC = "vgi_nlp/scalars.py"


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
            "SELECT nlp.detect_lang('The quick brown fox jumps over the lazy dog') AS lang",
            "Identify the language of a literal string",
        )
        tags = object_tags(
            "Detect Language Code",
            "Identify the dominant natural language of each input string and return its "
            "two-letter ISO-639-1 code (`en`, `fr`, `de`, `es`, ...), using the fastText "
            "`lid.176` model.\n\n"
            "**When to use:** routing mixed-language corpora, filtering a table down to one "
            "language before applying English-only models (e.g. VADER sentiment), or building "
            "a language histogram with `GROUP BY`.\n\n"
            "**Input:** one VARCHAR column. **Output:** a VARCHAR ISO-639 code. NULL or empty "
            "input yields NULL. Very short or ambiguous strings can be misclassified -- pair "
            "with `detect_lang_conf` and threshold on confidence when accuracy matters.",
            "# detect_lang\n\n"
            "Returns the ISO-639-1 language code of each text value, computed with the "
            "fastText `lid.176` identifier.\n\n"
            "```sql\n"
            "SELECT nlp.detect_lang('Bonjour le monde');  -- 'fr'\n"
            "```\n\n"
            "Use it to segment a multilingual column before language-specific processing. "
            "Combine with `nlp.detect_lang_conf` to discard low-confidence guesses; very short "
            "snippets are inherently hard to classify.",
            "detect language, language identification, language detection, langid, lid, "
            "iso-639, locale, fasttext, what language",
            _SRC,
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
            "SELECT nlp.detect_lang_conf('The quick brown fox') AS conf",
            "Confidence of the top language guess for a literal string",
        )
        tags = object_tags(
            "Detect Language Confidence Score",
            "Return the fastText `lid.176` confidence (roughly 0-1) for the **top** language "
            "guess of each input string -- the companion probability to `detect_lang`.\n\n"
            "**When to use:** gate downstream processing on identification quality, e.g. keep "
            "only rows the model is sure about with `WHERE nlp.detect_lang_conf(body) > 0.8`.\n\n"
            "**Input:** one VARCHAR column. **Output:** a FLOAT. NULL/empty input yields NULL. "
            "Edge case: fastText occasionally reports a confidence marginally above 1.0 (e.g. "
            "~1.00001), so do not assert a strict `<= 1.0` upper bound on the result.",
            "# detect_lang_conf\n\n"
            "Returns how confident the fastText `lid.176` model is in the language it picked "
            "for each text -- a float near the `[0, 1]` range.\n\n"
            "```sql\n"
            "SELECT body\n"
            "FROM docs\n"
            "WHERE nlp.detect_lang_conf(body) > 0.8;  -- confidently-identified rows only\n"
            "```\n\n"
            "Use alongside `nlp.detect_lang`. Note the confidence can read slightly above 1.0 "
            "for some inputs, so avoid a strict upper-bound assertion.",
            "language confidence, langid confidence, detection probability, language score, "
            "fasttext confidence, certainty, threshold",
            _SRC,
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
            "SELECT nlp.sentiment('I absolutely love this, it is fantastic!') AS score",
            "Score the sentiment of a literal English string",
        )
        tags = object_tags(
            "Sentiment Compound Score",
            "Score the sentiment of each text with VADER and return the **compound** value in "
            "`[-1, 1]`, where `-1` is maximally negative, `0` neutral, and `+1` maximally "
            "positive.\n\n"
            "**When to use:** rank or aggregate opinion in reviews, comments, tweets, and other "
            "short social/English text -- e.g. `avg(nlp.sentiment(body))` per product.\n\n"
            "**Input:** one VARCHAR column. **Output:** a FLOAT in `[-1, 1]`. NULL/empty input "
            "yields NULL. VADER is a lexicon-and-rules model tuned for English and social media; "
            "it understands emphasis, negation, and emoticons but is not a translator -- score "
            "non-English text only after detecting/translating it.",
            "# sentiment\n\n"
            "Returns the VADER compound sentiment score in `[-1, 1]` for each text.\n\n"
            "```sql\n"
            "SELECT product, avg(nlp.sentiment(body)) AS mood\n"
            "FROM reviews\n"
            "GROUP BY product;\n"
            "```\n\n"
            "VADER is purpose-built for English social text (handling negation, intensifiers, "
            "and emoji). For a coarse neg/neu/pos bucket instead of a number, use "
            "`nlp.sentiment_label`.",
            "sentiment, sentiment analysis, vader, opinion, polarity, mood, positive negative, compound score, reviews",
            _SRC,
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
            "SELECT nlp.sentiment_label('This is the worst experience ever') AS mood",
            "Bucket a literal string into neg / neu / pos",
        )
        tags = object_tags(
            "Sentiment Polarity Label",
            "Bucket each text into a coarse three-way sentiment **label** -- `neg`, `neu`, or "
            "`pos` -- by thresholding the VADER compound score.\n\n"
            "**When to use:** when you want a categorical opinion bucket for `GROUP BY` / "
            "filtering rather than a continuous number; cheaper to read in dashboards than the "
            "raw score.\n\n"
            "**Input:** one VARCHAR column. **Output:** a VARCHAR in `{neg, neu, pos}`. "
            "NULL/empty input yields NULL. Same English/social-text caveats as `sentiment`; if "
            "you need the underlying magnitude, call `nlp.sentiment` instead.",
            "# sentiment_label\n\n"
            "Returns a coarse sentiment class -- `neg`, `neu`, or `pos` -- derived from the "
            "VADER compound score.\n\n"
            "```sql\n"
            "SELECT nlp.sentiment_label(body) AS mood, count(*)\n"
            "FROM reviews\n"
            "GROUP BY 1;\n"
            "```\n\n"
            "Use this when a label is more convenient than the numeric `nlp.sentiment` score, "
            "for example to tally how many reviews fall into each mood bucket.",
            "sentiment label, sentiment class, positive negative neutral, neg neu pos, "
            "polarity bucket, vader, opinion category",
            _SRC,
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
            "SELECT nlp.lemmatize('The cats were running quickly') AS lemmas",
            "Lemmatize a literal string with per-row language auto-detect",
        )
        tags = object_tags(
            "Lemmatize Text (Auto-Detect Language)",
            "Reduce every token to its dictionary base form (lemma) and return the lemmas "
            "joined back into a single string, auto-detecting each row's language via fastText "
            "before running the matching spaCy pipeline.\n\n"
            "**When to use:** normalize inflected words (`running`/`ran` -> `run`, `cats` -> "
            "`cat`) before keyword search, deduplication, or bag-of-words featurization.\n\n"
            "**Input:** one VARCHAR column. **Output:** a VARCHAR of space-joined lemmas. "
            "NULL/empty input -- and rows whose detected language has no installed spaCy "
            "pipeline -- yield NULL. Pin the language with the `(text, lang)` overload when the "
            "corpus is monolingual to skip per-row detection and its throughput cost.",
            "# lemmatize(text)\n\n"
            "Lemmatizes each text with spaCy, auto-detecting the language per row, and returns "
            "the space-joined lemmas.\n\n"
            "```sql\n"
            "SELECT nlp.lemmatize(body) FROM reviews;  -- 'the cat be run quickly'\n"
            "```\n\n"
            "For monolingual data prefer `nlp.lemmatize(text, 'en')` to pin the language and "
            "avoid per-row detection overhead; pass a third argument to choose a specific spaCy "
            "model.",
            "lemmatize, lemmatization, lemma, dictionary form, base form, stemming, normalize "
            "words, spacy, text cleaning",
            _SRC,
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
            "SELECT nlp.lemmatize('The cats were running quickly', 'en') AS lemmas",
            "Lemmatize a literal string with the language pinned to English",
        )
        tags = object_tags(
            "Lemmatize Text (Pinned Language)",
            "Reduce every token to its lemma and return them space-joined, using the spaCy "
            "pipeline for the **pinned** ISO-639 language instead of per-row auto-detection.\n\n"
            "**When to use:** monolingual corpora -- pinning `lang` (e.g. `'en'`) skips fastText "
            "language detection on every row, which is faster and avoids mis-detection on short "
            "or ambiguous strings.\n\n"
            "**Inputs:** a VARCHAR text column and a constant ISO-639 `lang` code. **Output:** a "
            "VARCHAR of space-joined lemmas. NULL/empty text yields NULL; an unknown/unsupported "
            "language (no installed pipeline) also yields NULL. Use the `(text, lang, model)` "
            "overload to name a specific spaCy model.",
            "# lemmatize(text, lang)\n\n"
            "Like `lemmatize(text)`, but the spaCy pipeline language is fixed to the supplied "
            "ISO-639 code rather than auto-detected.\n\n"
            "```sql\n"
            "SELECT nlp.lemmatize(body, 'en') FROM reviews;\n"
            "```\n\n"
            "Prefer this overload whenever the column is known to be a single language: it is "
            "faster and more reliable than auto-detect. Add a third `model` argument to override "
            "the default spaCy model for that language.",
            "lemmatize language, lemmatize pinned, lemma, dictionary form, spacy pipeline, "
            "iso-639, monolingual, text cleaning",
            _SRC,
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
        description = "Lemmatize each text with an explicit spaCy model (e.g. en_core_web_sm)"
        categories = ["cleaning"]
        examples = _ex(
            "SELECT nlp.lemmatize('The cats were running quickly', 'en', 'en_core_web_sm') AS lemmas",
            "Lemmatize a literal string with an explicit spaCy model",
        )
        tags = object_tags(
            "Lemmatize Text (Explicit Model)",
            "Reduce every token to its lemma and return them space-joined, loading a **named** "
            "spaCy model rather than the default for the language.\n\n"
            "**When to use:** when you need a particular model -- e.g. a larger/transformer "
            "model (`en_core_web_trf`) for higher accuracy, or a non-default model already "
            "installed in the worker environment.\n\n"
            "**Inputs:** a VARCHAR text column, an ISO-639 `lang` (`''` ignores it when a model "
            "is given), and a `model` name. **Output:** a VARCHAR of space-joined lemmas. "
            "NULL/empty text yields NULL. The named model must be installed in the worker's "
            "environment, or the call errors at load time.",
            "# lemmatize(text, lang, model)\n\n"
            "Lemmatizes each text using an explicitly named spaCy model, giving full control "
            "over accuracy/speed trade-offs.\n\n"
            "```sql\n"
            "SELECT nlp.lemmatize(body, 'en', 'en_core_web_sm') FROM reviews;\n"
            "```\n\n"
            "The model must be present in the worker's Python environment. When the `lang` "
            "argument is empty the model name alone selects the pipeline.",
            "lemmatize model, custom spacy model, en_core_web_sm, en_core_web_trf, lemma, "
            "explicit model, text cleaning, pipeline",
            _SRC,
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
            "SELECT nlp.strip_stopwords('this is a really great movie') AS kept",
            "Drop stop-words from a literal string with per-row auto-detect",
        )
        tags = object_tags(
            "Strip Stop-Words (Auto-Detect Language)",
            "Drop stop-words and punctuation from each text and return the surviving tokens "
            "space-joined, auto-detecting each row's language before applying that language's "
            "spaCy stop-word list.\n\n"
            "**When to use:** strip low-signal filler (`the`, `is`, `a`, ...) before keyword "
            "extraction, TF-IDF, or similarity search so the content words dominate.\n\n"
            "**Input:** one VARCHAR column. **Output:** a VARCHAR of the kept tokens. NULL/empty "
            "input -- and rows whose detected language has no spaCy pipeline -- yield NULL. Pin "
            "the language with the `(text, lang)` overload on monolingual data to skip per-row "
            "detection.",
            "# strip_stopwords(text)\n\n"
            "Removes stop-words and punctuation, returning the remaining content tokens joined "
            "by spaces; the language is auto-detected per row.\n\n"
            "```sql\n"
            "SELECT nlp.strip_stopwords(body) FROM reviews;  -- 'really great movie'\n"
            "```\n\n"
            "For single-language columns prefer `nlp.strip_stopwords(text, 'en')` to pin the "
            "language and skip detection; a third argument selects a specific spaCy model.",
            "stop words, stopword removal, remove stopwords, filter words, content words, "
            "text cleaning, preprocessing, spacy",
            _SRC,
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
            "SELECT nlp.strip_stopwords('this is a really great movie', 'en') AS kept",
            "Drop English stop-words from a literal string",
        )
        tags = object_tags(
            "Strip Stop-Words (Pinned Language)",
            "Drop stop-words and punctuation and return the surviving tokens space-joined, "
            "using the spaCy stop-word list for the **pinned** ISO-639 language instead of "
            "per-row auto-detection.\n\n"
            "**When to use:** monolingual corpora -- pinning `lang` (e.g. `'en'`) skips fastText "
            "detection on every row, which is faster and avoids mis-detection on short strings."
            "\n\n"
            "**Inputs:** a VARCHAR text column and a constant ISO-639 `lang` code. **Output:** a "
            "VARCHAR of the kept tokens. NULL/empty text yields NULL; an unsupported language "
            "(no installed pipeline) also yields NULL. Use the `(text, lang, model)` overload "
            "to name a specific spaCy model.",
            "# strip_stopwords(text, lang)\n\n"
            "Like `strip_stopwords(text)`, but the spaCy stop-word list is fixed to the supplied "
            "ISO-639 language rather than auto-detected.\n\n"
            "```sql\n"
            "SELECT nlp.strip_stopwords(body, 'en') FROM reviews;\n"
            "```\n\n"
            "Prefer this overload for single-language columns: it is faster and more reliable "
            "than auto-detect. A third `model` argument overrides the default spaCy model.",
            "stop words pinned, remove stopwords language, filter words, content words, "
            "iso-639, monolingual, text cleaning, spacy",
            _SRC,
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
            "SELECT nlp.strip_stopwords('this is a really great movie', 'en', 'en_core_web_sm') AS kept",
            "Drop stop-words from a literal string with an explicit spaCy model",
        )
        tags = object_tags(
            "Strip Stop-Words (Explicit Model)",
            "Drop stop-words and punctuation and return the surviving tokens space-joined, "
            "loading a **named** spaCy model rather than the language default.\n\n"
            "**When to use:** when a particular model is required -- a larger/transformer model "
            "for better tokenization, or a non-default model already installed in the worker.\n\n"
            "**Inputs:** a VARCHAR text column, an ISO-639 `lang` (`''` ignores it when a model "
            "is given), and a `model` name. **Output:** a VARCHAR of the kept tokens. NULL/empty "
            "text yields NULL. The named model must be installed in the worker's environment or "
            "the call errors at load time.",
            "# strip_stopwords(text, lang, model)\n\n"
            "Removes stop-words and punctuation using an explicitly named spaCy model.\n\n"
            "```sql\n"
            "SELECT nlp.strip_stopwords(body, 'en', 'en_core_web_sm') FROM reviews;\n"
            "```\n\n"
            "The model must be present in the worker's Python environment. When `lang` is empty "
            "the model name alone selects the pipeline.",
            "stop words model, custom spacy model, remove stopwords, en_core_web_sm, "
            "explicit model, content words, text cleaning",
            _SRC,
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
            "SELECT nlp.normalize('  Café   DELUXE\t—  ') AS canonical",
            "Canonicalize a messy literal string for dedup / matching",
        )
        tags = object_tags(
            "Normalize Text Form",
            "Canonicalize each text into a comparable form: apply Unicode **NFKC** "
            "normalization, casefold (aggressive lowercase), and collapse all runs of "
            "whitespace to single spaces (trimming the ends).\n\n"
            "**When to use:** before equality joins, deduplication, or grouping on free-text "
            "keys so that visually-identical strings differing only in case, full/half-width "
            "forms, ligatures, or spacing compare equal.\n\n"
            "**Input:** one VARCHAR column. **Output:** a normalized VARCHAR. NULL input yields "
            "NULL. This function is **pure Python** -- no spaCy/fastText model is loaded -- so "
            "it is the cheapest function in the worker and always available.",
            "# normalize\n\n"
            "Returns a canonical form of each text: Unicode NFKC + casefold + whitespace "
            "collapse.\n\n"
            "```sql\n"
            "SELECT nlp.normalize(body) FROM reviews;  -- 'café deluxe —'\n"
            "```\n\n"
            "Use it to make free-text keys join and group reliably. Unlike the other cleaners "
            "this needs no model, so it runs everywhere with no setup.",
            "normalize, canonicalize, unicode nfkc, casefold, lowercase, collapse whitespace, "
            "dedup, matching, text cleaning",
            _SRC,
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
