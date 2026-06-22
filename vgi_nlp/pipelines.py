"""Model lifecycle: load spaCy / fastText / VADER once, cache in the worker process.

VGI keeps the worker process alive across queries, so the expensive thing a
classical-NLP worker does -- loading a model -- happens once and is amortised over
every row of every query. This module centralises that caching so the scalar and
table functions only ask for "the pipeline for language X" and get a ready spaCy
``Language`` back.

Three model families:

* **fastText ``lid.176``** -- compact (~917 KB) language identifier. Used both by
  ``detect_lang``/``detect_lang_conf`` and, when ``lang`` is not pinned, to route
  each row to the right spaCy pipeline.
* **spaCy pipelines** -- one per language, keyed by ISO-639 code (``en`` ->
  ``en_core_web_sm``) or by an explicit ``model :=`` name. Loaded on first use.
* **VADER** -- a stateless lexicon sentiment analyzer, instantiated once.

Everything is lazy: importing this module is cheap; nothing is loaded until the
first row needs it. Missing models raise a clear, actionable error rather than a
deep library traceback.
"""

from __future__ import annotations

import contextlib
import os
import threading
from functools import cache, lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fasttext
    from spacy.language import Language
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ISO-639 language code -> default small spaCy pipeline. Small (`_sm`) models are
# the v1 default: fast, ~12-50 MB, MIT/CC-BY-SA. Users opt into accuracy with
# `model := 'en_core_web_trf'`.
_DEFAULT_SPACY_MODEL: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "el": "el_core_news_sm",
    "nb": "nb_core_news_sm",
    "lt": "lt_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ca": "ca_core_news_sm",
    "da": "da_core_news_sm",
    "fi": "fi_core_news_sm",
    "hr": "hr_core_news_sm",
    "ko": "ko_core_news_sm",
    "mk": "mk_core_news_sm",
    "pl": "pl_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "sl": "sl_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
}

# Where the fastText lid.176 model lives. Override with VGI_NLP_FASTTEXT_MODEL.
# The compressed `.ftz` (917 KB) is preferred; the full `.bin` (126 MB) also works.
_FASTTEXT_ENV = "VGI_NLP_FASTTEXT_MODEL"
_FASTTEXT_FILENAMES = ("lid.176.ftz", "lid.176.bin")

_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """A required model/pipeline is not installed and could not be loaded.

    Carries an actionable hint (the exact install/download command) so the
    DuckDB-side error tells the user how to fix it.
    """


# ---------------------------------------------------------------------------
# fastText language identification
# ---------------------------------------------------------------------------


def _fasttext_model_path() -> str:
    """Locate the fastText lid.176 model file, or raise a helpful error."""
    explicit = os.environ.get(_FASTTEXT_ENV)
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise ModelNotAvailableError(
            f"{_FASTTEXT_ENV}={explicit!r} does not point to an existing file."
        )
    # Search a few conventional locations: CWD, ~/.cache/vgi-nlp, package dir.
    search_dirs = [
        os.getcwd(),
        os.path.join(os.path.expanduser("~"), ".cache", "vgi-nlp"),
        os.path.dirname(__file__),
    ]
    for d in search_dirs:
        for fn in _FASTTEXT_FILENAMES:
            candidate = os.path.join(d, fn)
            if os.path.exists(candidate):
                return candidate
    raise ModelNotAvailableError(
        "fastText language-ID model not found. Download it with:\n"
        "  mkdir -p ~/.cache/vgi-nlp && "
        "curl -L -o ~/.cache/vgi-nlp/lid.176.ftz "
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz\n"
        f"or set {_FASTTEXT_ENV} to its path."
    )


@lru_cache(maxsize=1)
def _fasttext() -> fasttext.FastText._FastText:
    """The fastText lid.176 model, loaded once."""
    import fasttext

    # fastText prints a deprecation warning to stderr on load; silence it so it
    # does not pollute the VGI stderr channel.
    fasttext.FastText.eprint = lambda *_a, **_k: None  # type: ignore[attr-defined]
    return fasttext.load_model(_fasttext_model_path())


def _clean_for_fasttext(text: str) -> str:
    """fastText.predict rejects newlines; collapse whitespace to a single line."""
    return " ".join(text.split())


def detect_language(text: str | None) -> tuple[str | None, float]:
    """Return ``(iso_639_code, confidence)`` for ``text``.

    ``(None, 0.0)`` for NULL/empty input. The code is the bare ISO-639 label
    (fastText emits ``__label__en``; we strip the prefix).
    """
    if not text or not text.strip():
        return None, 0.0
    model = _fasttext()
    # Call the underlying C predictor directly rather than model.predict():
    # fasttext-wheel's Python wrapper uses np.array(..., copy=False), which
    # raises under NumPy >= 2. The C method returns [(prob, "__label__xx"), ...].
    predictions = model.f.predict(_clean_for_fasttext(text), 1, 0.0, "strict")
    if not predictions:
        return None, 0.0
    prob, label = predictions[0]
    code = label.replace("__label__", "")
    return code, float(prob)


# ---------------------------------------------------------------------------
# spaCy pipelines (one per language / model name)
# ---------------------------------------------------------------------------


def _resolve_model_name(lang: str | None, model: str | None) -> str:
    """Map a (lang, model) request to a concrete spaCy package name."""
    if model:
        return model
    code = (lang or "en").lower()
    # Accept locale-ish codes like "en-US" / "en_GB".
    code = code.replace("_", "-").split("-")[0]
    name = _DEFAULT_SPACY_MODEL.get(code)
    if name is None:
        raise ModelNotAvailableError(
            f"No default spaCy pipeline for language {code!r}. "
            f"Pass model := '<spacy_model_name>' explicitly, or pin lang to one of: "
            f"{', '.join(sorted(_DEFAULT_SPACY_MODEL))}."
        )
    return name


@cache
def _load_spacy(model_name: str) -> Language:
    """Load (and cache) a spaCy pipeline by package name."""
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise ModelNotAvailableError(
            f"spaCy model {model_name!r} is not installed. Install it with:\n"
            f"  uv run python -m spacy download {model_name}\n"
            f"(original error: {exc})"
        ) from exc


def spacy_pipeline(*, lang: str | None, model: str | None) -> Language:
    """Get the cached spaCy pipeline for an explicit (lang, model) request.

    Thread-safe: spaCy ``load`` is serialised so two concurrent first-uses do not
    race on the same model.
    """
    name = _resolve_model_name(lang, model)
    with _lock:
        return _load_spacy(name)


def pipeline_for_text(text: str, *, lang: str | None, model: str | None) -> Language | None:
    """Pick the spaCy pipeline for one row.

    If ``lang``/``model`` are pinned, always returns that pipeline. Otherwise the
    language is auto-detected per row via fastText and the matching default
    pipeline is returned; ``None`` when the language has no default pipeline (the
    caller should emit no rows / a null result for that row).
    """
    if model or lang:
        return spacy_pipeline(lang=lang, model=model)
    code, _conf = detect_language(text)
    if code is None or code not in _DEFAULT_SPACY_MODEL:
        return None
    with _lock:
        return _load_spacy(_DEFAULT_SPACY_MODEL[code])


def group_by_pipeline(
    texts: list[str | None],
    *,
    lang: str | None,
    model: str | None,
) -> dict[str, list[int]]:
    """Bucket row indices by the spaCy model name that should process them.

    Lets the table functions run ``nlp.pipe()`` once per distinct pipeline instead
    of per row. Rows that are NULL/empty, or whose detected language has no default
    pipeline, are excluded (the caller emits nothing for them).
    """
    buckets: dict[str, list[int]] = {}
    if model or lang:
        name = _resolve_model_name(lang, model)
        for i, t in enumerate(texts):
            if t and t.strip():
                buckets.setdefault(name, []).append(i)
        return buckets
    for i, t in enumerate(texts):
        if not t or not t.strip():
            continue
        code, _conf = detect_language(t)
        if code is None or code not in _DEFAULT_SPACY_MODEL:
            continue
        buckets.setdefault(_DEFAULT_SPACY_MODEL[code], []).append(i)
    return buckets


def load_spacy_by_name(model_name: str) -> Language:
    """Public, thread-safe accessor for a pipeline by its spaCy package name."""
    with _lock:
        return _load_spacy(model_name)


# ---------------------------------------------------------------------------
# VADER sentiment (lexicon, language-agnostic English)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_compound(text: str | None) -> float | None:
    """VADER compound sentiment in [-1, 1]; ``None`` for NULL/empty input."""
    if not text or not text.strip():
        return None
    return float(_vader().polarity_scores(text)["compound"])


def sentiment_label_from_score(score: float | None) -> str | None:
    """Map a [-1, 1] compound score to neg / neu / pos using VADER's thresholds."""
    if score is None:
        return None
    if score >= 0.05:
        return "pos"
    if score <= -0.05:
        return "neg"
    return "neu"


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Load the default models once, eagerly, at worker startup.

    Everything in this module is lazy by design, so the *first* query of every
    ATTACH otherwise pays the spaCy/fastText load cost (~1-2 s) inline. Under the
    end-to-end SQL suite that load happens while the test runner is mid-assertion
    on the first file -- a long window in which a worker-pool teardown SIGTERM (or
    a heavily-loaded host) can kill the run and record a spurious failure, making
    the suite flaky even though every output is deterministic.

    Warming here moves that one-time cost to process-spawn (before the runner
    issues any query), so each per-file first query is fast and the vulnerable
    window shrinks to near zero. It only populates the existing caches -- it never
    changes any output. Best-effort: a missing model is not fatal (the relevant
    function will raise its own actionable error if actually invoked), so a worker
    that hosts, say, only the pure-Python ``normalize`` still starts cleanly.
    """
    with contextlib.suppress(Exception):
        _load_spacy(_DEFAULT_SPACY_MODEL["en"])
    with contextlib.suppress(Exception):
        _fasttext()


# ---------------------------------------------------------------------------
# Batch settings
# ---------------------------------------------------------------------------


def batch_size() -> int:
    """spaCy ``nlp.pipe()`` minibatch size (tunable via VGI_NLP_BATCH_SIZE)."""
    try:
        return max(1, int(os.environ.get("VGI_NLP_BATCH_SIZE", "256")))
    except ValueError:
        return 256
