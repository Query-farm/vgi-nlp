# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.14.0",
#     "spacy>=3.7",
#     "fasttext-wheel",
#     "vaderSentiment",
#     "en-core-web-sm",
# ]
#
# [tool.uv.sources]
# en-core-web-sm = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" }
# ///
"""VGI worker exposing classical NLP (spaCy + fastText + VADER) to DuckDB/SQL.

Assembles the scalar and table-in-out functions in ``vgi_nlp`` into a single
``nlp`` catalog and runs the worker over stdio (local) or HTTP (via serve.py).

Usage:
    uv run nlp_worker.py                 # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000          # serve over HTTP

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'nlp' (TYPE vgi, LOCATION 'uv run nlp_worker.py');

    SELECT nlp.detect_lang(review) AS lang, count(*) FROM reviews GROUP BY 1;
    SELECT * FROM nlp.entities((SELECT id, body FROM articles), id := 'id');
    SELECT id, nlp.sentiment(body) AS score FROM reviews;

First-use model requirements:
    uv run python -m spacy download en_core_web_sm
    curl -L -o ~/.cache/vgi-nlp/lid.176.ftz \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
"""

from __future__ import annotations

import json
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema, Table

from vgi_nlp import pipelines
from vgi_nlp.scalars import SCALAR_FUNCTIONS
from vgi_nlp.tables import TABLE_FUNCTIONS, SupportedLanguages

_CATALOG_DESCRIPTION_LLM = (
    "Classical, non-LLM natural-language processing over text columns: detect each "
    "row's language and confidence (fastText lid.176), score sentiment in [-1, 1] and "
    "label it neg/neu/pos (VADER), lemmatize, strip stop-words, and Unicode-normalize "
    "text (spaCy), plus table-valued functions that explode a text column into named "
    "entities, tokens with POS tags, sentences, and noun chunks. Use it for bulk, cheap, "
    "per-row text enrichment in SQL -- upstream of, not a wrapper around, LLM workers."
)

_CATALOG_DESCRIPTION_MD = (
    "# Classical NLP for DuckDB SQL\n\n"
    "![spaCy logo](https://upload.wikimedia.org/wikipedia/commons/8/88/SpaCy_logo.svg)\n\n"
    "Run **classical natural language processing in SQL** -- language detection, "
    "sentiment analysis, named entity recognition (NER), lemmatization, and "
    "tokenization -- directly over your text columns in DuckDB, with no LLM and no "
    "API key required. This worker turns the proven open-source NLP stack of "
    "[spaCy](https://spacy.io), [fastText](https://fasttext.cc/), and "
    "[VADER](https://github.com/cjhutto/vaderSentiment) into fast, deterministic "
    "SQL functions for bulk, cheap, per-row text enrichment.\n\n"
    "## Who it is for\n\n"
    "Data engineers and analysts who want to enrich large text tables -- product "
    "reviews, support tickets, articles, social posts -- without leaving SQL or "
    "paying per token. Because the underlying models are rule- and "
    "statistics-based rather than generative, output is reproducible and runs at "
    "column scale -- ideal *upstream* of, not as a replacement for, LLM workers.\n\n"
    "## Key concepts\n\n"
    "- **Models load once per worker process and are cached**, so the one-time "
    "model-load cost is amortized across every row of every query.\n"
    "- **Language identification** uses Facebook's "
    "[fastText lid.176](https://fasttext.cc/docs/en/language-identification.html) "
    "model (176 languages). When you do not pin a language, each row is routed to "
    "the matching language pipeline automatically.\n"
    "- **Sentiment** uses the lexicon-and-rule-based VADER scorer, tuned for "
    "English and social media text; it understands negation, intensifiers, and "
    "emoji.\n"
    "- **Linguistic analysis** -- structured span extraction, part-of-speech "
    "tagging, sentence boundaries, noun-phrase mining, lemmatization, and Unicode "
    "normalization -- uses spaCy's small English pipeline "
    "(`en_core_web_sm`) by default; pin a language or name a specific model to "
    "override it.\n\n"
    "## When to reach for it\n\n"
    "Use it whenever you need to enrich or filter free text at scale in SQL: "
    "detect and segment multilingual corpora, score or bucket opinion, clean and "
    "canonicalize text so free-text keys join and group reliably, or split "
    "documents into structured rows you can aggregate and join back to their "
    "source. Attach the worker, then list the schema to discover the available "
    "functions and their signatures:\n\n"
    "```sql\n"
    "INSTALL vgi FROM community;\n"
    "LOAD vgi;\n"
    "ATTACH 'nlp' (TYPE vgi, LOCATION 'uv run nlp_worker.py');\n"
    "```\n\n"
    "Powered by spaCy ([source](https://github.com/explosion/spaCy) - "
    "[docs](https://spacy.io/usage)), fastText "
    "([source](https://github.com/facebookresearch/fastText) - "
    "[docs](https://fasttext.cc/docs/en/language-identification.html)), and VADER "
    "([source & docs](https://github.com/cjhutto/vaderSentiment)). "
    "Source for this worker: "
    "[Query-farm/vgi-nlp](https://github.com/Query-farm/vgi-nlp)."
)

_SCHEMA_DESCRIPTION_LLM = (
    "Classical NLP functions over text columns: language identification, sentiment "
    "scoring/labelling, lemmatization, stop-word stripping, Unicode normalization, and "
    "table-valued entity/token/sentence/noun-chunk extraction."
)

_SCHEMA_DESCRIPTION_MD = (
    "# main\n\n"
    "Classical NLP functions exposed to SQL.\n\n"
    "- **Scalars:** language ID/confidence, sentiment score/label, lemmatize, "
    "strip stop-words, normalize.\n"
    "- **Table functions:** entities, tokens, sentences, noun_chunks "
    "(one text row in, N rows out, with an optional `id :=` passthrough)."
)

# Representative, catalog-qualified example queries for the schema (VGI506).
# All are self-contained so they bind/execute against the worker.
_SCHEMA_EXAMPLE_QUERIES = (
    "SELECT nlp.main.detect_lang('Bonjour tout le monde');\n"
    "SELECT nlp.main.sentiment('I absolutely love this product!');\n"
    "SELECT nlp.main.sentiment_label('This was a terrible experience');\n"
    "SELECT nlp.main.lemmatize('The cats were running', 'en');\n"
    "SELECT nlp.main.normalize('  Café   DELUXE  ');\n"
    "SELECT * FROM nlp.main.entities("
    "(SELECT 1 AS id, 'Apple is based in California.' AS body), id := 'id', lang := 'en');"
)

# Ordered navigation registry for the `main` schema (VGI408-413). Every function
# carries a `vgi.category` naming exactly one of these; the order here is the
# display order for listings/SEO. Keep names in sync with the per-function
# `category=` passed to `object_tags(...)` in scalars.py / tables.py.
_SCHEMA_CATEGORIES = json.dumps(
    [
        {
            "name": "language-id",
            "title": "Language identification",
            "description": "Detect the natural language of each text and how confident the model is.",
        },
        {
            "name": "sentiment",
            "title": "Sentiment analysis",
            "description": "Score and label the emotional polarity of text (VADER).",
        },
        {
            "name": "text-cleaning",
            "title": "Text cleaning & normalization",
            "description": (
                "Lemmatize, strip stop-words, and Unicode-normalize text so it is ready for "
                "matching, dedup, search, and featurization."
            ),
        },
        {
            "name": "extraction",
            "title": "Structured extraction",
            "description": (
                "Explode one text row into many structured rows: named entities, annotated "
                "tokens, sentences, and noun chunks."
            ),
        },
        {
            "name": "discovery",
            "title": "Capability discovery",
            "description": "Browse what the worker supports -- e.g. the languages that have a spaCy pipeline.",
        },
    ]
)


# Fixed analyst-suitability suite (VGI152 / VGI920). `vgi-lint simulate` drives an
# LLM analyst through each task -- it sees only the `prompt` plus the live catalog
# listing, never the `reference_sql`. Grading runs `reference_sql` against THIS
# worker and compares result sets, so model-dependent outputs (NER counts, lemmas)
# are still deterministic: both sides use the same pinned `en_core_web_sm`. Tasks
# use `ignore_column_names` (values are what matter) and `unordered` for the
# table functions, whose row order is intentionally not guaranteed.
_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "detect-language-code",
            "prompt": (
                "Identify the language of the text "
                "'Bonjour tout le monde, comment allez-vous aujourd''hui?' and return its "
                "two-letter ISO-639 language code in a column named lang."
            ),
            # Deterministic string output (fastText lid.176 is deterministic).
            "reference_sql": (
                "SELECT nlp.main.detect_lang('Bonjour tout le monde, comment allez-vous aujourd''hui?') AS lang"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "confident-english",
            "prompt": (
                "Decide whether the language detector is highly confident (more than 90%) that "
                "the text 'The quick brown fox jumps over the lazy dog' is English. Return a "
                "single boolean column named confident_english that is true only when the "
                "detected ISO-639 code is 'en' and the detection confidence exceeds 0.9."
            ),
            # Float confidence reframed as a stable boolean threshold predicate (VGI920).
            "reference_sql": (
                "SELECT (nlp.main.detect_lang('The quick brown fox jumps over the lazy dog') = 'en' "
                "AND nlp.main.detect_lang_conf('The quick brown fox jumps over the lazy dog') > 0.9) "
                "AS confident_english"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "sentiment-is-positive",
            "prompt": (
                "Decide whether the sentence 'I absolutely love this product, it works "
                "wonderfully!' expresses positive sentiment. Using the compound sentiment score, "
                "return a single boolean column named is_positive that is true when the score is "
                "above the standard positive threshold of 0.05."
            ),
            # Float score reframed as a stable boolean threshold predicate (VGI920).
            "reference_sql": (
                "SELECT nlp.main.sentiment('I absolutely love this product, it works wonderfully!') > 0.05 "
                "AS is_positive"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "sentiment-label",
            "prompt": (
                "Classify the sentiment of the sentence "
                "'This is the worst experience I have ever had' into a coarse label of "
                "negative, neutral, or positive, and return the label in a column named mood."
            ),
            # Deterministic coarse label (neg/neu/pos) from fixed VADER thresholds.
            "reference_sql": (
                "SELECT nlp.main.sentiment_label('This is the worst experience I have ever had') AS mood"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "normalize-text",
            "prompt": (
                "Produce a canonical, comparable form of the string '  Cafe   DELUXE  ' by "
                "applying Unicode normalization, lowercasing, and collapsing runs of whitespace. "
                "Return it in a column named canonical."
            ),
            # Pure-Python, fully deterministic string transform.
            "reference_sql": "SELECT nlp.main.normalize('  Cafe   DELUXE  ') AS canonical",
            "ignore_column_names": True,
        },
        {
            "name": "lemmatize-contains-base-form",
            "prompt": (
                "Lemmatize the English sentence 'The cats were running quickly', reducing each "
                "word to its dictionary base form. Decide whether the lemmatized result contains "
                "the base form 'run'. Return a single boolean column named has_run. Treat the "
                "text as English."
            ),
            # Model-dependent lemma string reframed as a stable containment predicate (VGI920).
            "reference_sql": (
                "SELECT nlp.main.lemmatize('The cats were running quickly', 'en') LIKE '%run%' AS has_run"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "strip-stopwords-keeps-content",
            "prompt": (
                "Remove English stop-words and punctuation from the sentence "
                "'this is a really great movie'. Decide whether the content word 'great' survives "
                "the stop-word removal. Return a single boolean column named keeps_great. Treat "
                "the text as English."
            ),
            # Model-dependent token string reframed as a stable containment predicate (VGI920).
            "reference_sql": (
                "SELECT nlp.main.strip_stopwords('this is a really great movie', 'en') LIKE '%great%' AS keeps_great"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "has-named-entities",
            "prompt": (
                "Decide whether the sentence 'Apple was founded by Steve Jobs in California.' "
                "contains at least two named entities. Treat the text as English. Return a single "
                "boolean column named has_entities."
            ),
            # Model-dependent entity count reframed as a stable threshold predicate (VGI920).
            "reference_sql": (
                "SELECT count(*) >= 2 AS has_entities FROM nlp.main.entities("
                "(SELECT 'Apple was founded by Steve Jobs in California.' AS body), lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "tokenizes-into-words",
            "prompt": (
                "Tokenize the English sentence 'The quick brown fox jumps.' into individual "
                "tokens with part-of-speech tags. Decide whether it yields more than three "
                "tokens. Treat the text as English. Return a single boolean column named "
                "many_tokens."
            ),
            # Model-dependent tokenization reframed as a stable threshold predicate (VGI920).
            "reference_sql": (
                "SELECT count(*) > 3 AS many_tokens FROM nlp.main.tokens("
                "(SELECT 'The quick brown fox jumps.' AS body), lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "counts-two-sentences",
            "prompt": (
                "Split the text 'First sentence here. Second one follows.' into sentences and "
                "decide whether it contains exactly two sentences. Treat the text as English. "
                "Return a single boolean column named two_sentences."
            ),
            # Sentence segmentation is stable on clear boundaries; compared as a boolean.
            "reference_sql": (
                "SELECT count(*) = 2 AS two_sentences FROM nlp.main.sentences("
                "(SELECT 'First sentence here. Second one follows.' AS body), lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "has-noun-chunks",
            "prompt": (
                "Extract the base noun phrases (noun chunks) from the English sentence "
                "'The big red car drove down the long road.' and decide whether it yields at "
                "least two noun chunks. Treat the text as English. Return a single boolean column "
                "named has_chunks."
            ),
            # Model-dependent noun-chunk count reframed as a stable threshold predicate (VGI920).
            "reference_sql": (
                "SELECT count(*) >= 2 AS has_chunks FROM nlp.main.noun_chunks("
                "(SELECT 'The big red car drove down the long road.' AS body), lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "english-pipeline-supported",
            "prompt": (
                "Determine whether this worker has a default spaCy pipeline for English (ISO-639 "
                "code 'en'), i.e. whether English text can be lemmatized and run through NER "
                "without naming a custom model. Return a single boolean column named supported."
            ),
            # Browses the supported_languages discovery table; stable boolean.
            "reference_sql": (
                "SELECT count(*) > 0 AS supported FROM nlp.main.supported_languages() WHERE lang_code = 'en'"
            ),
            "ignore_column_names": True,
        },
    ]
)


# VGI146/VGI311: `supported_languages` takes no arguments and always returns the
# same rows, so it is also exposed as a regular, browsable table (function-backed
# -- DuckDB scans the `SupportedLanguages` table function) that an agent can read
# with `SELECT * FROM nlp.main.supported_languages` -- no parentheses, no arguments
# to guess -- in addition to calling `supported_languages()`. The table's schema is
# derived from the function's bind(), so the two stay in lockstep. `lang_code` is a
# NOT NULL natural primary key (VGI806/VGI807).
_SUPPORTED_LANGUAGES_DOC_LLM = (
    "## `supported_languages` (table)\n\n"
    "One row per language that has a default spaCy pipeline, so the extraction table functions "
    "and the `lemmatize` / `strip_stopwords` scalars can process it without naming a model. "
    "Columns:\n\n"
    "- `lang_code` (`VARCHAR`, primary key) -- the ISO-639 code you pass as the `lang` argument "
    "(or that per-row auto-detect must resolve to).\n"
    "- `spacy_model` (`VARCHAR`) -- the default small spaCy pipeline that backs it.\n\n"
    "Read this table directly to discover which languages can be lemmatized, tokenized, or run "
    "through NER. Note `detect_lang` recognizes far more languages (fastText covers 176) than have "
    "a spaCy pipeline installed -- this table lists only the latter. Backed by the "
    "identically-named table function, so the rows are identical."
)

_SUPPORTED_LANGUAGES_DOC_MD = (
    "# `supported_languages`\n\n"
    "Discovery table of every language the worker has a default spaCy pipeline for, exposed as a "
    "regular table you can read without parentheses.\n\n"
    "## Columns\n\n"
    "- `lang_code` (VARCHAR, primary key) -- ISO-639 code accepted by the `lang` argument.\n"
    "- `spacy_model` (VARCHAR) -- the default small spaCy model backing it.\n\n"
    "Language detection (`detect_lang`) spans 176 languages via fastText, but only the languages "
    "listed here can be lemmatized, tokenized, or run through NER without naming a custom model. "
    "See the table's example queries for ready-to-run SQL."
)

_DISCOVERY_TABLES: list[Table] = [
    Table(
        name="supported_languages",
        function=SupportedLanguages,
        comment="Languages with a default spaCy pipeline: (lang_code, spacy_model) -- discovery table.",
        primary_key=(("lang_code",),),
        not_null=("lang_code", "spacy_model"),
        column_comments={
            "lang_code": "ISO-639 language code accepted by the `lang` argument.",
            "spacy_model": "Default small spaCy pipeline that backs this language.",
        },
        tags={
            "vgi.title": "Supported Languages Table",
            "vgi.doc_llm": _SUPPORTED_LANGUAGES_DOC_LLM,
            "vgi.doc_md": _SUPPORTED_LANGUAGES_DOC_MD,
            "vgi.keywords": json.dumps(
                [
                    "supported languages",
                    "languages",
                    "iso-639",
                    "spacy models",
                    "language support",
                    "capabilities",
                    "discovery",
                    "which languages",
                ]
            ),
            "vgi.category": "discovery",
            "domain": "text-analytics",
            "vgi.example_queries": json.dumps(
                [
                    {
                        "description": "Count the languages that have a default spaCy pipeline.",
                        "sql": "SELECT count(*) AS n_languages FROM nlp.main.supported_languages",
                    },
                    {
                        "description": "Look up the default spaCy model for English.",
                        "sql": "SELECT spacy_model FROM nlp.main.supported_languages WHERE lang_code = 'en'",
                    },
                ]
            ),
        },
    ),
]


_NLP_CATALOG = Catalog(
    name="nlp",
    default_schema="main",
    comment="Classical NLP (spaCy + fastText + VADER): language ID, sentiment, NER, tokenization for SQL.",
    source_url="https://github.com/Query-farm/vgi-nlp",
    tags={
        "vgi.title": "Classical NLP for SQL",
        "vgi.keywords": json.dumps(
            [
                "nlp",
                "natural language processing",
                "language detection",
                "sentiment analysis",
                "named entity recognition",
                "ner",
                "tokenization",
                "lemmatize",
                "stop words",
                "noun chunks",
                "spacy",
                "fasttext",
                "vader",
                "text enrichment",
            ]
        ),
        "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
        "vgi.agent_test_tasks": _AGENT_TEST_TASKS,
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "LicenseRef-QueryFarm-Source-Available-1.0",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-nlp/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-nlp/blob/main/README.md",
    },
    schemas=[
        Schema(
            name="main",
            comment="Classical NLP: language ID, sentiment, NER, tokenization for SQL",
            tags={
                "vgi.title": "NLP Functions (main)",
                "vgi.keywords": json.dumps(
                    [
                        "nlp",
                        "language detection",
                        "sentiment",
                        "ner",
                        "entities",
                        "tokens",
                        "sentences",
                        "noun chunks",
                        "lemmatize",
                        "strip stopwords",
                        "normalize",
                        "spacy",
                        "fasttext",
                        "vader",
                    ]
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "text-analytics",
                "category": "natural-language-processing",
                "topic": "language-detection-sentiment-ner",
                # VGI139: source_url lives only on the catalog object, not here.
                "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
                "vgi.categories": _SCHEMA_CATEGORIES,
            },
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
            tables=list(_DISCOVERY_TABLES),
        ),
    ],
)


class NlpWorker(Worker):
    """Worker process hosting the classical-NLP catalog."""

    catalog = _NLP_CATALOG

    def run(self, otel_config: Any = None) -> None:
        """Warm the default models, then serve.

        Loading spaCy/fastText is lazy, so without this the first query of every
        ATTACH pays the ~1-2 s model-load cost inline -- a window in which a
        worker-pool teardown SIGTERM (or a heavily-loaded host) can kill the run
        mid-assertion and record a spurious E2E failure. Warming at spawn moves
        that one-time cost ahead of any query, keeping the SQL suite deterministic
        without changing a single output value. Best-effort; never fatal.
        """
        pipelines.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the NLP worker process (stdio or, via flags, HTTP)."""
    NlpWorker.main()


if __name__ == "__main__":
    main()
