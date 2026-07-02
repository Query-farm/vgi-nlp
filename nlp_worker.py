# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.9.0",
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
from vgi.catalog import Catalog, Schema

from vgi_nlp import pipelines
from vgi_nlp.scalars import SCALAR_FUNCTIONS
from vgi_nlp.tables import TABLE_FUNCTIONS

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
            "reference_sql": (
                "SELECT nlp.detect_lang('Bonjour tout le monde, comment allez-vous aujourd''hui?') AS lang"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "language-code-and-confidence",
            "prompt": (
                "For the text 'The quick brown fox jumps over the lazy dog', return both its "
                "detected ISO-639 language code and the detector's confidence score, as two "
                "columns named lang and conf."
            ),
            "reference_sql": (
                "SELECT nlp.detect_lang('The quick brown fox jumps over the lazy dog') AS lang, "
                "nlp.detect_lang_conf('The quick brown fox jumps over the lazy dog') AS conf"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "sentiment-score",
            "prompt": (
                "Compute the sentiment polarity score (a number in [-1, 1]) of the sentence "
                "'I absolutely love this product, it works wonderfully!' and return it in a "
                "column named score."
            ),
            "reference_sql": ("SELECT nlp.sentiment('I absolutely love this product, it works wonderfully!') AS score"),
            "ignore_column_names": True,
        },
        {
            "name": "sentiment-label",
            "prompt": (
                "Classify the sentiment of the sentence "
                "'This is the worst experience I have ever had' into a coarse label of "
                "negative, neutral, or positive, and return the label in a column named mood."
            ),
            "reference_sql": ("SELECT nlp.sentiment_label('This is the worst experience I have ever had') AS mood"),
            "ignore_column_names": True,
        },
        {
            "name": "normalize-text",
            "prompt": (
                "Produce a canonical, comparable form of the string '  Cafe   DELUXE  ' by "
                "applying Unicode normalization, lowercasing, and collapsing runs of whitespace. "
                "Return it in a column named canonical."
            ),
            "reference_sql": "SELECT nlp.normalize('  Cafe   DELUXE  ') AS canonical",
            "ignore_column_names": True,
        },
        {
            "name": "lemmatize-english",
            "prompt": (
                "Lemmatize the English sentence 'The cats were running quickly' by reducing each "
                "word to its dictionary base form, and return the space-joined lemmas in a column "
                "named lemmas. Treat the text as English."
            ),
            "reference_sql": "SELECT nlp.lemmatize('The cats were running quickly', 'en') AS lemmas",
            "ignore_column_names": True,
        },
        {
            "name": "strip-stopwords",
            "prompt": (
                "Remove English stop-words and punctuation from the sentence "
                "'this is a really great movie' and return the remaining content words, "
                "space-joined, in a column named kept. Treat the text as English."
            ),
            "reference_sql": "SELECT nlp.strip_stopwords('this is a really great movie', 'en') AS kept",
            "ignore_column_names": True,
        },
        {
            "name": "count-named-entities",
            "prompt": (
                "Count how many named entities are found in the sentence "
                "'Apple was founded by Steve Jobs in California.' Treat the text as English. "
                "Return the count in a column named n."
            ),
            "reference_sql": (
                "SELECT count(*) AS n FROM nlp.entities("
                "(SELECT 1 AS id, 'Apple was founded by Steve Jobs in California.' AS body), "
                "id := 'id', lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "tokens-with-pos",
            "prompt": (
                "For the English sentence 'The quick brown fox jumps.', list each token together "
                "with its part-of-speech tag. Return columns token and pos, one row per token. "
                "Treat the text as English."
            ),
            "reference_sql": (
                "SELECT token, pos FROM nlp.tokens("
                "(SELECT 1 AS id, 'The quick brown fox jumps.' AS body), id := 'id', lang := 'en')"
            ),
            "unordered": True,
            "ignore_column_names": True,
        },
        {
            "name": "count-sentences",
            "prompt": (
                "Count how many sentences are in the text "
                "'First sentence here. Second one follows.' Treat the text as English. Return the "
                "count in a column named n."
            ),
            "reference_sql": (
                "SELECT count(*) AS n FROM nlp.sentences("
                "(SELECT 1 AS id, 'First sentence here. Second one follows.' AS body), "
                "id := 'id', lang := 'en')"
            ),
            "ignore_column_names": True,
        },
        {
            "name": "extract-noun-chunks",
            "prompt": (
                "Extract the base noun phrases (noun chunks) from the English sentence "
                "'The big red car drove down the long road.' Return them in a column named chunk, "
                "one row per noun chunk. Treat the text as English."
            ),
            "reference_sql": (
                "SELECT chunk FROM nlp.noun_chunks("
                "(SELECT 1 AS id, 'The big red car drove down the long road.' AS body), "
                "id := 'id', lang := 'en')"
            ),
            "unordered": True,
            "ignore_column_names": True,
        },
    ]
)


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
