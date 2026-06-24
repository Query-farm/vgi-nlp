# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
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
    "# nlp\n\n"
    "Classical NLP (spaCy + fastText language-ID + VADER sentiment) exposed to "
    "DuckDB/SQL as a VGI worker -- bulk, cheap, per-row text enrichment.\n\n"
    "**Scalars:** `detect_lang`, `detect_lang_conf`, `sentiment`, `sentiment_label`, "
    "`lemmatize`, `strip_stopwords`, `normalize`.\n\n"
    "**Table functions:** `entities`, `tokens`, `sentences`, `noun_chunks` "
    "(one text row in, N rows out, with an optional `id :=` passthrough)."
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

_NLP_CATALOG = Catalog(
    name="nlp",
    default_schema="main",
    comment="Classical NLP (spaCy + fastText + VADER): language ID, sentiment, NER, tokenization for SQL.",
    source_url="https://github.com/Query-farm/vgi-nlp",
    tags={
        "vgi.title": "Classical NLP for SQL",
        "vgi.keywords": (
            "nlp, natural language processing, language detection, sentiment analysis, "
            "named entity recognition, ner, tokenization, lemmatize, stop words, "
            "noun chunks, spacy, fasttext, vader, text enrichment"
        ),
        "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
        "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
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
                "vgi.keywords": (
                    "nlp, language detection, sentiment, ner, entities, tokens, "
                    "sentences, noun chunks, lemmatize, strip stopwords, normalize, "
                    "spacy, fasttext, vader"
                ),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "text-analytics",
                "category": "natural-language-processing",
                "topic": "language-detection-sentiment-ner",
                "vgi.source_url": "https://github.com/Query-farm/vgi-nlp/blob/main/nlp_worker.py",
                "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
                "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
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
