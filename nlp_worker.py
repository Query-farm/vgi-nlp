# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "spacy>=3.7",
#     "fasttext-wheel",
#     "vaderSentiment",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
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

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_nlp.scalars import SCALAR_FUNCTIONS
from vgi_nlp.tables import TABLE_FUNCTIONS

_NLP_CATALOG = Catalog(
    name="nlp",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="Classical NLP: language ID, sentiment, NER, tokenization for SQL",
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
        ),
    ],
)


class NlpWorker(Worker):
    """Worker process hosting the classical-NLP catalog."""

    catalog = _NLP_CATALOG


def main() -> None:
    """Run the NLP worker process (stdio or, via flags, HTTP)."""
    NlpWorker.main()


if __name__ == "__main__":
    main()
