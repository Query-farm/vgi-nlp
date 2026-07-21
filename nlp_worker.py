# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.16.0",
#     "spacy>=3.7",
#     "fasttext-wheel",
#     "vaderSentiment",
#     "en-core-web-sm",
# ]
#
# [tool.uv.sources]
# en-core-web-sm = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" }
# ///
"""Repo-root stdio entry for the VGI classical-NLP worker (PEP 723 shim).

The worker's catalog, ``NlpWorker`` class, and ``main()`` now live in the
wheel-importable ``vgi_nlp.worker`` module; this file is a thin PEP 723 shim
that re-exports them so ``uv run nlp_worker.py`` keeps working unchanged
(Makefile, ci/run-integration.sh, tests). The inline dependency block pins the
spaCy ``en_core_web_sm`` model wheel so the launched worker's own environment
can load it (see CLAUDE.md sharp-edge #1).

Usage:
    uv run nlp_worker.py                 # serve over stdio (DuckDB subprocess)
    uv run nlp_worker.py --http --port 8000   # serve over HTTP
    python serve.py --port 8000          # serve over HTTP (shim)

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

from vgi_nlp.worker import NlpWorker, main

__all__ = ["NlpWorker", "main"]


if __name__ == "__main__":
    main()
