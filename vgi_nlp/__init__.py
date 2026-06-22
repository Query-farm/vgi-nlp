"""Classical NLP as a VGI worker: language ID, sentiment, NER, and tokenization for DuckDB/SQL.

The implementation is split by concern so each module stays focused:

- ``pipelines`` -- loaded-once-and-cached spaCy / fastText / VADER model lifecycle
- ``scalars``   -- per-row enrichment (detect_lang, sentiment, lemmatize, ...) as scalar functions
- ``tables``    -- 1-row-in / N-rows-out explode (entities, tokens, sentences, noun_chunks)
- ``schema_utils`` -- shared Arrow-field/column-comment helpers

``nlp_worker.py`` at the repo root assembles these into the ``nlp`` catalog and
runs the worker over stdio (DuckDB subprocess) or, via ``serve.py``, HTTP.
"""

from __future__ import annotations

__version__ = "0.1.0"
