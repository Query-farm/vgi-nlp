"""Shared helpers for the per-object discovery/description metadata that the
``vgi-lint`` strict profile expects on **every** function and table.

Each function/table surfaces these in its ``Meta.tags``:

- ``vgi.title`` (VGI124)        -- human-friendly display name (must differ from
  the machine name; add an extra word so VGI125 stays quiet)
- ``vgi.doc_llm`` (VGI112)      -- Markdown narrative aimed at LLMs/agents
- ``vgi.doc_md`` (VGI113)       -- Markdown narrative for human docs (distinct
  content from ``doc_llm``)
- ``vgi.keywords`` (VGI126/VGI138) -- a JSON array of search-term/synonym strings

Per-object ``vgi.source_url`` is intentionally NOT emitted here: VGI139 requires
``source_url`` to live only on the catalog object (set on ``Catalog(...)`` in
``nlp_worker.py``), not on every function/schema.
"""  # noqa: D205

from __future__ import annotations

import json


def keywords_array(keywords: str) -> str:
    """Serialize comma-separated keywords as a JSON array of strings (VGI138).

    ``keywords`` is a comma-separated list (e.g. ``"ner, entities, spacy"``);
    each term is trimmed and emitted as one element of a JSON string array.
    """
    terms = [k.strip() for k in keywords.split(",") if k.strip()]
    return json.dumps(terms)


def object_tags(
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: str,
    relative_path: str,
    category: str = "",
    example_queries: str = "",
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    ``relative_path`` is the implementing file relative to the repo root; it is
    retained for call-site documentation but no longer emitted as a per-object
    ``vgi.source_url`` (VGI139 keeps ``source_url`` on the catalog only).

    ``category`` is the object's primary ``vgi.category`` -- it must name one of
    the categories declared in the schema's ``vgi.categories`` registry (VGI409);
    every object in a schema that declares categories should carry one (VGI411).

    ``example_queries`` is a pre-serialized JSON array of ``{description, sql}``
    objects. When non-empty it is emitted as ``vgi.example_queries`` (VGI515): the
    native ``duckdb_functions().examples`` carrier drops per-example descriptions,
    so this described carrier restores them.
    """
    tags = {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_array(keywords),
    }
    if category:
        tags["vgi.category"] = category
    if example_queries:
        tags["vgi.example_queries"] = example_queries
    return tags
