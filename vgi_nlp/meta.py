"""Shared helpers for the per-object discovery/description metadata that the
``vgi-lint`` strict profile expects on **every** function and table.

Each function/table surfaces these in its ``Meta.tags``:

- ``vgi.title`` (VGI124)        -- human-friendly display name (must differ from
  the machine name; add an extra word so VGI125 stays quiet)
- ``vgi.doc_llm`` (VGI112)      -- Markdown narrative aimed at LLMs/agents
- ``vgi.doc_md`` (VGI113)       -- Markdown narrative for human docs (distinct
  content from ``doc_llm``)
- ``vgi.keywords`` (VGI126)     -- comma-separated search terms/synonyms
- ``vgi.source_url`` (VGI128)   -- link to the implementing source file

``source_url(file)`` builds the canonical GitHub blob URL so every object points
at exactly where it is implemented.
"""  # noqa: D205

from __future__ import annotations

#: Base GitHub blob URL for source files in this repo (pinned to ``main``).
SOURCE_BASE = "https://github.com/Query-farm/vgi-nlp/blob/main"


def source_url(relative_path: str) -> str:
    """Build the implementation ``vgi.source_url`` for a repo-relative file.

    e.g. ``source_url("vgi_nlp/scalars.py")``.
    """
    return f"{SOURCE_BASE}/{relative_path}"


def object_tags(
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery/description tags.

    ``relative_path`` is the implementing file relative to the repo root.
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
