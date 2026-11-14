"""Placement and drift checks for the composed example queries (vgi_nlp/composed.py).

The rows themselves are asserted end to end by ``test/sql/composed.test``; these
tests keep that file, the private graders, and the catalog metadata in sync.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from vgi.metadata import FunctionExample

from vgi_nlp.composed import (
    COMPOSED_EXAMPLES,
    LANGUAGE_ROUTING,
    NEGATIVE_REVIEW_ASPECTS,
    SENTIMENT_PER_ORGANIZATION,
    WORST_SENTENCE_PER_REVIEW,
)
from vgi_nlp.scalars import DetectLang
from vgi_nlp.tables import Entities, NounChunks, Sentences
from vgi_nlp.worker import _AGENT_TEST_TASKS, _NLP_CATALOG

_ROOT = Path(__file__).resolve().parent.parent
_SQL_TEST = (_ROOT / "test" / "sql" / "composed.test").read_text(encoding="utf-8")
_GRADERS = (_ROOT / "vgi-agent-tests.yaml").read_text(encoding="utf-8")
_COMPOSED_TASKS = ["most-negative-sentence-index", "count-texts-with-default-pipeline"]


@pytest.mark.parametrize(
    ("driver", "example"),
    [
        (DetectLang, LANGUAGE_ROUTING),
        (Entities, SENTIMENT_PER_ORGANIZATION),
        (Sentences, WORST_SENTENCE_PER_REVIEW),
        (NounChunks, NEGATIVE_REVIEW_ASPECTS),
    ],
)
def test_composed_example_on_driver_within_first_five(driver: type, example: FunctionExample) -> None:
    """The agent simulator only shows an object's first five examples."""
    assert example in driver.Meta.examples[:5]
    described = json.loads(driver.Meta.tags["vgi.example_queries"])
    assert {"description": example.description, "sql": example.sql} in described[:5]


def test_composed_examples_in_schema_list() -> None:
    schema_examples = json.loads(_NLP_CATALOG.schemas[0].tags["vgi.example_queries"])
    for example in COMPOSED_EXAMPLES:
        assert {"description": example.description, "sql": example.sql} in schema_examples
    assert len(_NLP_CATALOG.schemas[0].tags["vgi.example_queries"]) <= 8000  # VGI417


def test_composed_sql_asserted_verbatim_in_sql_suite() -> None:
    for example in COMPOSED_EXAMPLES:
        assert f"\n{example.sql};\n" in _SQL_TEST


@pytest.mark.parametrize("task", _COMPOSED_TASKS)
def test_composed_grader_asserted_and_published(task: str) -> None:
    match = re.search(rf'- name: {task}\n    reference_sql: "(.*)"\n', _GRADERS)
    assert match, f"no grader for {task}"
    assert f"\n{match.group(1)};\n" in _SQL_TEST
    assert task in {t["name"] for t in json.loads(_AGENT_TEST_TASKS)}
    # The answer must not be published verbatim as an example.
    published = json.dumps(_NLP_CATALOG.schemas[0].tags["vgi.example_queries"])
    assert match.group(1) not in published
