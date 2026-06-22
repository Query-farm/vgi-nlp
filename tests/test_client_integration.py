"""End-to-end tests through ``vgi.client.Client``, spawning the real worker.

These exercise the full Arrow-IPC round trip the way DuckDB would: the worker
runs as a subprocess and we drive it over stdin/stdout. Skipped when the
required models are unavailable, so a bare checkout stays green.
"""

from __future__ import annotations

import os
import shlex
import sys

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

from tests.harness import fasttext_available, spacy_model_available

_WORKER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nlp_worker.py")

needs_fasttext = pytest.mark.skipif(not fasttext_available(), reason="fastText lid.176 model not installed")
needs_spacy = pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_sm not installed")


def _client() -> Client:
    # Launch the worker with the same interpreter running the tests, so it sees
    # the installed models (rather than going through `uv run`). Client wants a
    # shell-style command string.
    return Client(f"{shlex.quote(sys.executable)} {shlex.quote(_WORKER)}")


@needs_fasttext
def test_detect_lang_scalar_end_to_end() -> None:
    batch = pa.RecordBatch.from_pydict(
        {"text": ["I really love this wonderful product.", "Ceci est un texte en français."]}
    )
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="detect_lang",
                input=iter([batch]),
                arguments=Arguments(positional=[pa.scalar("text")]),
            )
        )
    assert results[0]["result"].to_pylist() == ["en", "fr"]


def test_sentiment_scalar_end_to_end() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["I love this!", "This is terrible and awful."]})
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="sentiment",
                input=iter([batch]),
                arguments=Arguments(positional=[pa.scalar("text")]),
            )
        )
    scores = results[0]["result"].to_pylist()
    assert scores[0] > 0.3
    assert scores[1] < -0.3


@needs_spacy
def test_entities_table_in_out_end_to_end() -> None:
    batch = pa.RecordBatch.from_pydict(
        {"id": [1, 2], "body": ["Apple Inc. hired Steve Jobs.", "Google is in California."]}
    )
    with _client() as client:
        out = list(
            client.table_in_out_function(
                function_name="entities",
                input=iter([batch]),
                arguments=Arguments(named={"id": pa.scalar("id"), "lang": pa.scalar("en")}),
            )
        )
    table = pa.Table.from_batches(out)
    assert table.column_names == ["id", "ent_text", "label", "start_char", "end_char"]
    ents = table.column("ent_text").to_pylist()
    assert "Apple Inc." in ents
    assert "Steve Jobs" in ents
