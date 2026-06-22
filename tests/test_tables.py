"""Tests for the table-in-out NLP functions (1 row -> N rows, id passthrough)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.harness import fasttext_available, run_table_function, spacy_model_available
from vgi_nlp.pipelines import ModelNotAvailableError
from vgi_nlp.tables import Entities, NounChunks, Sentences, Tokens

needs_spacy = pytest.mark.skipif(not spacy_model_available(), reason="en_core_web_sm not installed")
needs_fasttext = pytest.mark.skipif(not fasttext_available(), reason="fastText lid.176 model not installed")

BATCH = pa.record_batch(
    {
        "id": [1, 2],
        "body": [
            "Apple Inc. was founded by Steve Jobs in California. He loved it.",
            "Google and Microsoft are based in the United States.",
        ],
    }
)


@needs_spacy
class TestEntities:
    def test_schema(self) -> None:
        tbl = run_table_function(Entities, BATCH, named={"id": "id", "lang": "en"})
        assert tbl.column_names == ["id", "ent_text", "label", "start_char", "end_char"]

    def test_finds_org_person_gpe(self) -> None:
        tbl = run_table_function(Entities, BATCH, named={"id": "id", "lang": "en"})
        labels = set(tbl.column("label").to_pylist())
        assert {"ORG", "PERSON", "GPE"} <= labels

    def test_id_passthrough(self) -> None:
        tbl = run_table_function(Entities, BATCH, named={"id": "id", "lang": "en"})
        d = tbl.to_pydict()
        # Steve Jobs belongs to row id=1; United States to row id=2.
        for ent, id_val in zip(d["ent_text"], d["id"], strict=True):
            if ent == "Steve Jobs":
                assert id_val == 1
            if ent == "the United States":
                assert id_val == 2

    def test_char_offsets_are_consistent(self) -> None:
        tbl = run_table_function(Entities, BATCH, named={"id": "id", "lang": "en"})
        d = tbl.to_pydict()
        for s, e in zip(d["start_char"], d["end_char"], strict=True):
            assert 0 <= s < e


@needs_spacy
class TestSentences:
    def test_segmentation_and_index(self) -> None:
        tbl = run_table_function(Sentences, BATCH, named={"id": "id", "lang": "en"})
        d = tbl.to_pydict()
        assert tbl.column_names == ["id", "sent_index", "sentence"]
        # Row 1 has two sentences (indices 0, 1); row 2 has one (index 0).
        row1 = [si for i, si in zip(d["id"], d["sent_index"], strict=True) if i == 1]
        assert row1 == [0, 1]
        row2 = [si for i, si in zip(d["id"], d["sent_index"], strict=True) if i == 2]
        assert row2 == [0]


@needs_spacy
class TestTokens:
    def test_columns_and_nonempty(self) -> None:
        tbl = run_table_function(Tokens, BATCH, named={"id": "id", "lang": "en"})
        assert tbl.column_names == ["id", "token", "lemma", "pos", "tag", "is_stop", "dep"]
        assert tbl.num_rows > 0
        assert set(tbl.column("is_stop").to_pylist()) <= {True, False}


@needs_spacy
class TestNounChunks:
    def test_chunks_and_roots(self) -> None:
        tbl = run_table_function(NounChunks, BATCH, named={"id": "id", "lang": "en"})
        assert tbl.column_names == ["id", "chunk", "root"]
        chunks = tbl.column("chunk").to_pylist()
        assert "Apple Inc." in chunks


@needs_spacy
class TestConventions:
    def test_no_id_means_no_id_column(self) -> None:
        # With no id, the sole text column is used and no id is carried through.
        batch = pa.record_batch({"body": ["Apple Inc. hired Steve Jobs."]})
        tbl = run_table_function(Entities, batch, named={"lang": "en"})
        assert "id" not in tbl.column_names
        assert tbl.column_names[0] == "ent_text"
        assert "Apple Inc." in tbl.column("ent_text").to_pylist()

    def test_explicit_text_column(self) -> None:
        batch = pa.record_batch({"id": [1], "headline": ["Microsoft hired Elon Musk."]})
        tbl = run_table_function(Entities, batch, named={"id": "id", "text": "headline", "lang": "en"})
        assert "Microsoft" in tbl.column("ent_text").to_pylist()

    def test_default_text_column_is_first_non_id(self) -> None:
        batch = pa.record_batch({"id": [1], "content": ["Amazon is huge."]})
        tbl = run_table_function(Entities, batch, named={"id": "id", "lang": "en"})
        assert "Amazon" in tbl.column("ent_text").to_pylist()

    def test_missing_id_column_raises(self) -> None:
        with pytest.raises(ValueError, match="id column"):
            run_table_function(Entities, BATCH, named={"id": "nope", "lang": "en"})

    def test_missing_text_column_raises(self) -> None:
        with pytest.raises(ValueError, match="text column"):
            run_table_function(Entities, BATCH, named={"id": "id", "text": "nope", "lang": "en"})

    def test_non_string_text_column_raises(self) -> None:
        # The text column must be VARCHAR; an int column is rejected at bind time.
        batch = pa.record_batch({"id": [1], "body": [42]})
        with pytest.raises(ValueError, match="must be VARCHAR"):
            run_table_function(Entities, batch, named={"id": "id", "text": "body", "lang": "en"})


@needs_spacy
class TestErrorAndEdgeCases:
    def test_empty_and_null_rows_emit_nothing(self) -> None:
        # NULL / empty / whitespace-only text rows simply emit zero output rows
        # (no error), and their ids never appear in the output.
        batch = pa.record_batch({"id": [1, 2, 3], "body": [None, "", "   "]})
        tbl = run_table_function(Entities, batch, named={"id": "id", "lang": "en"})
        assert tbl.num_rows == 0
        assert tbl.column_names == ["id", "ent_text", "label", "start_char", "end_char"]

    def test_text_with_no_entities_is_empty_not_error(self) -> None:
        # A sentence with no named entities yields an empty result, not an error.
        batch = pa.record_batch({"id": [1], "body": ["the small cat sat quietly on a soft mat"]})
        tbl = run_table_function(Entities, batch, named={"id": "id", "lang": "en"})
        assert tbl.num_rows == 0

    def test_tokens_still_produced_when_no_entities(self) -> None:
        # The same plain sentence still tokenizes (tokens are not entity-gated).
        batch = pa.record_batch({"id": [1], "body": ["the small cat sat quietly"]})
        tbl = run_table_function(Tokens, batch, named={"id": "id", "lang": "en"})
        assert tbl.num_rows > 0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ModelNotAvailableError, match="not installed"):
            run_table_function(Entities, BATCH, named={"id": "id", "model": "no_such_model_xyz"})

    @needs_fasttext
    def test_unknown_pinned_language_raises(self) -> None:
        with pytest.raises(ModelNotAvailableError, match="No default spaCy pipeline"):
            run_table_function(Entities, BATCH, named={"id": "id", "lang": "xx"})

    def test_long_text_does_not_crash(self) -> None:
        big = "Apple Inc. hired Steve Jobs in California. " * 500
        batch = pa.record_batch({"id": [1], "body": [big]})
        tbl = run_table_function(Entities, batch, named={"id": "id", "lang": "en"})
        assert tbl.num_rows > 0
        assert "Apple Inc." in tbl.column("ent_text").to_pylist()
