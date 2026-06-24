"""Table-in-out NLP functions: one text row in, N rows out.

spaCy turns a document into many entities / tokens / sentences / noun-chunks, so
these map onto VGI's *table-in-out* primitive: the input relation streams in
batch by batch, and each text row explodes into zero-or-more output rows.

    SELECT * FROM nlp.entities((SELECT id, body FROM articles), id := 'id');
    SELECT * FROM nlp.sentences((SELECT id, content FROM docs), id := 'id');

Conventions (same family as ``vgi-sklearn``):

* The input is a ``(SELECT ...)`` subquery passed positionally.
* ``id := 'col'`` names a passthrough column copied onto every emitted row, so you
  can join the entities/tokens back to the source row. Optional; omit it and no id
  is carried.
* ``text := 'col'`` names the text column. Default: the sole non-id column (or the
  first non-id column if several).
* ``lang := 'en'`` pins the spaCy pipeline language; default is per-row fastText
  auto-detect. ``model := '...'`` overrides the spaCy model.

Rows are grouped by pipeline so ``nlp.pipe()`` batches each language once per input
batch. NULL/empty text rows (and rows whose auto-detected language lacks a default
pipeline) simply emit nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, cast

import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import BindParams, ProcessParams
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector

from . import pipelines
from .meta import object_tags
from .schema_utils import field

_SRC = "vgi_nlp/tables.py"

#: A self-contained input relation used by per-function examples so they
#: bind/execute against the worker without depending on any pre-existing table.
_DEMO_INPUT = "(SELECT 1 AS id, 'Apple unveiled the new iPhone in California yesterday. Critics loved it.' AS body)"


#: Guaranteed-runnable, catalog-qualified examples (VGI509 / VGI906). Each `sql`
#: is self-contained -- it builds its own one-row input relation inline -- so it
#: executes against an attached `nlp` worker without any pre-existing table.
#: `expected_result` is omitted on purpose (NER/token output is model-version
#: dependent; the linter only needs each query to execute cleanly).
_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Extract named entities from a literal document.",
            "sql": (
                "SELECT ent_text, label FROM nlp.entities("
                "(SELECT 1 AS id, 'Apple was founded by Steve Jobs in California.' AS body), "
                "id := 'id', lang := 'en')"
            ),
        },
        {
            "description": "Tokenize a literal document with part-of-speech tags.",
            "sql": (
                "SELECT token, pos FROM nlp.tokens("
                "(SELECT 1 AS id, 'The quick brown fox jumps.' AS body), "
                "id := 'id', lang := 'en')"
            ),
        },
        {
            "description": "Split a literal document into sentences.",
            "sql": (
                "SELECT sent_index, sentence FROM nlp.sentences("
                "(SELECT 1 AS id, 'First sentence here. Second one follows.' AS body), "
                "id := 'id', lang := 'en')"
            ),
        },
        {
            "description": "Extract noun chunks from a literal document.",
            "sql": (
                "SELECT chunk, root FROM nlp.noun_chunks("
                "(SELECT 1 AS id, 'The big red car drove down the long road.' AS body), "
                "id := 'id', lang := 'en')"
            ),
        },
    ]
)


@dataclass(slots=True, frozen=True, kw_only=True)
class NlpTableArgs:
    """Arguments shared by every table-in-out NLP function."""

    data: Annotated[TableInput, Arg(0, doc="Input relation (a (SELECT ...) subquery)")]
    id: Annotated[str, Arg("id", default="", doc="Passthrough column copied onto every emitted row")]
    text: Annotated[str, Arg("text", default="", doc="Text column name; default = the sole/first non-id column")]
    lang: Annotated[str, Arg("lang", default="", doc="Pipeline language (ISO-639); '' = auto-detect per row")]
    model: Annotated[str, Arg("model", default="", doc="Override spaCy model name; '' = default for lang")]


def _ex(name: str, extra: str = "") -> list[FunctionExample]:
    args = f", {extra}" if extra else ""
    return [
        FunctionExample(
            sql=f"SELECT * FROM nlp.{name}({_DEMO_INPUT}, id := 'id'{args})",
            description=f"Explode a literal document into {name}",
        )
    ]


def _resolve_text_column(input_schema: pa.Schema, id_col: str, requested: str) -> str:
    """Pick the text column: explicit ``text :=``, else the sole/first non-id column."""
    if requested:
        if requested not in input_schema.names:
            raise ValueError(
                f"text column {requested!r} not found in input; available: {', '.join(input_schema.names)}"
            )
        return requested
    candidates: list[str] = [n for n in input_schema.names if n != id_col]
    if not candidates:
        raise ValueError("input relation has no text column (every column is the id)")
    return candidates[0]


class _ExplodeFunction(TableInOutGenerator[NlpTableArgs]):
    """Base: run a spaCy pipeline per input batch and explode each Doc into rows.

    Subclasses declare their emitted (non-id) output fields via ``emit_fields`` and
    turn a Doc into row dicts via ``explode``.
    """

    FunctionArguments = NlpTableArgs

    # --- subclass hooks ---------------------------------------------------

    @classmethod
    def emit_fields(cls) -> list[pa.Field]:
        """The output columns this function emits, excluding the optional id."""
        raise NotImplementedError

    @classmethod
    def explode(cls, doc: Any) -> list[dict[str, Any]]:
        """Turn one spaCy Doc into zero-or-more output-row dicts (id excluded)."""
        raise NotImplementedError

    # --- framework ---------------------------------------------------------

    @classmethod
    def on_bind(cls, params: BindParams[NlpTableArgs]) -> BindResponse:
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        args = params.args
        if args.id and args.id not in input_schema.names:
            raise ValueError(f"id column {args.id!r} not found in input; available: {', '.join(input_schema.names)}")
        text_col = _resolve_text_column(input_schema, args.id, args.text)
        # Validate the text column is string-ish at plan time for a clear error.
        text_type = input_schema.field(text_col).type
        if not (pa.types.is_string(text_type) or pa.types.is_large_string(text_type)):
            raise ValueError(f"text column {text_col!r} must be VARCHAR, got {text_type}")
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(cls.emit_fields())
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def process(
        cls,
        params: ProcessParams[NlpTableArgs],
        state: None,
        batch: pa.RecordBatch,
        out: OutputCollector,
    ) -> None:
        args = params.args
        output_schema = params.output_schema
        assert params.init_call is not None
        input_schema = params.init_call.bind_call.input_schema
        assert input_schema is not None
        text_col = _resolve_text_column(input_schema, args.id, args.text)

        texts: list[str | None] = batch.column(text_col).to_pylist()
        id_values = batch.column(args.id).to_pylist() if args.id else None

        emit_names = [f.name for f in cls.emit_fields()]
        columns: dict[str, list[Any]] = {name: [] for name in output_schema.names}

        buckets = pipelines.group_by_pipeline(texts, lang=args.lang or None, model=args.model or None)
        # Process each pipeline's rows in order, then re-sort by source row index
        # so output is stable. We collect (row_index, row_dict) then flatten.
        produced: list[tuple[int, dict[str, Any]]] = []
        for model_name, idxs in buckets.items():
            pipe = pipelines.load_spacy_by_name(model_name)
            # group_by_pipeline only buckets indices whose text is a non-empty
            # str, so texts[i] is never None here.
            docs = pipe.pipe(
                (cast(str, texts[i]) for i in idxs),
                batch_size=pipelines.batch_size(),
            )
            for i, doc in zip(idxs, docs, strict=False):
                for row in cls.explode(doc):
                    produced.append((i, row))

        produced.sort(key=lambda pr: pr[0])
        for i, row in produced:
            if args.id:
                columns[args.id].append(id_values[i])  # type: ignore[index]
            for name in emit_names:
                columns[name].append(row.get(name))

        out.emit(pa.RecordBatch.from_pydict(columns, schema=output_schema))


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------


class Entities(_ExplodeFunction):
    """Named-entity recognition: one row per entity (PERSON, ORG, GPE, DATE, ...)."""

    class Meta:
        """Function metadata."""

        name = "entities"
        description = "Named entities per text row: (id, ent_text, label, start_char, end_char)"
        categories = ["ner"]
        examples = _ex("entities")
        tags = {
            **object_tags(
                "Extract Named Entities",
                "Named-entity recognition as a table-in-out function: each input text row "
                "explodes into **one row per detected entity** (people, organizations, places, "
                "dates, money, ...) found by the spaCy NER pipeline.\n\n"
                "**When to use:** pull structured mentions out of unstructured text -- e.g. "
                "every company named across a news corpus, or every date in a contract -- so "
                "you can aggregate, filter, or join on them.\n\n"
                "**Input:** a `(SELECT ...)` relation passed positionally; `id := 'col'` names a "
                "passthrough key copied onto every emitted row (so you can join entities back to "
                "the source); `text :=`, `lang :=`, and `model :=` select the text column, pin "
                "the language, and override the model. **Output:** rows of `(ent_text, label, "
                "start_char, end_char)` plus the optional id. Text rows with no entities emit "
                "nothing; output order is not guaranteed -- add `ORDER BY` for determinism.",
                "# entities\n\n"
                "Runs spaCy named-entity recognition over a text column, emitting one row per "
                "entity span.\n\n"
                "```sql\n"
                "SELECT * FROM nlp.entities((SELECT id, body FROM articles), id := 'id');\n"
                "```\n\n"
                "Each output row carries the entity text, its type label (`PERSON`, `ORG`, "
                "`GPE`, `DATE`, `MONEY`, ...), and the character span within the source. Use "
                "`id :=` to join results back to the source row; rows with no entities produce "
                "no output.",
                "named entity recognition, ner, entities, people organizations places, "
                "person org gpe date money, extract entities, spacy",
                _SRC,
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `<id>` | (input) | Passthrough id from `id :=`, copied onto every row (omitted if no `id`). |\n"
                "| `ent_text` | VARCHAR | The entity span text. |\n"
                "| `label` | VARCHAR | Entity type (`PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`, ...). |\n"
                "| `start_char` | INTEGER | Start character offset within the source text. |\n"
                "| `end_char` | INTEGER | End character offset within the source text. |"
            ),
            "vgi.executable_examples": _EXECUTABLE_EXAMPLES,
        }

    @classmethod
    def emit_fields(cls) -> list[pa.Field]:
        """Output columns emitted per row (excluding the optional id)."""
        return [
            field("ent_text", pa.string(), "The entity span text.", nullable=False),
            field("label", pa.string(), "Entity type (PERSON, ORG, GPE, DATE, MONEY, ...).", nullable=False),
            field("start_char", pa.int32(), "Start character offset within the source text.", nullable=False),
            field("end_char", pa.int32(), "End character offset within the source text.", nullable=False),
        ]

    @classmethod
    def explode(cls, doc: Any) -> list[dict[str, Any]]:
        """Turn one spaCy Doc into output-row dicts (id excluded)."""
        return [
            {
                "ent_text": ent.text,
                "label": ent.label_,
                "start_char": int(ent.start_char),
                "end_char": int(ent.end_char),
            }
            for ent in doc.ents
        ]


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


class Tokens(_ExplodeFunction):
    """Tokenization + POS: one row per token (token, lemma, pos, tag, is_stop, dep)."""

    class Meta:
        """Function metadata."""

        name = "tokens"
        description = "Tokens per text row: (id, token, lemma, pos, tag, is_stop, dep)"
        categories = ["tokenization", "pos"]
        examples = _ex("tokens")
        tags = {
            **object_tags(
                "Tokenize With POS Tags",
                "Tokenization plus linguistic annotation as a table-in-out function: each input "
                "text row explodes into **one row per token**, each carrying its lemma, "
                "part-of-speech, fine-grained tag, stop-word flag, and dependency relation from "
                "the spaCy pipeline.\n\n"
                "**When to use:** linguistic analysis, building token-level features, counting "
                "parts of speech, or filtering to content words -- anything that needs the words "
                "of a document as queryable rows.\n\n"
                "**Input:** a `(SELECT ...)` relation passed positionally; `id := 'col'` names a "
                "passthrough key; `text :=`, `lang :=`, `model :=` pick the text column, pin the "
                "language, and override the model. **Output:** rows of `(token, lemma, pos, tag, "
                "is_stop, dep)` plus the optional id. Output order is not guaranteed -- add "
                "`ORDER BY` for determinism.",
                "# tokens\n\n"
                "Tokenizes a text column with spaCy and emits one row per token with rich "
                "annotations.\n\n"
                "```sql\n"
                "SELECT * FROM nlp.tokens((SELECT id, body FROM docs), id := 'id');\n"
                "```\n\n"
                "Each row gives the token text, its lemma, coarse and fine POS tags, a stop-word "
                "flag, and its dependency label. Filter on `is_stop` or `pos` to isolate content "
                "words; use `id :=` to join tokens back to their source document.",
                "tokenize, tokenization, tokens, part of speech, pos tagging, lemma, "
                "dependency parse, stop word flag, spacy",
                _SRC,
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `<id>` | (input) | Passthrough id from `id :=`, copied onto every row (omitted if no `id`). |\n"
                "| `token` | VARCHAR | The token text. |\n"
                "| `lemma` | VARCHAR | The token's lemma (dictionary form). |\n"
                "| `pos` | VARCHAR | Coarse universal part-of-speech tag. |\n"
                "| `tag` | VARCHAR | Fine-grained part-of-speech tag. |\n"
                "| `is_stop` | BOOLEAN | Whether the token is a stop-word. |\n"
                "| `dep` | VARCHAR | Syntactic dependency relation. |"
            ),
        }

    @classmethod
    def emit_fields(cls) -> list[pa.Field]:
        """Output columns emitted per row (excluding the optional id)."""
        return [
            field("token", pa.string(), "The token text.", nullable=False),
            field("lemma", pa.string(), "The token's lemma (dictionary form).", nullable=False),
            field("pos", pa.string(), "Coarse universal part-of-speech tag.", nullable=False),
            field("tag", pa.string(), "Fine-grained part-of-speech tag.", nullable=False),
            field("is_stop", pa.bool_(), "Whether the token is a stop-word.", nullable=False),
            field("dep", pa.string(), "Syntactic dependency relation.", nullable=False),
        ]

    @classmethod
    def explode(cls, doc: Any) -> list[dict[str, Any]]:
        """Turn one spaCy Doc into output-row dicts (id excluded)."""
        return [
            {
                "token": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "is_stop": bool(tok.is_stop),
                "dep": tok.dep_,
            }
            for tok in doc
        ]


# ---------------------------------------------------------------------------
# sentences
# ---------------------------------------------------------------------------


class Sentences(_ExplodeFunction):
    """Sentence segmentation: one row per sentence (sent_index, sentence)."""

    class Meta:
        """Function metadata."""

        name = "sentences"
        description = "Sentences per text row: (id, sent_index, sentence) -- chunking for embeddings"
        categories = ["segmentation"]
        examples = _ex("sentences")
        tags = {
            **object_tags(
                "Split Into Sentences",
                "Sentence segmentation as a table-in-out function: each input text row explodes "
                "into **one row per sentence**, in document order, with a 0-based index.\n\n"
                "**When to use:** chunk long documents into sentence-sized units before "
                "embedding/retrieval, sentence-level sentiment, or any per-sentence analysis.\n\n"
                "**Input:** a `(SELECT ...)` relation passed positionally; `id := 'col'` names a "
                "passthrough key copied onto every sentence row; `text :=`, `lang :=`, `model :=` "
                "pick the text column, pin the language, and override the model. **Output:** rows "
                "of `(sent_index, sentence)` plus the optional id. The `sent_index` preserves "
                "order; output row order across documents is otherwise not guaranteed -- add "
                "`ORDER BY id, sent_index` for determinism.",
                "# sentences\n\n"
                "Splits a text column into sentences with spaCy, emitting one indexed row per "
                "sentence.\n\n"
                "```sql\n"
                "SELECT * FROM nlp.sentences((SELECT id, content FROM docs), id := 'id');\n"
                "```\n\n"
                "Each row carries the 0-based `sent_index` and the trimmed sentence text. This "
                "is the standard first step for sentence-level embeddings or retrieval chunking; "
                "use `id :=` to keep sentences tied to their source document.",
                "sentence segmentation, sentence splitting, sentences, sentence tokenizer, "
                "chunking, sbd, embeddings chunks, spacy",
                _SRC,
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `<id>` | (input) | Passthrough id from `id :=`, copied onto every row (omitted if no `id`). |\n"
                "| `sent_index` | INTEGER | 0-based index of the sentence within the source text. |\n"
                "| `sentence` | VARCHAR | The sentence text. |"
            ),
        }

    @classmethod
    def emit_fields(cls) -> list[pa.Field]:
        """Output columns emitted per row (excluding the optional id)."""
        return [
            field("sent_index", pa.int32(), "0-based index of the sentence within the source text.", nullable=False),
            field("sentence", pa.string(), "The sentence text.", nullable=False),
        ]

    @classmethod
    def explode(cls, doc: Any) -> list[dict[str, Any]]:
        """Turn one spaCy Doc into output-row dicts (id excluded)."""
        rows = []
        for idx, sent in enumerate(doc.sents):
            rows.append({"sent_index": idx, "sentence": sent.text.strip()})
        return rows


# ---------------------------------------------------------------------------
# noun_chunks
# ---------------------------------------------------------------------------


class NounChunks(_ExplodeFunction):
    """Noun-phrase extraction: one row per noun chunk (chunk, root)."""

    class Meta:
        """Function metadata."""

        name = "noun_chunks"
        description = "Noun chunks per text row: (id, chunk, root) -- keyword/topic candidates"
        categories = ["keywords"]
        examples = _ex("noun_chunks")
        tags = {
            **object_tags(
                "Extract Noun Chunks",
                "Noun-phrase extraction as a table-in-out function: each input text row explodes "
                "into **one row per noun chunk** (a base noun phrase such as 'the long road'), "
                "each with its head/root token.\n\n"
                "**When to use:** mine candidate keywords, topics, or product/feature mentions "
                "from free text without training a model -- noun chunks are a cheap, "
                "high-recall source of 'what this text is about'.\n\n"
                "**Input:** a `(SELECT ...)` relation passed positionally; `id := 'col'` names a "
                "passthrough key; `text :=`, `lang :=`, `model :=` pick the text column, pin the "
                "language, and override the model. **Output:** rows of `(chunk, root)` plus the "
                "optional id. Text rows with no noun chunks emit nothing; output order is not "
                "guaranteed -- add `ORDER BY` for determinism.",
                "# noun_chunks\n\n"
                "Extracts base noun phrases from a text column with spaCy, emitting one row per "
                "chunk.\n\n"
                "```sql\n"
                "SELECT * FROM nlp.noun_chunks((SELECT id, body FROM docs), id := 'id');\n"
                "```\n\n"
                "Each row gives the noun-phrase text and its head token (`root`). Aggregate the "
                "chunks to surface frequent topics/keywords, or join back to the source via "
                "`id :=`.",
                "noun chunks, noun phrases, keyword extraction, topic candidates, key phrases, "
                "phrase extraction, np chunking, spacy",
                _SRC,
            ),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `<id>` | (input) | Passthrough id from `id :=`, copied onto every row (omitted if no `id`). |\n"
                "| `chunk` | VARCHAR | The noun-phrase text. |\n"
                "| `root` | VARCHAR | The head/root token of the noun phrase. |"
            ),
        }

    @classmethod
    def emit_fields(cls) -> list[pa.Field]:
        """Output columns emitted per row (excluding the optional id)."""
        return [
            field("chunk", pa.string(), "The noun-phrase text.", nullable=False),
            field("root", pa.string(), "The head/root token of the noun phrase.", nullable=False),
        ]

    @classmethod
    def explode(cls, doc: Any) -> list[dict[str, Any]]:
        """Turn one spaCy Doc into output-row dicts (id excluded)."""
        return [{"chunk": nc.text, "root": nc.root.text} for nc in doc.noun_chunks]


TABLE_FUNCTIONS: list[type] = [
    Entities,
    Tokens,
    Sentences,
    NounChunks,
]
