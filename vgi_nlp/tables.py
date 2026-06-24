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
from .schema_utils import field


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
            sql=f"SELECT * FROM nlp.{name}((SELECT id, body FROM docs), id := 'id'{args})",
            description=f"Explode each document into {name}",
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
            "vgi.columns_md": (
                "| column | type | description |\n"
                "|---|---|---|\n"
                "| `<id>` | (input) | Passthrough id from `id :=`, copied onto every row (omitted if no `id`). |\n"
                "| `ent_text` | VARCHAR | The entity span text. |\n"
                "| `label` | VARCHAR | Entity type (`PERSON`, `ORG`, `GPE`, `DATE`, `MONEY`, ...). |\n"
                "| `start_char` | INTEGER | Start character offset within the source text. |\n"
                "| `end_char` | INTEGER | End character offset within the source text. |"
            ),
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
            "vgi.columns_md": (
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
            "vgi.columns_md": (
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
            "vgi.columns_md": (
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
