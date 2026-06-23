"""In-process VGI invocation helpers for the NLP worker test suite.

These drive a function through its real bind -> process lifecycle without
spawning a worker subprocess, so tests stay fast and debuggable. Scalars call
``compute`` directly (its declarative array-in/array-out contract); table-in-out
functions go through bind -> process so the ``id`` passthrough and dynamic output
schema are exercised.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_function import ProcessParams


def fasttext_available() -> bool:
    """Whether the fastText lid.176 model can be located (for skip guards)."""
    from vgi_nlp import pipelines

    try:
        pipelines._fasttext_model_path()
        return True
    except Exception:
        return False


def spacy_model_available(model_name: str = "en_core_web_sm") -> bool:
    """Whether a spaCy model is installed (for skip guards)."""
    try:
        import spacy

        spacy.load(model_name)
        return True
    except Exception:
        return False


class _Collector:
    """Captures batches emitted by a table-in-out ``process()`` call."""

    def __init__(self) -> None:
        self.batches: list[pa.RecordBatch] = []

    def emit(self, batch: pa.RecordBatch, *args: Any, **kwargs: Any) -> None:
        self.batches.append(batch)

    def finish(self) -> None:  # noqa: D102
        pass

    def client_log(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass


def run_table_function(
    func_cls: type,
    batch: pa.RecordBatch,
    *,
    named: dict[str, Any] | None = None,
) -> pa.Table:
    """Drive a table-in-out function through bind -> process over one input batch.

    ``named`` values may be plain Python values or ``pa.Scalar``; plain values are
    wrapped as string scalars (every NLP table arg is a string).
    """
    named_scalars = {k: v if isinstance(v, pa.Scalar) else pa.scalar(str(v)) for k, v in (named or {}).items()}
    args = Arguments(positional=(), named=named_scalars)

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE,
        input_schema=batch.schema,
    )
    bind_resp = func_cls.bind(bind_req)
    parsed = func_cls._parse_arguments(func_cls.FunctionArguments, args)
    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    params = ProcessParams(
        args=parsed,
        init_call=init_req,
        init_response=None,
        output_schema=bind_resp.output_schema,
        settings={},
        secrets={},
        storage=None,
    )

    out = _Collector()
    func_cls.process(params, None, batch, out)
    if not out.batches:
        return bind_resp.output_schema.empty_table()
    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
