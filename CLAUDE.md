# CLAUDE.md — vgi-nlp

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker exposing **classical NLP** (spaCy + fastText
language-ID + VADER sentiment) to DuckDB/SQL. `nlp_worker.py` assembles every
function into one `nlp` catalog (single `main` schema) and runs it over stdio.
The point is bulk, cheap, per-row text enrichment that sits *upstream* of the
LLM workers — not an LLM wrapper.

## Layout

```
nlp_worker.py          repo-root stdio entry; PEP 723 inline deps (incl. the spaCy model wheel); main()
serve.py               HTTP entry shim
vgi_nlp/
  pipelines.py         loaded-once-and-cached spaCy/fastText/VADER lifecycle (the per-process state VGI pools)
  scalars.py           7 scalar functions (arity overloads for language/model options)
  tables.py            4 table-in-out functions (1 text row -> N rows, id passthrough)
  schema_utils.py      pa.Field comment helper
tests/                 pytest: scalars / tables / Client integration (model-gated tests self-skip)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / models / spacy / fasttext / lint / typecheck
```

## Scalars vs table functions — core convention (read first)

VGI **scalar functions are positional-only** (no `name := value`). So the
scalars that take a language/model option are exposed as **arity overloads**:
`lemmatize(text)` / `lemmatize(text, lang)` / `lemmatize(text, lang, model)`,
same for `strip_stopwords`. The **table-in-out** functions (`entities`,
`tokens`, `sentences`, `noun_chunks`) DO accept named args (`id :=`, `lang :=`,
`model :=`, `text :=`) because table functions support them.

## Sharp edges (learned the hard way)

1. **The worker's OWN environment must contain the spaCy model.** `uv run
   nlp_worker.py` builds the PEP 723 inline env — if `en_core_web_sm` is only in
   a dev `.venv` extra, the launched worker can't load it and every spaCy call
   fails. The model wheel is pinned in `nlp_worker.py`'s PEP 723 header for this
   reason. `make models` (or `make spacy` / `make fasttext`) provisions models
   for local dev; the fastText `lid.176` model caches under `~/.cache/vgi-nlp`.
2. **`haybarn-unittest` skips `require vgi`** — use explicit `statement ok` /
   `LOAD vgi;` in `.test` files (the ones here do).
3. **fastText confidence can exceed 1.0.** Observed `1.0000131` for German;
   don't assert `<= 1.0` on `detect_lang_conf`. (Bit both a unit test and would
   have bitten an SQL assertion.)
4. **Determinism in SQL assertions.** Table-in-out output order isn't
   guaranteed — use `ORDER BY` / `query ... rowsort`. Pin `lang` in cleaning
   tests rather than relying on per-row auto-detect for short/ambiguous strings.
   Under heavy concurrent CPU load the worker pool's cold model-load can
   transiently slow a first call; the suite is reliable when not CPU-starved.
5. **Auto-detect default.** With no `lang`, scalars/tables auto-detect per row
   via fastText (mixed-language columns work, at a throughput cost). Pin `lang`
   when the corpus is monolingual.

## Testing

```sh
uv run pytest -q              # unit (model-gated tests self-skip on bare checkout)
make models                   # provision spaCy + fastText models for local dev
make test-sql                 # E2E: haybarn-unittest over test/sql/*  (authoritative)
make test                     # both
uv run ruff check . && uv run mypy vgi_nlp/
```

`make test-sql` exports `VGI_NLP_WORKER="uv run --python 3.13 nlp_worker.py"`,
ensures models present, and runs `haybarn-unittest --test-dir . "test/sql/*"`
(install once: `uv tool install haybarn-unittest`). **The SQL suite is
authoritative** — it exercises the real RPC + model path that unit tests skip.
CI runs unit + lint plus a gated `e2e` job (installs worker deps from PyPI +
models + haybarn-unittest, launches the worker from the prepared venv so it
resolves `vgi-python` from PyPI).
