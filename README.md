<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi/main/docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-nlp

A [VGI](https://github.com/query-farm/vgi-python) worker that brings **classical
NLP** into DuckDB/SQL: language detection, tokenization/lemmatization,
named-entity recognition, part-of-speech tagging, noun-phrase extraction, and
lexicon sentiment — all callable as SQL scalar and table functions. Built on
[spaCy](https://spacy.io/), [fastText](https://fasttext.cc/), and
[VADER](https://github.com/cjhutto/vaderSentiment).

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'nlp' (TYPE vgi, LOCATION 'uv run nlp_worker.py');

SELECT nlp.detect_lang(review) AS lang, count(*) FROM reviews GROUP BY 1;
SELECT id, nlp.sentiment(body) AS score FROM reviews;
SELECT * FROM nlp.entities((SELECT id, body FROM articles), id := 'id');
```

> **The marketplace gap it fills:** the community catalog has LLM extensions
> (`llm`, `flock`, `open_prompt`, `web_search`) but **no classical NLP**. For NER
> / sentiment / language-ID over millions of rows, a few-MB spaCy pipeline is
> orders of magnitude cheaper and faster than a per-row LLM call. This is the
> bulk-text-enrichment primitive that sits *upstream* of the LLM workers.

## How it maps spaCy onto SQL

spaCy processes text through a stateful, model-backed `Language` *pipeline*
(`nlp(text) -> Doc`). One-doc-in / one-value-out maps to a **scalar**; one-doc-in
/ many-rows-out (entities, tokens, sentences) maps to a **table-in-out** that
streams a text column and explodes each row.

| Area | SQL surface | VGI primitive |
| --- | --- | --- |
| **Language ID** | `nlp.detect_lang(text)` | scalar (fastText `lid.176`) |
| **Sentiment** | `nlp.sentiment(text)` → score in [-1, 1] | scalar (VADER lexicon) |
| **Entities (NER)** | `SELECT * FROM nlp.entities((SELECT id, text ...), id := 'id')` | table-in-out (1 row → N entity rows) |
| **Tokens / lemmas / POS** | `SELECT * FROM nlp.tokens((SELECT ...), id := 'id')` | table-in-out |
| **Sentences** | `SELECT * FROM nlp.sentences((SELECT ...), id := 'id')` | table-in-out (chunking) |
| **Noun phrases** | `SELECT * FROM nlp.noun_chunks((SELECT ...), id := 'id')` | table-in-out |
| **Cleaning** | `nlp.lemmatize(text)`, `nlp.strip_stopwords(text)`, `nlp.normalize(text)` | scalar |

## Conventions

Same family as [`vgi-sklearn`](https://github.com/query-farm/vgi-scikit-learn).

### Argument syntax: scalars are positional, table functions are named

DuckDB **scalar** functions take **positional** arguments and resolve overloads
by *arity* — the `name := value` named-argument syntax is a property of table
functions and macros, *not* scalars. So the spaCy-backed cleaners expose their
`lang` / `model` options as positional arity overloads (the same pattern
[`vgi-translate`](https://github.com/query-farm/vgi-translate) uses for
`translate(text, 'es')` vs `translate(text, 'es', 'en')`):

```sql
SELECT nlp.lemmatize(body)                      FROM reviews;  -- auto-detect per row
SELECT nlp.lemmatize(body, 'en')                FROM reviews;  -- pin the language
SELECT nlp.lemmatize(body, 'en', 'en_core_web_trf') FROM reviews;  -- pin the model
```

The **table-in-out** functions (`entities` / `tokens` / `sentences` /
`noun_chunks`) *do* accept the `name := value` form:

- The input relation is passed as a `(SELECT ...)` subquery (positional).
- **`id := 'col'`** names a passthrough column, copied onto every emitted row so
  you can join the entities/tokens/sentences back to the source row they came
  from. Optional — omit it and no id is carried.
- **`text := 'col'`** names the text column (default: the sole / first non-`id`
  column).
- **`lang := 'en'`** pins the pipeline language. The default is **auto-detect per
  row** via fastText, so a mixed-language column "just works" at some throughput
  cost. Pin `lang` when the corpus is monolingual.
- **`model := 'en_core_web_trf'`** overrides the spaCy pipeline (`_trf`
  transformer variants for accuracy, `_sm` for speed).

Pipelines are **loaded once and cached** in the persistent pooled worker
process — the cost VGI is built to amortize — and `nlp.pipe()` batches each
language. Tune the minibatch with the `VGI_NLP_BATCH_SIZE` env var.

## Function catalog

### Scalars

| Function | Returns | Notes |
| --- | --- | --- |
| `detect_lang(text)` | `VARCHAR` | ISO-639 language code (fastText `lid.176`) |
| `detect_lang_conf(text)` | `FLOAT` | confidence 0–1 of the detected language |
| `sentiment(text)` | `FLOAT` | VADER compound score in [-1, 1] |
| `sentiment_label(text)` | `VARCHAR` | `neg` / `neu` / `pos` |
| `lemmatize(text [, lang [, model]])` | `VARCHAR` | tokens replaced by their lemma; `lang`/`model` are **positional** overloads |
| `strip_stopwords(text [, lang [, model]])` | `VARCHAR` | stop-words + punctuation removed; `lang`/`model` are **positional** overloads |
| `normalize(text)` | `VARCHAR` | Unicode NFKC + lowercase + whitespace collapse |

```sql
SELECT product, avg(nlp.sentiment(body)) AS mood
FROM reviews WHERE nlp.detect_lang(body) = 'en'
GROUP BY product ORDER BY mood;
```

### Table-in-out (1 row → N rows)

| Function | Output columns |
| --- | --- |
| `entities` | `(id, ent_text, label, start_char, end_char)` — PERSON, ORG, GPE, DATE, MONEY, … |
| `tokens` | `(id, token, lemma, pos, tag, is_stop, dep)` |
| `sentences` | `(id, sent_index, sentence)` |
| `noun_chunks` | `(id, chunk, root)` |

```sql
-- entity frequency across a news corpus
SELECT label, ent_text, count(*) AS n
FROM nlp.entities((SELECT id, body FROM articles), id := 'id')
WHERE label IN ('ORG', 'PERSON')
GROUP BY 1, 2 ORDER BY n DESC LIMIT 20;

-- sentence chunks ready to hand to an embedding worker
SELECT id, sent_index, sentence
FROM nlp.sentences((SELECT id, content FROM docs), id := 'id');
```

## Models

Model loading is lazy: a pipeline is loaded the first time a row needs it (and
cached for the life of the worker process). On startup the worker additionally
*warms* the default English spaCy pipeline and the fastText language ID model, so
the one-time load cost is paid at spawn rather than inside the first query — this
keeps latency predictable and the end-to-end SQL suite deterministic. Warming is
best-effort: a missing model never blocks startup (the function that needs it
raises its own actionable error if actually called).

- **spaCy pipelines** are loaded by language. The default is the small (`_sm`)
  model per language (`en` → `en_core_web_sm`). Install the ones you need:

  ```sh
  uv run python -m spacy download en_core_web_sm
  ```

- **fastText `lid.176`** powers language detection and per-row routing. Download
  the compact (~917 KB) compressed model once:

  ```sh
  mkdir -p ~/.cache/vgi-nlp
  curl -L -o ~/.cache/vgi-nlp/lid.176.ftz \
      https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
  ```

  The worker searches `$PWD`, `~/.cache/vgi-nlp`, and the package directory, or
  set `VGI_NLP_FASTTEXT_MODEL` to an explicit path.

- **VADER** ships its lexicon with the `vaderSentiment` package — no download.

## Dependencies & licensing

- `spacy` — **MIT**; pipeline models (`en_core_web_sm`, …) **MIT / CC-BY-SA**
  per language.
- fastText language-ID model `lid.176` — **CC-BY-SA 3.0** (attribution required).
- `vaderSentiment` — **MIT**.

All permissive; the only attribution obligation is fastText's CC-BY-SA. This
worker itself is distributed under the Query Farm Source-Available License (see
`LICENSE`).

## Development

```sh
# set up the env (downloads the en_core_web_sm wheel via the dev extra)
uv sync --extra dev

# fetch the fastText language-ID model
mkdir -p ~/.cache/vgi-nlp
curl -L -o ~/.cache/vgi-nlp/lid.176.ftz \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

# run everything (unit + SQL end-to-end)
make test

# or individually:
make test-unit      # pytest
make test-sql       # end-to-end .test files via haybarn-unittest
make models         # download the spaCy + fastText models

# lint / type-check
uv run ruff check .
uv run mypy vgi_nlp
```

Two test layers:

- **Unit** (`tests/`, pytest) drive the real `bind → init → process` lifecycle
  in-process (fast, no subprocess) plus an end-to-end pass through
  `vgi.client.Client`. Tests that need a model are skipped automatically when it
  is not installed, so the suite is green on a bare checkout.
- **SQL end-to-end** (`test/sql/*.test`) run the worker as a real subprocess
  under DuckDB's sqllogictest runner ([`haybarn-unittest`](https://pypi.org/project/haybarn-unittest/)),
  exercising the full `ATTACH` → SQL path the way a user would. Install the
  runner with `uv tool install haybarn-unittest` (binary lands in `~/.local/bin`),
  then `make test-sql`. The worker command is exported as `VGI_NLP_WORKER` and the
  `.test` files `LOAD vgi` + `ATTACH 'nlp' (TYPE vgi, LOCATION '${VGI_NLP_WORKER}')`.

### Layout

```
nlp_worker.py        # stdio entrypoint + inline PEP 723 script metadata
serve.py             # HTTP entrypoint
vgi_nlp/
  pipelines.py       # loaded-once-and-cached spaCy / fastText / VADER lifecycle
  scalars.py         # detect_lang, sentiment, lemmatize, strip_stopwords, normalize, ...
  tables.py          # entities, tokens, sentences, noun_chunks (1 row → N rows)
  schema_utils.py    # Arrow-field / column-comment helpers
tests/               # pytest unit / integration tests
test/sql/            # haybarn-unittest .test files (end-to-end via ATTACH)
Makefile             # test / test-unit / test-sql / models targets
```

---

## Authorship & License

Written by [Query.Farm](https://query.farm).

Copyright 2026 Query Farm LLC - https://query.farm

