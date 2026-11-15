# Classical NLP examples

Two independently executable notebooks retain text provenance while auditing local model outputs:

- [Does text cleaning preserve a sentiment score?](01_audit_cleaning_and_sentiment.ipynb) preserves the cleaning/sentiment walkthrough. Both transforms are materialized visibly, one is selected for VADER comparison, and token counts explain what changed without claiming a causal or accuracy benchmark.
- [Do extracted spans still identify their source?](02_trace_extracted_spans_to_text.ipynb) retains all six input documents, separately counts sentence/entity rows, reconstructs half-open entity offsets against the exact processed text, and distinguishes annotation coverage from recognition recall.

## Run locally or in Colab

Open either notebook's Colab badge and run all cells. Setup installs the base shared package and retrieves one immutable hash-verified presentation companion. For a local copy of this directory, install [uv](https://docs.astral.sh/uv/) and Git, then:

```bash
uv sync --locked
uv run --locked vgi-notebooks 01_audit_cleaning_and_sentiment.ipynb
uv run --locked vgi-notebooks 02_trace_extracted_spans_to_text.ipynb
```

The independent notebook lock includes the shared `[runner]` extra. Colab installs only the base package, preserving its existing kernel/Jupyter client. For an interactive local kernel:

```bash
uv run --locked python -m ipykernel install --user --name vgi-nlp-notebooks --display-name 'Python (VGI NLP examples)'
```

Both books visibly attach the published release with the same short command:

```sql
ATTACH 'nlp' (TYPE vgi, LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-nlp@v0.4.0 vgi-nlp-worker');
```

The release declares its runtime dependencies and official `en_core_web_sm` **3.8.0** wheel. No local worker checkout, manually built binary or transitive `--with` override is needed. First use needs internet for packages and the published Haybarn/VGI extension; text processing is local. The notebook uses a fresh temporary SQL-secret directory and disables persistent secrets before connection creation, then verifies an empty inventory before attachment and at cleanup.

The authored inputs are English. Each spaCy operation explicitly names both `lang='en'` and `model='en_core_web_sm'`. `supported_languages` validates that mapping, but its 24 advertised routes are not proof that 24 models are installed. Optional fastText downloading is disabled with `VGI_NLP_NO_MODEL_DOWNLOAD=1` during execution; cleanup restores its previous value. No language-ID call or external inference service is used. Only the released English pipeline and VADER are tested.

## Focused experiments

```bash
uv run --locked vgi-notebooks 01_audit_cleaning_and_sentiment.ipynb \
  --parameters '{"CLEANING_METHOD":"normalize"}' --output-dir artifacts/normalize-sentiment
uv run --locked vgi-notebooks 02_trace_extracted_spans_to_text.ipynb \
  --parameters '{"TEXT_FORM":"normalized"}' --output-dir artifacts/normalize-extraction
```

The first selects normalization instead of stopword removal while preserving the original corpus, both stored transform alternatives and original sentiment scores. The second changes only the exact text passed to extraction; original bodies remain unchanged. Offsets must be interpreted against that selected text, never reused against another form. The runner rejects missing or source-directory output destinations and unknown parameters, preserves successful defaults, and saves experiment copies separately.

All corpora, labels and examples are invented. They are neither customer traffic nor independent ground truth. VADER outputs are dimensionless lexicon-and-rule scores, not probabilities or measurements of a person's emotions. Named-entity labels are model predictions, not verified facts. Empty/NULL inputs remain in audit populations; missing scores are not filled with zero. Zero emitted rows describe the known input/pipeline behavior, not an absent document's real contents.

## Coverage and validation

Actual v0.4.0 exposes **12 logical named surfaces**: 16 function rows plus the same-named `supported_languages` table alias, counted once. There are no views, COPY handlers, authenticated or write operations. These two stories meaningfully cover **8/12 released surfaces** and **8/12 reviewed current-interface surfaces**, exceeding the six-surface requirement. Uncovered names remain explicit: `detect_lang`, `detect_lang_conf`, `lemmatize`, and `noun_chunks`. No coverage is claimed from an untaken CASE branch: both cleaning functions run and their returned text is shown in the saved default.

[coverage.json](coverage.json) binds complete catalogs, overloads, native argument exports, bound table-in-out schemas, visible SQL, interpreted results and successful execution receipts. Table-in-out schema exports use an INTEGER ID example; an optional passthrough ID's type follows the caller's input schema. Current public source `638dd3037d0757df64fa4cb6795084c57c3780a4` adds composed examples/agent metadata only; compute methods, registration, schema and root dependencies are unchanged. [Source compatibility](coverage/source-compatibility.json) retains the full reviewed package diff and resolved package versions. Exact-current runtime count is **0**, distinct from reviewed interface compatibility. Proper title/subtitle come from actual released catalog metadata.

Each book has ten SQL invariants. Additional extraction checks reconcile entity/sentence counts and processed characters. Default and separate parameter runs use the actual release; a fresh notebook-only Python 3.12 kernel preloads eight plotting/Jupyter libraries and verifies unchanged versions plus `pip check`. This checks Colab-compatible setup, not manual operation of hosted Colab. All changed figures are visually reviewed; SVG chart geometry/text and content-addressed GitHub images are committed. Logs, PNG reviews and experiment copies remain ignored.

The short Git launcher resolves declared dependencies; actual tested versions are recorded and upgrades require validation. Worker source is unchanged and its [existing green source CI](https://github.com/Query-farm/vgi-nlp/actions/runs/43580007217) is reused. No worker rebuild, repeated full transport suite or deployment is performed. [STANDARD.md](STANDARD.md) links the maintained shared standard.
