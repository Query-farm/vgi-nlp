# Text cleaning and lexicon scores through VGI

[01 — Audit cleaning and sentiment](01_audit_cleaning_and_sentiment.ipynb) compares original English text with spaCy stopword removal, then changes only the policy to normalization. The authored corpus is not collected user data or human-labeled ground truth.

## Run independently

Requires uv, Git and Python 3.13/3.14 for the notebook. The worker specifically runs Python 3.13 because its compiled NLP dependencies do not support Python 3.14. Initial internet access downloads published wheels, the public Git source and signed VGI extension. No credentials or remote text API are used; inference runs locally through the worker.

```sh
cd notebooks
uv sync --frozen
uv run --frozen python execute.py --timeout 240
uv run --frozen python execute.py --parameters '{"CLEANING_METHOD":"normalize"}' --output-dir artifacts/experiment --timeout 240
```

Select `.venv/bin/python` in Jupyter, or register it:

```sh
uv run --frozen python -m ipykernel install --user --name vgi-nlp-notebooks --display-name "Python (vgi-nlp notebooks)"
```

The runner uses a temporary kernel spec and notebook-directory cwd, requires a separate destination for parameter overrides, and saves only after success. Unknown and invalid parameters fail without replacing the last successful default. Experiment copies include helpers, branding, lock and README. Generated artifacts/environments are ignored.

## Worker and model pins

Visible ATTACH runs the unchanged public Git revision `1bc38daf487fb5c65deb0306ebcaa557a135a018` through `vgi-nlp-worker`. It pins `vgi-python 0.51.0`, `vgi-rpc 0.53.1`, Arrow 26.0.1, spaCy 3.8.18, thinc 8.3.14, NumPy 2.5.4, fasttext-predict 0.9.2.4 and VADER 3.3.2. The source package directly pins the English `en_core_web_sm 3.8.0` wheel. The separate notebook lock pins published Haybarn 2.0.0rc11. No root checkout/environment discovery or engine build is involved. Upgrade pins together only after rerunning the notebook, experiment and source suite.

The example pins English and the English model for all spaCy calls. It never performs language detection. Optional fastText downloading is disabled during the notebook and the previous setting restored at cleanup. A pre-existing fastText cache may be loaded by the worker's best-effort startup warmup but does not affect these English-pinned outputs. No model is supplied by the notebook Python kernel.

## Interpretation

The 13 SQL-authored cases include negation, emphasis, mixed wording, a context-sensitive phrase, empty text and NULL. VADER scores are dimensionless lexicon-and-rules outputs, not probabilities or measures of a person's emotion. Worker labels use ±0.05 thresholds; unavailable scores remain missing. The paired chart excludes unavailable pairs with explicit counts. Token categories track a limited negation list, other stopwords, punctuation and remaining tokens. Removing multiple features at once does not isolate one causal effect; there are no accuracy claims or universal cleaning recommendations.

Validation outcomes are recorded after default and standalone normalization runs, chart review and required source checks.

Validated on macOS arm64: default stopword removal 8.8 s and detached notebooks-only normalization 8.4 s both pass ten SQL checks and produce two reviewed charts. Initial remote-worker cold launch also succeeded (36.4 s). Stopword removal changes three labels among ten comparable pairs, with three unavailable pairs and maximum absolute score shift 1.1585. Normalization changes zero labels among eleven comparable pairs, with two unavailable pairs and maximum shift 0.1101. Listed negation tokens are 4 → 0 under stripping and 4 → 4 under normalization; punctuation 11 → 0 versus 11 → 11. Three runner rejection/preservation guards pass. Required source gates pass: 60 unit tests with models available, 96 SQL assertions across five cases, Ruff lint/format, mypy over eight source files, docstring lint and executed metadata 100/100 L2 with zero findings. Root source and lockfile are unchanged.
