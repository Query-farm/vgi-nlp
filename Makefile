# vgi-nlp worker -- dev and test targets.
#
# Usage:
#   make test        # unit (pytest) + SQL (end-to-end via haybarn-unittest)
#   make test-unit   # pytest only
#   make test-sql    # SQL end-to-end only (ensures models, runs haybarn glob)
#   make models      # download the spaCy + fastText models the worker needs
#
# The SQL suite drives the worker as a real subprocess over stdio: haybarn-unittest
# ATTACHes `${VGI_NLP_WORKER}`, then runs the .test files in test/sql/.

# Worker stdio command (overridable). The PEP-723 header in nlp_worker.py pins the
# spaCy en_core_web_sm wheel, so `uv run` gives the worker its model.
WORKER_STDIO   ?= uv run --python 3.13 nlp_worker.py

# haybarn-unittest: the DuckDB sqllogictest runner (uv tool install haybarn-unittest).
HAYBARN        ?= haybarn-unittest
TEST_DIR        = .
TEST_PATTERN    = test/sql/*

# fastText language-ID model location (the worker also searches ~/.cache/vgi-nlp).
FASTTEXT_DIR   ?= $(HOME)/.cache/vgi-nlp
FASTTEXT_MODEL  = $(FASTTEXT_DIR)/lid.176.ftz
FASTTEXT_URL    = https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

.PHONY: test test-unit test-sql pytest models fasttext spacy lint typecheck

test: test-unit test-sql

test-unit: pytest

pytest:
	uv run pytest -q

# End-to-end SQL: ensure models are present, then run the haybarn glob with the
# worker command exported. `uv run nlp_worker.py` resolves the spaCy model from
# the script's pinned deps; fastText is fetched to ~/.cache/vgi-nlp by `make models`.
test-sql: fasttext
	@command -v $(HAYBARN) >/dev/null 2>&1 || { \
		echo "ERROR: $(HAYBARN) not found. Install it with:" >&2; \
		echo "  uv tool install haybarn-unittest" >&2; \
		echo "  (then ensure ~/.local/bin is on PATH)" >&2; \
		exit 1; \
	}
	VGI_NLP_WORKER="$(WORKER_STDIO)" $(HAYBARN) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# Download every model the worker/tests need.
models: spacy fasttext

# The spaCy model ships as a pinned wheel in nlp_worker.py's PEP-723 deps, so the
# stdio worker always has it. This target also installs it into the dev .venv so
# the in-process unit tests (which load it directly) are not skipped.
spacy:
	uv run --python 3.13 python -c "import en_core_web_sm" 2>/dev/null \
		|| uv run --python 3.13 python -m spacy download en_core_web_sm

# fastText lid.176: download once to ~/.cache/vgi-nlp if absent.
fasttext: $(FASTTEXT_MODEL)

$(FASTTEXT_MODEL):
	mkdir -p "$(FASTTEXT_DIR)"
	curl -L -o "$(FASTTEXT_MODEL)" "$(FASTTEXT_URL)"

lint:
	uv run ruff check .

typecheck:
	uv run mypy vgi_nlp/
