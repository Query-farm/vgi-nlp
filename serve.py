# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "vgi-python[http,oauth]>=0.51.0",
#     "spacy>=3.7",
#     "fasttext-predict>=0.9.2.4",
#     "vaderSentiment",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# ///
"""HTTP entrypoint for the NLP worker.

Forces the worker's CLI into HTTP mode (``Worker.main()`` serves stdio by
default) so callers only pass ``--host``/``--port``.
"""

import sys

from nlp_worker import NlpWorker

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    NlpWorker.main()
