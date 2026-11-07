# Copyright 2026 Query Farm LLC - https://query.farm
#
# Single image serving BOTH transports of the vgi-nlp worker:
#   docker run ... IMG            -> HTTP server on $PORT (default 8000; /health, VGI RPC)
#   docker run -i ... IMG stdio   -> stdio worker DuckDB spawns on-host
# See docker-entrypoint.sh. The classical-NLP models (spaCy en_core_web_sm +
# fastText lid.176) are baked into the image so the first query is fast and the
# container is fully self-contained (no network at query time).
# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Query-farm/vgi-nlp

LABEL org.opencontainers.image.title="vgi-nlp" \
      org.opencontainers.image.description="Classical NLP (spaCy + fastText + VADER) for DuckDB via VGI (stdio + HTTP)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="LicenseRef-QueryFarm-Source-Available-1.0" \
      farm.query.vgi.transports='["http","stdio"]'

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    VGI_NLP_FASTTEXT_MODEL=/app/models/lid.176.ftz

WORKDIR /app

# The worker + HTTP-serving extra install from the source tree (all deps,
# including fasttext-predict and the en_core_web_sm pipeline, are prebuilt
# wheels). curl backs the HEALTHCHECK + CI /health smoke.
COPY pyproject.toml README.md LICENSE ./
COPY vgi_nlp ./vgi_nlp
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && pip install '.[serve]' \
    && rm -rf /var/lib/apt/lists/*

# fastText lid.176 language-ID model (~917 KB), located via VGI_NLP_FASTTEXT_MODEL.
RUN mkdir -p /app/models \
    && curl -fsSL -o /app/models/lid.176.ftz \
       https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
