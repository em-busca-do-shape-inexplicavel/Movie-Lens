FROM python:3.11-slim-bookworm AS builder

ARG UV_VERSION=0.11.31

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Keep dependency installation cached while application code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim-bookworm AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid app --create-home app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --chown=app:app pyproject.toml uv.lock README.md params.yaml dvc.yaml dvc.lock ./
COPY --chown=app:app configs ./configs
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app src ./src
COPY --chown=app:app .dvc ./.dvc
COPY --chown=app:app data/raw.dvc ./data/raw.dvc

RUN dvc config core.no_scm true --local \
    && mkdir -p artifacts data/raw mlflow mlartifacts .dvc/cache .dvc-storage \
    && chown -R app:app /app

USER app

CMD ["dvc", "repro", "--force"]
