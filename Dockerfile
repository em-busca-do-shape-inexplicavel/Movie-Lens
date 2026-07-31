FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install project dependencies from pyproject.toml
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY params.yaml ./params.yaml
COPY data ./data

RUN pip install --upgrade pip && pip install .

# Default execution: training pipeline
CMD ["python", "scripts/train.py", "--params", "params.yaml", "--output-dir", "artifacts"]
