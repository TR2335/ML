FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install ".[prod]"
RUN useradd --create-home --uid 10001 medintel \
    && mkdir -p /app/data /app/model \
    && python scripts/train_model.py --output /app/model/model.json --version baseline-logistic-v1 \
    && python scripts/train_model.py --output /app/model/readmission_model.json --version baseline-readmission-v1 --target-id readmission_30d \
    && chown -R medintel:medintel /app

ENV MODEL_PATH=/app/model/model.json \
    READMISSION_MODEL_PATH=/app/model/readmission_model.json
USER medintel

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
