FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["ledger-rec"]
CMD ["serve"]
