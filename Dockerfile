# syntax=docker/dockerfile:1
# Multi-stage, multi-arch (amd64 + arm64):
#   docker buildx build --platform linux/amd64,linux/arm64 -t furnace .
# Build (local):  docker build -t furnace .
# Run:            docker run --rm furnace --world-size 2 --max-steps 120
FROM python:3.12-slim AS builder
WORKDIR /app
# CPU-only torch wheels exist for both amd64 and arm64
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --prefix=/install torch==2.4.1 numpy
COPY . .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
COPY --from=builder --chown=appuser:appuser /app /app
RUN mkdir -p /app/checkpoints && chown appuser:appuser /app/checkpoints
USER appuser
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "furnace.py"]
CMD ["--help"]
