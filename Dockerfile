# ─── Tunix DPO :: GPU inference image ───────────────────────────────────────
# Builds a CUDA image with vLLM for serving the trained Gemma 3 1B IT model.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src/
COPY configs /app/configs/

RUN pip install -e '.[serving]'

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["tunix-serve"]
CMD ["--model", "/models/tunix_dpo_gemma_3_1b_it", "--port", "8000"]
