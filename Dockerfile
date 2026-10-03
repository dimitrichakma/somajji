FROM python:3.13-slim

# uv (pinned) for a fast, lockfile-exact install
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# deps first, so a code change doesn't bust the dependency layer.
# --no-default-groups: skip the `dev` group (pytest) in the deployed image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

# runtime code only (not tests/, eval/, docs/, specs/, web/)
COPY app ./app
COPY data ./data

ENV PATH="/app/.venv/bin:$PATH"

# Railway injects $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
