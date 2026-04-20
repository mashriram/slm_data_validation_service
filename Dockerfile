FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Copy dependencies definitions
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into system environment
RUN uv sync --frozen --no-cache

COPY . /app

EXPOSE 8002

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
