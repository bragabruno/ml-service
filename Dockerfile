FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --no-dev

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "ml_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]