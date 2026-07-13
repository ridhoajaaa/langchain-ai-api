FROM python:3.11-slim

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install system dependencies needed for sentence-transformers
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for better caching
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . $HOME/app/

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run FastAPI with Uvicorn on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
