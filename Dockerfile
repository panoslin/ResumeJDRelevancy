FROM python:3.10-slim

# Set environment variables to improve Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create and set permissions for the home directory
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

WORKDIR /app

# Copy application files with appropriate permissions for the non-root user
COPY --chown=appuser:appuser . /app

RUN pip install --no-cache-dir -r requirements-api.txt

USER appuser
#ENV TRANSFORMERS_CACHE=/app/.cache
#ENV TORCH_HOME=/app/.cache/torch
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]