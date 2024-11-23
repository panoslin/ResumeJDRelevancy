FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . /app
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx &&\
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt &&\
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]