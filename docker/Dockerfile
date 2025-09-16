FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY . .
RUN pip install --upgrade pip && pip install -e .
CMD ["news-mvp", "--dry-run"]
