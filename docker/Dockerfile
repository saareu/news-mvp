FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY . .
RUN pip install -U pip && pip install .
CMD ["python","-m","news_mvp.cli","etl","list-sources","--env","dev"]
