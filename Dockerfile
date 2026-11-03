FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY satwx ./satwx
COPY scripts ./scripts
COPY tests ./tests
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e .
ENTRYPOINT ["classify"]
