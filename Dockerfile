FROM python:3.12-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 \
 && pip install --no-cache-dir numpy==2.1.3 netCDF4==1.7.2 pandas==2.2.3 pillow==11.0.0 pytest==8.3.4
WORKDIR /app
COPY pyproject.toml README.md ./
COPY satwx ./satwx
COPY scripts ./scripts
COPY tests ./tests
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e .
ENV SATWX_CACHE=/tmp/satwx-cache
ENTRYPOINT ["classify"]
