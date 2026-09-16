FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_CACHE_HOME=/app/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    GLOG_minloglevel=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ocr

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.0 torchvision==0.29.0
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py meter_reading.py counter_reader.py counter_vision.py api.py ./
RUN mkdir -p /app/models && chown -R ocr:ocr /app

USER ocr
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
