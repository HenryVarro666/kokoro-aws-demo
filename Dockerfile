FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Bake model weights into the image (immutable artifact: deploy == ready,
# no startup-time downloads, rollback == switch image tag).
ENV HF_HOME=/opt/hf

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Trigger one model download so the weights land in an image layer.
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
