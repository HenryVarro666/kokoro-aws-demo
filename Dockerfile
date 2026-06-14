FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Bake model weights into the image (immutable artifact: deploy == ready,
# no startup-time downloads, rollback == switch image tag).
ENV HF_HOME=/opt/hf

WORKDIR /srv
COPY requirements.txt .
# torch pinned: vocoder output is sensitive to kernel/version changes
# (see the Graviton numerics incident) — upgrades must be deliberate.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake model weights into image layers (TTS + base ASR) so deploy == ready,
# no startup-time downloads.
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
