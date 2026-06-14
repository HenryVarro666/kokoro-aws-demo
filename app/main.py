import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz
ASR_MODEL = os.environ.get("ASR_MODEL", "base")  # faster-whisper base, CPU int8
_pipelines = {}
_asr = {}


def get_pipeline(lang_code: str):
    # 'a' = American English, 'z' = Mandarin
    if lang_code not in _pipelines:
        from kokoro import KPipeline  # lazy import so CI tests don't need the model
        _pipelines[lang_code] = KPipeline(
            lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _pipelines[lang_code]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("SKIP_MODEL_LOAD"):
        get_pipeline("a")  # warm English pipeline; health check passing == model ready
    yield


app = FastAPI(title="Kokoro TTS Demo", lifespan=lifespan)


class SynthRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice: str = "af_heart"
    lang: str = Field("a", pattern="^[az]$")
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.get("/health")
def health():
    return {"status": "ok"}


def _tts_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    pipe = get_pipeline(lang)
    chunks = []
    for _graphemes, _phonemes, audio in pipe(text, voice=voice, speed=speed):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
    if not chunks:
        raise HTTPException(400, "no audio generated")
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), SAMPLE_RATE, format="WAV")
    return buf.getvalue()


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    return Response(_tts_wav(req.text, req.voice, req.lang, req.speed),
                    media_type="audio/wav")


# ---------- ASR: speech -> text (base Whisper, CPU, no GPU / no fine-tuning) ----------
# Phase 2 split: this serves the BASE model now. Swap in a LoRA-fine-tuned,
# CTranslate2-converted model later (set ASR_MODEL to its path) with zero
# code change — the GPU training is deferred, the serving path is live today.
def get_asr():
    if "m" not in _asr:
        from faster_whisper import WhisperModel  # lazy: CI/tests don't load it
        # cpu_threads set explicitly so it is unaffected by OMP_NUM_THREADS=1
        # (that env var exists only to keep the Kokoro/torch vocoder clean).
        _asr["m"] = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8",
                                 cpu_threads=2)
    return _asr["m"]


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = None):
    import tempfile

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        # language=None lets Whisper auto-detect; pass "zh"/"en" to force.
        segments, info = get_asr().transcribe(tmp.name, language=language)
        text = "".join(seg.text for seg in segments).strip()
    return {"text": text, "language": info.language,
            "language_probability": round(info.language_probability, 3)}


# ---------- Phase 3: avatar pipeline (TTS -> SQS -> GPU worker -> MP4) ----------
class AvatarRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    voice: str = "af_heart"
    lang: str = Field("a", pattern="^[az]$")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    image_key: str | None = None  # portrait object key in the avatar bucket


def _avatar_cfg():
    cfg = {k: os.environ.get(k)
           for k in ("AVATAR_BUCKET", "AVATAR_QUEUE_URL", "AVATAR_TABLE")}
    if not all(cfg.values()):
        raise HTTPException(
            503, "avatar pipeline not deployed (deploy AvatarStack and set AVATAR_* env)")
    return cfg


@app.post("/avatar")
def create_avatar(req: AvatarRequest):
    cfg = _avatar_cfg()
    import json
    import uuid

    import boto3

    job_id = uuid.uuid4().hex[:12]
    wav = _tts_wav(req.text, req.voice, req.lang, req.speed)
    audio_key = f"jobs/{job_id}/speech.wav"
    boto3.client("s3").put_object(Bucket=cfg["AVATAR_BUCKET"], Key=audio_key, Body=wav)
    image_key = req.image_key or os.environ.get(
        "AVATAR_DEFAULT_IMAGE_KEY", "portraits/default.jpg")
    boto3.resource("dynamodb").Table(cfg["AVATAR_TABLE"]).put_item(
        Item={"job_id": job_id, "status": "queued"})
    boto3.client("sqs").send_message(
        QueueUrl=cfg["AVATAR_QUEUE_URL"],
        MessageBody=json.dumps(
            {"job_id": job_id, "audio_key": audio_key, "image_key": image_key}),
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/avatar/{job_id}")
def avatar_status(job_id: str):
    cfg = _avatar_cfg()
    import boto3

    item = boto3.resource("dynamodb").Table(cfg["AVATAR_TABLE"]).get_item(
        Key={"job_id": job_id}).get("Item")
    if not item:
        raise HTTPException(404, "job not found")
    out = {"job_id": job_id, "status": item["status"]}
    if item.get("video_key"):
        out["video_url"] = boto3.client("s3").generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg["AVATAR_BUCKET"], "Key": item["video_key"]},
            ExpiresIn=3600,
        )
    if item.get("error"):
        out["error"] = item["error"]
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kokoro TTS &mdash; AWS Demo</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#127908;</text></svg>">
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
    background:radial-gradient(1200px 800px at 20% -10%,#1e3a5f 0%,#0b1220 45%,#070b14 100%);color:#e8eef7}
  .card{width:min(640px,92vw);background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
    border-radius:20px;padding:32px 28px;backdrop-filter:blur(12px);box-shadow:0 24px 60px rgba(0,0,0,.45)}
  h1{font-size:22px;margin:0 0 6px;letter-spacing:.3px}
  .sub{font-size:13px;color:#8da3bf;margin:0 0 22px}
  textarea{width:100%;height:110px;resize:vertical;padding:14px;border-radius:12px;
    border:1px solid rgba(255,255,255,.12);background:rgba(0,0,0,.25);color:#e8eef7;
    font-size:15px;line-height:1.5;outline:none;font-family:inherit}
  textarea:focus{border-color:#5aa2ff;box-shadow:0 0 0 3px rgba(90,162,255,.15)}
  .row{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap}
  select{flex:1;min-width:200px;padding:11px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.12);
    background:rgba(0,0,0,.25);color:#e8eef7;font-size:14px;outline:none}
  button{padding:11px 26px;border:0;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;
    color:#06121f;background:linear-gradient(135deg,#7cc4ff,#5aa2ff);transition:transform .08s,filter .15s}
  button:hover{filter:brightness(1.1)}
  button:active{transform:scale(.97)}
  button:disabled{opacity:.55;cursor:wait}
  .status{margin-top:14px;font-size:13px;color:#8da3bf;min-height:18px}
  .metrics{display:none;gap:18px;margin-top:6px;font-size:12px;color:#9fd0a9}
  audio{width:100%;margin-top:14px;display:none}
  .foot{margin-top:20px;font-size:11px;color:#62758f;text-align:center}
</style></head><body>
<div class="card">
  <h1>Kokoro TTS</h1>
  <p class="sub">82M open-weight TTS &middot; ECS Fargate (Graviton) &middot; AWS CDK &middot; GitHub Actions OIDC</p>
  <textarea id="t">Hello! This speech is generated by Kokoro running on AWS Fargate, deployed with CDK.</textarea>
  <div class="row">
    <select id="v">
      <option value="af_heart">af_heart &middot; 英文女声</option>
      <option value="am_adam">am_adam &middot; 英文男声</option>
      <option value="zf_xiaobei">zf_xiaobei &middot; 中文女声</option>
      <option value="zm_yunjian">zm_yunjian &middot; 中文男声</option>
    </select>
    <button id="go" onclick="go()">合成 Synthesize</button>
  </div>
  <div class="status" id="s"></div>
  <div class="metrics" id="m"><span id="m1"></span><span id="m2"></span><span id="m3"></span></div>
  <audio id="a" controls></audio>
  <hr style="border:0;border-top:1px solid rgba(255,255,255,.08);margin:24px 0">
  <h1 style="font-size:18px">语音转写 Transcribe <span style="font-size:12px;color:#8da3bf">(base Whisper, CPU)</span></h1>
  <div class="row">
    <input id="f" type="file" accept="audio/*"
      style="flex:1;min-width:200px;color:#8da3bf;font-size:13px">
    <button id="tx" onclick="tx()">转写 Transcribe</button>
  </div>
  <div class="status" id="ts"></div>
  <div id="tr" style="margin-top:8px;font-size:15px;line-height:1.5"></div>
  <div class="foot">text &rarr; mel &rarr; waveform &middot; 24 kHz &nbsp;|&nbsp; speech &rarr; text</div>
</div>
<script>
async function go(){
  const btn=document.getElementById("go"), s=document.getElementById("s"),
        a=document.getElementById("a"), m=document.getElementById("m");
  const voice=document.getElementById("v").value;
  btn.disabled=true; s.textContent="合成中…"; m.style.display="none";
  const t0=performance.now();
  try{
    const r=await fetch("/synthesize",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:document.getElementById("t").value,voice:voice,lang:voice[0]==="z"?"z":"a"})});
    if(!r.ok){s.textContent="出错了：HTTP "+r.status;return;}
    const blob=await r.blob(), ms=performance.now()-t0;
    a.src=URL.createObjectURL(blob); a.style.display="block";
    a.onloadedmetadata=()=>{
      const dur=a.duration, rtf=(ms/1000/dur).toFixed(2);
      document.getElementById("m1").textContent="合成 "+Math.round(ms)+" ms";
      document.getElementById("m2").textContent="音频 "+dur.toFixed(1)+" s";
      document.getElementById("m3").textContent="RTF "+rtf;
      m.style.display="flex"; s.textContent="";
    };
    a.play();
  }catch(e){s.textContent="请求失败："+e.message;}
  finally{btn.disabled=false;}
}
async function tx(){
  const btn=document.getElementById("tx"), ts=document.getElementById("ts"),
        tr=document.getElementById("tr"), f=document.getElementById("f");
  if(!f.files.length){ts.textContent="先选一个音频文件";return;}
  btn.disabled=true; ts.textContent="转写中…"; tr.textContent="";
  const fd=new FormData(); fd.append("file", f.files[0]);
  const t0=performance.now();
  try{
    const r=await fetch("/transcribe",{method:"POST",body:fd});
    if(!r.ok){ts.textContent="出错了：HTTP "+r.status;return;}
    const j=await r.json(), ms=Math.round(performance.now()-t0);
    ts.textContent="检测语言 "+j.language+" ("+j.language_probability+") · "+ms+" ms";
    tr.textContent=j.text;
  }catch(e){ts.textContent="请求失败："+e.message;}
  finally{btn.disabled=false;}
}
</script></body></html>"""
