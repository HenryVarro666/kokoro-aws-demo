# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A three-phase AWS portfolio project: Kokoro TTS served on Fargate (Phase 1, complete),
Whisper bilingual LoRA fine-tuning (Phase 2, in progress), MuseTalk avatar pipeline
(Phase 3, scaffolded). Chinese step-by-step guides live in a separate private repo
(aws-hands-on-guides).

## Commands

```bash
# App tests (no model/network needed; SKIP_MODEL_LOAD=1 is set inside the tests)
.venv/bin/python -m pytest -q

# Local dev server (venv MUST be python3.12 — kokoro requires <3.13)
source .venv/bin/activate && uvicorn app.main:app --reload

# Deploy / destroy (billing toggle, ~$74/mo at 2 vCPU)
cd infra && source .venv/bin/activate
cdk deploy InfraStack      # power on (URL changes each time)
cdk destroy InfraStack     # power off -> $0
# Phase 3 stacks only materialize with: --context avatar=true --context avatar_ami=ami-XXX

# Training smoke test (Mac, no GPU; bilingual zh+en by default)
cd training && source .venv/bin/activate
python train.py --model_id openai/whisper-tiny --train_samples 32 --epochs 1 --batch_size 4
```

## Critical Constraints

- **Push to main auto-deploys** via GitHub Actions OIDC. If the stack is meant to stay
  destroyed, include `[skip deploy]` in the commit message. Manual deploys: Actions tab
  -> Run workflow.
- **`torch.backends.mkldnn.enabled = False` is load-bearing** (set in `app/main.py`
  `get_pipeline`): torch's oneDNN(ACL) CPU backend silently corrupts the Kokoro vocoder on
  Graviton/aarch64 (-6 dB, ~8 dB spectral distance vs a golden local run), INDEPENDENT of
  thread count on torch 2.12.0. The old `OMP_NUM_THREADS=1` workaround (blaming multi-threaded
  kernels) stopped working after a torch bump — the single-threaded service was still corrupted.
  The `OMP_NUM_THREADS=1` / `DNNL_DEFAULT_FPMATH_MODE=F32` env vars are now inert belt-and-suspenders.
  Verified 2026-07 by golden-referenced spectral A/B on Fargate (fix -> 0.8 dB = noise floor).
  Re-verify against a GOLDEN reference (Apple Silicon), NOT cloud-vs-cloud, before changing.
- torch is version-pinned in the Dockerfile for the same reason.
- Bilingual training requires per-sample language tokens: tokenize each FLEURS config
  with `set_prefix_tokens(language=...)` before concatenating (see train.py).
- OidcStack and CDKToolkit are foundation stacks — never destroy them.

## Layout

- `app/` — FastAPI service: `/synthesize` (Kokoro TTS), `/transcribe` (faster-whisper
  base ASR, CPU — base model now, fine-tuned model swaps in via `ASR_MODEL` env),
  `/avatar` (dormant, 503 until AvatarStack is deployed)
- `infra/` — CDK (Python): InfraStack (Fargate/ALB/alarms), OidcStack (CI trust),
  AvatarStack (Phase 3, context-gated)
- `training/` — Whisper LoRA: train.py / evaluate_cer.py (CER for zh, WER for en) /
  merge_and_convert.sh (LoRA merge -> CTranslate2 int8) / launch_sagemaker.py
- `avatar_worker/` — SQS-driven MuseTalk worker for the Phase 3 GPU ASG
