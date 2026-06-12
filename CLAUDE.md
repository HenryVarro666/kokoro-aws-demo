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
- **OMP_NUM_THREADS=1 on Fargate is load-bearing**: multi-threaded oneDNN/ACL kernels on
  Graviton audibly corrupt the vocoder output (-4 dB, 6.7 dB spectral distance). Verified
  by bisection. Do not "optimize" it away without re-running the spectral A/B
  (golden ref vs cloud output, threshold < 2 dB).
- torch is version-pinned in the Dockerfile for the same reason.
- Bilingual training requires per-sample language tokens: tokenize each FLEURS config
  with `set_prefix_tokens(language=...)` before concatenating (see train.py).
- OidcStack and CDKToolkit are foundation stacks — never destroy them.

## Layout

- `app/` — FastAPI + Kokoro TTS service (+ dormant `/avatar` endpoints, 503 until
  AvatarStack is deployed)
- `infra/` — CDK (Python): InfraStack (Fargate/ALB/alarms), OidcStack (CI trust),
  AvatarStack (Phase 3, context-gated)
- `training/` — Whisper LoRA: train.py / evaluate_cer.py (CER for zh, WER for en) /
  merge_and_convert.sh (LoRA merge -> CTranslate2 int8) / launch_sagemaker.py
- `avatar_worker/` — SQS-driven MuseTalk worker for the Phase 3 GPU ASG
