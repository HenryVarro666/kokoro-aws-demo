# kokoro-aws-demo

Open-source TTS (Kokoro-82M) served as a production-style API on AWS — plus a GPU
fine-tuning track (Whisper-large-v3 + LoRA) that feeds a CPU-served ASR endpoint.

```
git push ──► GitHub Actions (OIDC, no stored AWS keys)
               └─► cdk deploy
                     ├─► ECR  (Docker image, model weights baked in)
                     └─► ECS Fargate (Graviton ARM64, 1 vCPU / 2 GB)
                           └─► ALB ──► browser demo  (type text → hear speech)

GPU track:  EC2 g5 Spot / SageMaker ──► LoRA adapter ──► merge + CTranslate2 int8
            (train on GPU)                               (serve on CPU Fargate)
```

## Stack

- **Serving**: FastAPI on ECS Fargate (ARM64/Graviton) behind an ALB
  - `POST /synthesize` — Kokoro-82M TTS (text → speech)
  - `POST /transcribe` — faster-whisper **base** ASR on CPU (speech → text), no GPU needed
- **IaC**: AWS CDK (Python) — VPC (no NAT), cluster, service, health checks, CloudWatch alarms
- **CI/CD**: GitHub Actions with OIDC role federation (zero long-lived secrets)
- **Training (deferred)**: Whisper-large-v3 + LoRA (PEFT, 8-bit) on g5.xlarge Spot / SageMaker.
  The serving path above runs the base model today; a fine-tuned, CTranslate2-int8 model
  drops in later via the `ASR_MODEL` env var with zero code change.

## Quickstart (local)

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # kokoro requires Python >=3.10,<3.13
pip install -r requirements.txt                          # torch comes in as a kokoro dependency
uvicorn app.main:app --reload    # open http://127.0.0.1:8000
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Deploy

```bash
cd infra
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cdk bootstrap        # once per account+region
cdk deploy InfraStack
# tear down everything: cdk destroy InfraStack
```

CI/CD: merges to `main` auto-deploy via GitHub Actions OIDC. Include
`[skip deploy]` in the commit message to land changes while the stack stays
destroyed; trigger manual deploys from the Actions tab (`workflow_dispatch`).

## Before first deploy — replace CHANGE_ME

| File | What | Status |
|------|------|--------|
| `infra/infra/oidc_stack.py` | `GITHUB_REPO` — your `user/repo` | done |
| `.github/workflows/deploy.yml` | `ROLE_ARN` — account ID (from `cdk deploy OidcStack` output) | done |
| `training/merge_and_convert.sh` | `BUCKET` — your S3 bucket | TODO (Phase 2) |
| `training/launch_sagemaker.py` | `ROLE`, `BUCKET` | TODO (Phase 2) |

## Measured performance

Same ~7-second sentence, warm service:

| Environment | RTF (synthesis time ÷ audio duration) |
|---|---|
| Apple Silicon dev box (CPU) | 0.11 |
| Fargate 1 vCPU Graviton | 2.4 |
| Fargate 2 vCPU, default math | ~~0.82~~ — retracted: fast-math kernel path audibly degraded the vocoder output |
| **Fargate 2 vCPU, fp32 + single thread** | **1.48 — audio verified identical to local** (spectral distance 1.67 dB ≈ run-to-run noise floor) |

**War story**: on Graviton, PyTorch's multi-threaded oneDNN/ACL kernel path
audibly corrupts vocoder output (-4 dB level, 6.7 dB spectral distance vs
identical-image local runs). Diagnosed via spectral A/B against a
synthesis-randomness baseline and same-image bisection; a second bisection
(F32 pinned, threads released) proved the parallel kernel path itself — not
fast-math — is the culprit. Fix: `OMP_NUM_THREADS=1`, deployed through the
pipeline. Moral: speed numbers mean nothing without a quality regression
check.

Load test (`hey`, single 2 vCPU task): c=1 → p50 6.8 s; c=4 → p50 22 s,
16/16 HTTP 200, throughput flat — textbook single-task CPU saturation.
Scale out with `desired_count`, not bigger tasks.

Observability: CloudWatch alarms (ALB 5xx, CPU > 85%) defined in CDK,
notifying via SNS email.

## Cost

~$74/mo always-on at 2 vCPU (Fargate $57 + ALB $16). `cdk destroy` → $0.
Training runs: $2–7 each on g5.xlarge Spot.

## Roadmap

- [x] Phase 1 — TTS service: Fargate + CDK + OIDC CI/CD
- [ ] Phase 2 — `/transcribe` endpoint backed by LoRA-fine-tuned Whisper (CT2 int8, CPU)
- [ ] Phase 3 — talking-head avatar (MuseTalk): `POST /avatar` → SQS → scale-to-zero
      GPU worker (g4dn Spot ASG, golden AMI) → MP4 via presigned URL.
      Code is in place (`avatar_worker/`, `infra/infra/avatar_stack.py`); deploy with
      `cdk deploy --context avatar=true --context avatar_ami=ami-XXX --all`
