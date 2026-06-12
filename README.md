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

- **Serving**: FastAPI + Kokoro-82M, Docker, ECS Fargate (ARM64/Graviton), ALB
- **IaC**: AWS CDK (Python) — VPC (no NAT), cluster, service, health checks
- **CI/CD**: GitHub Actions with OIDC role federation (zero long-lived secrets)
- **Training**: Whisper-large-v3 + LoRA (PEFT, 8-bit base) on g5.xlarge Spot / SageMaker
- **Inference optimization**: LoRA merge → CTranslate2 int8 → faster-whisper on CPU

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

## Before first deploy — replace CHANGE_ME

| File | What | Status |
|------|------|--------|
| `infra/infra/oidc_stack.py` | `GITHUB_REPO` — your `user/repo` | done |
| `.github/workflows/deploy.yml` | `ROLE_ARN` — account ID (from `cdk deploy OidcStack` output) | done |
| `training/merge_and_convert.sh` | `BUCKET` — your S3 bucket | TODO (Phase 2) |
| `training/launch_sagemaker.py` | `ROLE`, `BUCKET` | TODO (Phase 2) |

## Cost

~$45/mo always-on (Fargate $28 + ALB $16). `cdk destroy` → $0.
Training runs: $2–7 each on g5.xlarge Spot.

## Roadmap

- [x] Phase 1 — TTS service: Fargate + CDK + OIDC CI/CD
- [ ] Phase 2 — `/transcribe` endpoint backed by LoRA-fine-tuned Whisper (CT2 int8, CPU)
- [ ] Phase 3 — talking-head avatar (MuseTalk): `POST /avatar` → SQS → scale-to-zero
      GPU worker (g4dn Spot ASG, golden AMI) → MP4 via presigned URL.
      Code is in place (`avatar_worker/`, `infra/infra/avatar_stack.py`); deploy with
      `cdk deploy --context avatar=true --context avatar_ami=ami-XXX --all`
