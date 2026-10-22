# Problem and acceptance

> **Can an AWS speech service preserve reference audio and transcription quality across deployment changes while meeting a measured latency and operating-cost budget?**

**Why it matters:** A healthy endpoint can still return degraded speech. Deployable speech systems need quality checks alongside availability, infrastructure reproducibility and cost control.

**Central tension:** CPU serving, model conversion and asynchronous GPU jobs can reduce operating cost, but introduce backend differences, cold starts and failure paths. A local model test does not establish cloud audio equivalence or concurrent service capacity.

**Current evidence and gap:** [README.md](README.md) distinguishes serving, the fine-tuning track and the avatar track. [CLAUDE.md](CLAUDE.md) records the backend-specific audio regression and staged deployment boundaries. Those records do not certify every later image or optional phase.

**Acceptance test:** Replay a fixed bilingual corpus through a pinned local reference and the deployed candidate. Record transcription error, audio regression checks, failures, cold/warm latency distributions and resource use under stated concurrency. Exercise model load failure and asynchronous job retries. Promote a change only when its quality and operational checks pass; report unavailable phases separately.

The comparison is prospective. Existing service status and historical costs must be checked before reuse.

中文版：[PROBLEM.zh.md](PROBLEM.zh.md)
