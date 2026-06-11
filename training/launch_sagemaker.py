"""Launch train.py as a SageMaker Training Job (managed GPU, auto-teardown).

Requires: pip install sagemaker, an execution role, and ml.g5.xlarge spot quota.
"""
import sagemaker
from sagemaker.huggingface import HuggingFace

ROLE = "arn:aws:iam::CHANGE_ME:role/CHANGE_ME-SageMakerExecutionRole"
BUCKET = "CHANGE_ME"  # e.g. yourname-whisper-finetune

session = sagemaker.Session()

estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",  # requirements.txt here is auto-installed in the container
    role=ROLE,
    instance_type="ml.g5.xlarge",
    instance_count=1,
    # Pin to a combination from the official HF DLC support matrix.
    transformers_version="4.49",
    pytorch_version="2.5",
    py_version="py311",
    use_spot_instances=True,
    max_run=4 * 3600,
    max_wait=6 * 3600,
    checkpoint_s3_uri=f"s3://{BUCKET}/whisper-lora/checkpoints/",
    hyperparameters={
        "use_8bit": "",
        "batch_size": 8,
        "train_samples": 2000,
        "epochs": 3,
    },
)
estimator.fit()
print("model artifact:", estimator.model_data)
