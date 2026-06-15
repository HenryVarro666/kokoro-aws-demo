"""SQS-driven MuseTalk worker. Runs on the GPU instance (golden AMI).

Loop: poll SQS -> download audio + portrait from S3 -> run MuseTalk ->
upload MP4 -> update DynamoDB -> delete message. Scales itself to zero
(via ASG desired capacity) after IDLE_EXIT_MIN minutes without work.

Env (written by CDK user-data to /etc/avatar-worker.env):
  AVATAR_QUEUE_URL, AVATAR_BUCKET, AVATAR_TABLE, ASG_NAME (optional),
  MUSETALK_DIR (default /opt/MuseTalk), MUSETALK_CMD (optional override),
  IDLE_EXIT_MIN (default 10), JOB_TIMEOUT_SEC (default 600)
"""
import glob
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import time

import boto3

QUEUE_URL = os.environ["AVATAR_QUEUE_URL"]
BUCKET = os.environ["AVATAR_BUCKET"]
TABLE = os.environ["AVATAR_TABLE"]
ASG_NAME = os.environ.get("ASG_NAME")
MUSETALK_DIR = os.environ.get("MUSETALK_DIR", "/opt/MuseTalk")
IDLE_EXIT_MIN = int(os.environ.get("IDLE_EXIT_MIN", "10"))
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "600"))
WORK = pathlib.Path("/tmp/avatar-jobs")

# Flags follow the MuseTalk README; adjust here (or via env) if upstream renames them.
DEFAULT_CMD = (
    "python -m scripts.inference --inference_config {config} "
    "--result_dir {result_dir} --unet_model_path ./models/musetalkV15/unet.pth "
    "--unet_config ./models/musetalkV15/musetalk.json --version v15"
)

sqs = boto3.client("sqs")
s3 = boto3.client("s3")
table = boto3.resource("dynamodb").Table(TABLE)


def set_status(job_id: str, status: str, **extra):
    expr = "SET #s = :s"
    names = {"#s": "status"}
    vals = {":s": status}
    for i, (k, v) in enumerate(extra.items()):
        expr += f", #k{i} = :v{i}"
        names[f"#k{i}"] = k
        vals[f":v{i}"] = v
    table.update_item(Key={"job_id": job_id}, UpdateExpression=expr,
                      ExpressionAttributeNames=names,
                      ExpressionAttributeValues=vals)


def run_musetalk(audio_path: pathlib.Path, image_path: pathlib.Path,
                 job_dir: pathlib.Path) -> pathlib.Path:
    config = job_dir / "job.yaml"
    result_dir = job_dir / "results"
    result_dir.mkdir(exist_ok=True)
    config.write_text(
        f'task_0:\n  video_path: "{image_path}"\n  audio_path: "{audio_path}"\n'
    )
    cmd = os.environ.get("MUSETALK_CMD", DEFAULT_CMD).format(
        config=config, result_dir=result_dir)
    subprocess.run(shlex.split(cmd), cwd=MUSETALK_DIR, check=True,
                   timeout=JOB_TIMEOUT_SEC)
    videos = sorted(glob.glob(str(result_dir / "**" / "*.mp4"), recursive=True),
                    key=os.path.getmtime)
    if not videos:
        raise RuntimeError("MuseTalk produced no mp4")
    return pathlib.Path(videos[-1])


def handle(msg: dict):
    body = json.loads(msg["Body"])
    job_id, audio_key, image_key = body["job_id"], body["audio_key"], body["image_key"]
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        set_status(job_id, "processing")
        audio_path = job_dir / "speech.wav"
        image_path = job_dir / pathlib.Path(image_key).name
        s3.download_file(BUCKET, audio_key, str(audio_path))
        s3.download_file(BUCKET, image_key, str(image_path))

        video = run_musetalk(audio_path, image_path, job_dir)

        video_key = f"jobs/{job_id}/avatar.mp4"
        s3.upload_file(str(video), BUCKET, video_key,
                       ExtraArgs={"ContentType": "video/mp4"})
        set_status(job_id, "done", video_key=video_key)
        print(f"[ok] {job_id} -> s3://{BUCKET}/{video_key}", flush=True)
    except Exception as e:  # mark failed; SQS redrive handles retries
        set_status(job_id, "failed", error=str(e)[:500])
        print(f"[fail] {job_id}: {e}", flush=True)
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def scale_self_to_zero():
    print(f"idle {IDLE_EXIT_MIN} min -> scaling {ASG_NAME} to 0", flush=True)
    boto3.client("autoscaling").set_desired_capacity(
        AutoScalingGroupName=ASG_NAME, DesiredCapacity=0)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    idle_since = time.time()
    print(f"worker up, polling {QUEUE_URL}", flush=True)
    while True:
        resp = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=1,
                                   WaitTimeSeconds=20)
        msgs = resp.get("Messages", [])
        if not msgs:
            if ASG_NAME and time.time() - idle_since > IDLE_EXIT_MIN * 60:
                scale_self_to_zero()
                idle_since = time.time()  # instance terminates shortly after
            continue
        msg = msgs[0]
        try:
            handle(msg)
            sqs.delete_message(QueueUrl=QUEUE_URL,
                               ReceiptHandle=msg["ReceiptHandle"])
        except Exception as e:
            # Leave the message undeleted; visibility timeout + DLQ govern
            # retries. Log it — don't swallow silently, or failures are invisible.
            print(f"[loop-error] {e}", flush=True)
        idle_since = time.time()


if __name__ == "__main__":
    main()
