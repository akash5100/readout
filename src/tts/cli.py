"""Speak text through the local readout server."""

from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFERENCES_DOMAIN = "AuraTTS"


def read_preference(name: str, default: str) -> str:
  try:
    result = subprocess.run(
      ["defaults", "read", PREFERENCES_DOMAIN, name], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip() or default
  except (OSError, subprocess.CalledProcessError):
    return default


def server_is_ready(base_url: str) -> bool:
  try:
    with urllib.request.urlopen(base_url + "/", timeout=1) as response:
      return response.status == 200
  except (OSError, ValueError):
    return False


def ensure_server(base_url: str, host: str, port: int) -> None:
  def is_running() -> bool:
    if server_is_ready(base_url):
      return True
    # A busy server may hold the model lock while streaming, making its
    # health response slow. Do not start a second process in that case.
    try:
      with socket.create_connection((host, port), timeout=1):
        return True
    except OSError:
      return False

  if is_running():
    return
  if host not in ("127.0.0.1", "localhost", "::1"):
    raise RuntimeError(f"The configured speech server at {base_url} is offline.")

  # Serialize concurrent CLI starts so they do not launch duplicate servers.
  lock_path = Path(tempfile.gettempdir()) / "tts-ext-server-start.lock"
  with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if is_running():
      return
    log_path = Path(tempfile.gettempdir()) / "tts_server.log"
    with log_path.open("ab") as log:
      subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "src/tts/server.py"), "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
      )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
      if server_is_ready(base_url):
        return
      time.sleep(0.2)
  raise RuntimeError(f"The speech server did not start. Check {log_path}.")


def speak(text: str) -> float:
  import sounddevice as sd

  text = text.strip()
  if not text:
    raise ValueError("Provide some text to speak.")
  model = read_preference("selectedModelId", "pocket")
  voice = read_preference("selectedVoiceName", "alba")
  host = read_preference("serverHost", "127.0.0.1")
  port = int(read_preference("serverPort", "5050"))
  address_host = f"[{host}]" if ":" in host else host
  base_url = f"http://{address_host}:{port}"
  ensure_server(base_url, host, port)

  body = {"text": text, "model": model, "voice": voice}
  # A voice imported by the app may have only a precomputed 100M cache.
  pocket_cache = Path.home() / ".cache/tts-ext/voices" / f"pocket__{voice}.safetensors"
  if model == "pocket" and pocket_cache.is_file():
    body["ref_audio"] = str(pocket_cache)
  request = urllib.request.Request(
    base_url + "/tts/stream", data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST",
  )

  output = None
  received_end = False
  duration = 0.0
  try:
    with urllib.request.urlopen(request, timeout=60) as response:
      for line in response:
        frame = json.loads(line)
        if frame.get("type") == "error":
          raise RuntimeError(frame.get("message", "Speech generation failed."))
        if frame.get("type") == "end":
          received_end = True
          break
        if frame.get("type") != "audio" or not isinstance(frame.get("wav"), str):
          raise RuntimeError("The speech server returned an invalid audio frame.")
        wav = base64.b64decode(frame["wav"], validate=True)
        samples, sample_rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        if output is None:
          output = sd.OutputStream(samplerate=sample_rate, channels=samples.shape[1], dtype="float32")
          output.start()
        elif output.samplerate != sample_rate or output.channels != samples.shape[1]:
          raise RuntimeError("The speech server changed audio format mid-stream.")
        output.write(samples)
        duration += len(samples) / sample_rate
    if not received_end or output is None:
      raise RuntimeError("Speech generation ended before the audio was complete.")
  finally:
    if output is not None:
      output.stop()
      output.close()
  return duration


# Job files contain status and timing only; speech text is passed over a pipe.
JOB_DIR = Path(tempfile.gettempdir()) / f"readout-jobs-{os.getuid()}"


def prepare_jobs() -> None:
  JOB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
  if JOB_DIR.is_symlink() or JOB_DIR.stat().st_uid != os.getuid():
    raise RuntimeError("Speech job directory is not owned by the current user.")
  os.chmod(JOB_DIR, 0o700)


@contextmanager
def queue_lock():
  prepare_jobs()
  with (JOB_DIR / "queue.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


def job_path(job_id: str) -> Path:
  if not re.fullmatch(r"[0-9a-f]{32}", job_id):
    raise ValueError("Invalid speech job ID.")
  return JOB_DIR / f"{job_id}.json"


def write_job(job_id: str, job: dict) -> None:
  path = job_path(job_id)
  temporary = path.with_suffix(f".{os.getpid()}.tmp")
  temporary.write_text(json.dumps(job), encoding="utf-8")
  os.chmod(temporary, 0o600)
  os.replace(temporary, path)


def read_job(job_id: str) -> dict:
  try:
    return json.loads(job_path(job_id).read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise ValueError("Speech job not found.") from exc


def read_queue() -> list[str]:
  path = JOB_DIR / "queue.json"
  return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_queue(queue: list[str]) -> None:
  path = JOB_DIR / "queue.json"
  temporary = path.with_suffix(f".{os.getpid()}.tmp")
  temporary.write_text(json.dumps(queue), encoding="utf-8")
  os.chmod(temporary, 0o600)
  os.replace(temporary, path)


def process_exists(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


def remove_dead_head(queue: list[str]) -> list[str]:
  while queue:
    try:
      head = read_job(queue[0])
    except ValueError:
      queue.pop(0)
      continue
    if process_exists(head["pid"]):
      break
    head.update(status="failed", error="Speech worker stopped unexpectedly.")
    write_job(queue.pop(0), head)
  return queue


def enqueue(text: str) -> dict:
  text = text.strip()
  if not text:
    raise ValueError("Provide some text to speak.")
  job_id = uuid.uuid4().hex
  # Approximate spoken duration, not generation or queue time.
  estimate = round(max(0.4, len(text.split()) / 2.5), 1)
  with queue_lock():
    queue = remove_dead_head(read_queue())
    worker = subprocess.Popen(
      [sys.executable, "-m", "tts.cli", "--_worker", job_id],
      stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
    job = {"job_id": job_id, "status": "queued", "estimated_seconds": estimate,
           "queued_ahead": len(queue), "pid": worker.pid}
    write_job(job_id, job)
    queue.append(job_id)
    write_queue(queue)
  try:
    assert worker.stdin is not None
    worker.stdin.write(text.encode("utf-8"))
    worker.stdin.close()
  except (BrokenPipeError, OSError) as exc:
    raise RuntimeError(f"Could not start speech worker: {exc}") from exc
  return {key: job[key] for key in ("job_id", "status", "estimated_seconds", "queued_ahead")}


def run_worker(job_id: str, text: str) -> int:
  try:
    while True:
      with queue_lock():
        queue = remove_dead_head(read_queue())
        write_queue(queue)
        if not queue or job_id not in queue:
          raise RuntimeError("Speech job disappeared from the queue.")
        if queue[0] == job_id:
          job = read_job(job_id)
          job["status"] = "playing"
          write_job(job_id, job)
          break
      time.sleep(0.1)
    duration = speak(text)
    job.update(status="completed", duration_seconds=round(duration, 2))
    return_code = 0
  except Exception as exc:
    job = read_job(job_id)
    job.update(status="failed", error=str(exc))
    return_code = 1
  with queue_lock():
    write_job(job_id, job)
    write_queue([item for item in read_queue() if item != job_id])
  return return_code


def get_job_status(job_id: str) -> dict:
  with queue_lock():
    queue = remove_dead_head(read_queue())
    write_queue(queue)
    job = read_job(job_id)
  return {key: value for key, value in job.items() if key != "pid"}


def main() -> int:
  parser = argparse.ArgumentParser(description="Speak text using readout.")
  parser.add_argument("--wait", action="store_true", help="Wait until playback finishes")
  parser.add_argument("--status", metavar="JOB_ID", help="Check a background speech job")
  parser.add_argument("--_worker", metavar="JOB_ID", help=argparse.SUPPRESS)
  parser.add_argument("text", nargs="*", help="Text to speak")
  args = parser.parse_args()
  try:
    if args._worker:
      return run_worker(args._worker, sys.stdin.read())
    if args.status:
      if args.text or args.wait:
        parser.error("--status cannot be combined with text or --wait")
      print(json.dumps(get_job_status(args.status)))
    elif args.wait:
      # --wait uses the same queue, so it cannot overlap background requests.
      job = enqueue(" ".join(args.text))
      while True:
        result = get_job_status(job["job_id"])
        if result["status"] == "completed":
          print(json.dumps(result))
          return 0
        if result["status"] == "failed":
          raise RuntimeError(result.get("error", "Speech failed."))
        time.sleep(0.1)
    else:
      print(json.dumps(enqueue(" ".join(args.text))))
    return 0
  except KeyboardInterrupt:
    return 130
  except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
    print(f"tts: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
