"""Tests for the command-line streaming speech client."""

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sounddevice
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tts import cli  # noqa: E402


class FakeOutput:
  def __init__(self, samplerate, channels, dtype):
    self.samplerate = samplerate
    self.channels = channels
    self.started = False
    self.closed = False
    self.writes = []

  def start(self):
    self.started = True

  def write(self, samples):
    self.writes.append(samples)

  def stop(self):
    pass

  def close(self):
    self.closed = True


def make_wav_frame():
  buffer = io.BytesIO()
  sf.write(buffer, np.full(240, 0.05, dtype=np.float32), 24000, format="WAV")
  return json.dumps({"type": "audio", "wav": base64.b64encode(buffer.getvalue()).decode()}).encode() + b"\n"


def test_speak_uses_selected_model_voice_and_streams_audio(monkeypatch):
  preferences = {"selectedModelId": "pocket-18m", "selectedVoiceName": "custom"}
  monkeypatch.setattr(cli, "read_preference", lambda key, default: preferences.get(key, default))
  monkeypatch.setattr(cli, "ensure_server", lambda *args: None)
  captured = {}

  def open_request(request, timeout):
    captured["body"] = json.loads(request.data)
    captured["url"] = request.full_url
    return io.BytesIO(make_wav_frame() + make_wav_frame() + b'{"type":"end"}\n')

  monkeypatch.setattr(cli.urllib.request, "urlopen", open_request)

  def make_output(**kwargs):
    captured["output"] = FakeOutput(**kwargs)
    return captured["output"]

  monkeypatch.setattr(sounddevice, "OutputStream", make_output)
  cli.speak('Hello "Sample", from snake_case.')
  assert captured["url"].endswith("/tts/stream")
  assert captured["body"] == {"text": 'Hello "Sample", from snake_case.', "model": "pocket-18m", "voice": "custom"}
  assert captured["output"].started and captured["output"].closed
  assert len(captured["output"].writes) == 2


def test_speak_reports_incomplete_stream_and_closes_audio(monkeypatch):
  monkeypatch.setattr(cli, "read_preference", lambda key, default: default)
  monkeypatch.setattr(cli, "ensure_server", lambda *args: None)
  monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(make_wav_frame()))
  output = FakeOutput(24000, 1, "float32")
  monkeypatch.setattr(sounddevice, "OutputStream", lambda **kwargs: output)
  with pytest.raises(RuntimeError, match="before the audio was complete"):
    cli.speak("Hello")
  assert output.closed


def test_launcher_works_through_symlink_from_another_directory(tmp_path):
  launcher = Path(__file__).resolve().parent.parent / "tts"
  command = tmp_path / "tts"
  command.symlink_to(launcher)
  result = subprocess.run([str(command), "speak", "--help"], cwd=tmp_path, capture_output=True, text=True, check=True)
  assert "Speak text using readout" in result.stdout


def test_speech_jobs_play_in_queue_order_without_overlap(tmp_path, monkeypatch):
  import os
  import threading
  import time

  monkeypatch.setattr(cli, "JOB_DIR", tmp_path / "jobs")
  first, second = "a" * 32, "b" * 32
  order = []
  first_started = threading.Event()
  release_first = threading.Event()

  def fake_speak(text):
    order.append("start " + text)
    if text == "first":
      first_started.set()
      assert release_first.wait(3)
    order.append("end " + text)
    return 1.25

  monkeypatch.setattr(cli, "speak", fake_speak)
  with cli.queue_lock():
    for job_id in (first, second):
      cli.write_job(job_id, {"job_id": job_id, "status": "queued", "pid": os.getpid()})
    cli.write_queue([first, second])

  t1 = threading.Thread(target=cli.run_worker, args=(first, "first"))
  t2 = threading.Thread(target=cli.run_worker, args=(second, "second"))
  t2.start()
  t1.start()
  assert first_started.wait(2)
  time.sleep(0.2)
  assert order == ["start first"]
  release_first.set()
  t1.join(3)
  t2.join(3)
  assert not t1.is_alive() and not t2.is_alive()
  assert order == ["start first", "end first", "start second", "end second"]
  assert cli.get_job_status(second)["duration_seconds"] == 1.25
  assert cli.get_job_status(second)["status"] == "completed"


def test_failed_speech_job_unblocks_next(tmp_path, monkeypatch):
  import os

  monkeypatch.setattr(cli, "JOB_DIR", tmp_path / "jobs")
  first = "c" * 32
  with cli.queue_lock():
    cli.write_job(first, {"job_id": first, "status": "queued", "pid": os.getpid()})
    cli.write_queue([first])
  monkeypatch.setattr(cli, "speak", lambda text: (_ for _ in ()).throw(RuntimeError("audio failed")))
  assert cli.run_worker(first, "hello") == 1
  assert cli.get_job_status(first)["error"] == "audio failed"
  assert cli.read_queue() == []
