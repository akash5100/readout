"""Exercise the HTTP stream's first-audio and completion behavior."""

import base64
import io
import json
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tts import server  # noqa: E402
from tts.speech_chunking import SpeechChunk  # noqa: E402


def test_stream_yields_audio_before_second_chunk_and_finishes(monkeypatch):
  allow_second = threading.Event()

  class FakeManager:
    lock = threading.RLock()
    model_id = "kyutai/pocket-tts"
    _trim_silence = staticmethod(server.ModelManager._trim_silence)

    def ensure_loaded(self):
      pass

    def prepare_chunks_for_model(self, text):
      return [SpeechChunk("First.", 250), SpeechChunk("Second.", 200)]

    def synthesize(self, text, **kwargs):
      if text == "Second.":
        assert allow_second.wait(5), "Second chunk must not finish before first is received"
      return np.full(2400, 0.1, dtype=np.float32), 24000

  monkeypatch.setattr(server, "manager", FakeManager())
  http = server.TTSHTTPServer(("127.0.0.1", 0), server.TTSRequestHandler)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  request = urllib.request.Request(
    f"http://127.0.0.1:{http.server_port}/tts/stream",
    data=json.dumps({"text": "First.\nSecond.", "model": "pocket"}).encode(),
    headers={"Content-Type": "application/json"},
  )
  try:
    with urllib.request.urlopen(request, timeout=10) as response:
      first = json.loads(response.readline())
      assert first["type"] == "audio"
      audio, sample_rate = sf.read(io.BytesIO(base64.b64decode(first["wav"])))
      assert sample_rate == 24000 and len(audio) > 0
      allow_second.set()
      assert json.loads(response.readline())["type"] == "audio"
      assert json.loads(response.readline())["type"] == "end"
  finally:
    allow_second.set()
    http.shutdown()
    http.server_close()
