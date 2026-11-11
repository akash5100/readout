import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tts.server import app, manager, normalize_text_for_speech  # noqa: E402


client = TestClient(app)


def test_status_and_voices():
  resp = client.get("/")
  assert resp.status_code == 200
  data = resp.json()
  assert data["status"] == "ok"
  assert "current_model" in data
  assert "available_voices" in data
  assert any("alba" in v for v in data["available_voices"])
  print("[PASS] test_status_and_voices:", data)

  v_resp = client.get("/voices")
  assert v_resp.status_code == 200
  v_data = v_resp.json()
  assert "voices" in v_data
  assert "alba" in v_data["voices"]
  print("[PASS] test_voices endpoint:", v_data)


def test_supratts_synthesis():
  if manager.type != "supratts":
    resp = client.post("/load_model", json={"repo_id": "supratts"})
    assert resp.status_code == 200

  resp = client.post("/tts", json={"text": "Hello, testing local SupraTTS."})
  assert resp.status_code == 200
  assert resp.headers["content-type"] == "audio/wav"
  audio_data, sr = sf.read(io.BytesIO(resp.content))
  assert sr == 24000
  assert len(audio_data) > 0
  print(f"[PASS] test_supratts_synthesis: {len(audio_data)} samples at {sr}Hz")

  # Test empty text error
  err_resp = client.post("/tts", json={"text": "   "})
  assert err_resp.status_code == 400
  print("[PASS] test_empty_text returns 400")


def test_qwen3_load_and_synthesis(monkeypatch):
  if not (Path(__file__).resolve().parent.parent / "voices/default.wav").exists():
    pytest.skip("Add voices/default.wav to exercise reference voice cloning")
  monkeypatch.setattr(manager, "selected_voice", "default.wav")
  print("[*] Switching to Qwen3 via alias 'qwen3'...")
  load_resp = client.post("/load_model", json={"repo_id": "qwen3"})
  assert load_resp.status_code == 200
  load_data = load_resp.json()
  assert load_data["status"] == "ok"
  assert load_data["model_type"] == "qwen3"
  assert load_data["current_model"] in ("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
  assert manager.type == "qwen3"
  print("[PASS] Loaded Qwen3TTS via alias successfully")

  # 1. Synthesis with auto-resolved default voice (default.wav + default.txt)
  tts_resp1 = client.post(
    "/tts",
    json={"text": "This is a test of voice cloning using a reference voice."},
  )
  assert tts_resp1.status_code == 200
  assert tts_resp1.headers["content-type"] == "audio/wav"
  audio1, sr1 = sf.read(io.BytesIO(tts_resp1.content))
  assert sr1 == 24000
  assert len(audio1) > 0
  print(f"[PASS] Default voice clone synthesis: {len(audio1)} samples ({len(audio1) / sr1:.2f}s)")

  # 2. Synthesis with explicit voice 'default.wav'
  tts_resp2 = client.post(
    "/tts",
    json={
      "text": "Testing with sample reference voice.",
      "voice": "default.wav",
    },
  )
  assert tts_resp2.status_code == 200
  audio2, sr2 = sf.read(io.BytesIO(tts_resp2.content))
  assert sr2 == 24000
  assert len(audio2) > 0
  print(f"[PASS] Explicit voice 'default.wav' synthesis: {len(audio2)} samples ({len(audio2) / sr2:.2f}s)")

  # 3. Synthesis with x_vector_only_mode = True
  tts_resp3 = client.post(
    "/tts",
    json={
      "text": "Testing x-vector speaker embedding mode.",
      "voice": "default",
      "x_vector_only_mode": True,
    },
  )
  assert tts_resp3.status_code == 200
  audio3, sr3 = sf.read(io.BytesIO(tts_resp3.content))
  assert sr3 == 24000
  assert len(audio3) > 0
  print(f"[PASS] x-vector mode synthesis: {len(audio3)} samples ({len(audio3) / sr3:.2f}s)")

  # 4. Requesting nonexistent voice returns 404 (not silent fallback)
  tts_resp_err = client.post(
    "/tts",
    json={
      "text": "Testing missing voice error.",
      "voice": "nonexistent_voice_name",
    },
  )
  assert tts_resp_err.status_code == 404
  print("[PASS] Nonexistent voice returns 404:", tts_resp_err.json())


def test_pocket_tts_load_and_synthesis():
  print("[*] Switching to Pocket TTS via alias 'pocket-tts'...")
  load_resp = client.post("/load_model", json={"repo_id": "pocket-tts"})
  assert load_resp.status_code == 200
  load_data = load_resp.json()
  assert load_data["status"] == "ok"
  assert load_data["model_type"] == "pocket_tts"
  assert load_data["current_model"] == "kyutai/pocket-tts"
  assert manager.type == "pocket_tts"
  print("[PASS] Loaded Pocket TTS via alias successfully")

  # 1. Status check reports pocket_tts
  status_resp = client.get("/")
  assert status_resp.status_code == 200
  assert status_resp.json()["model_type"] == "pocket_tts"

  # 2. Synthesis with auto-resolved default voice (default.wav)
  tts_resp1 = client.post(
    "/tts",
    json={"text": "Testing Pocket TTS generation with default reference voice."},
  )
  assert tts_resp1.status_code == 200
  assert tts_resp1.headers["content-type"] == "audio/wav"
  audio1, sr1 = sf.read(io.BytesIO(tts_resp1.content))
  assert sr1 == 24000
  assert len(audio1) > 0
  print(f"[PASS] Pocket TTS default voice synthesis: {len(audio1)} samples ({len(audio1) / sr1:.2f}s)")

  # 3. Synthesis with predefined voice 'alba'
  tts_resp2 = client.post(
    "/tts",
    json={
      "text": "Testing Pocket TTS with predefined Alba voice.",
      "voice": "alba",
    },
  )
  assert tts_resp2.status_code == 200
  audio2, sr2 = sf.read(io.BytesIO(tts_resp2.content))
  assert sr2 == 24000
  assert len(audio2) > 0
  print(f"[PASS] Pocket TTS predefined 'alba' synthesis: {len(audio2)} samples ({len(audio2) / sr2:.2f}s)")

  if (Path(__file__).resolve().parent.parent / "voices/default.wav").exists():
    # 4. Synthesis with explicit local voice 'default.wav'
    tts_resp3 = client.post(
      "/tts",
      json={
        "text": "Testing Pocket TTS with explicit local reference voice.",
        "voice": "default.wav",
      },
    )
    assert tts_resp3.status_code == 200
    audio3, sr3 = sf.read(io.BytesIO(tts_resp3.content))
    assert sr3 == 24000
    assert len(audio3) > 0
    print(f"[PASS] Pocket TTS explicit 'default.wav' synthesis: {len(audio3)} samples ({len(audio3) / sr3:.2f}s)")

  # 5. Nonexistent voice returns 404
  tts_resp_err = client.post(
    "/tts",
    json={
      "text": "Testing missing voice error on Pocket TTS.",
      "voice": "unknown_voice_xyz",
    },
  )
  assert tts_resp_err.status_code == 404
  print("[PASS] Pocket TTS nonexistent voice returns 404:", tts_resp_err.json())

  # 6. Dynamic switching via 'model' field in /tts request
  # Switch to supratts first
  client.post("/load_model", json={"repo_id": "supratts"})
  assert manager.type == "supratts"
  dyn_resp = client.post(
    "/tts",
    json={"text": "Dynamic switch test to Pocket TTS.", "model": "pocket-tts"},
  )
  assert dyn_resp.status_code == 200
  assert dyn_resp.headers["content-type"] == "audio/wav"
  assert manager.type == "pocket_tts"
  print("[PASS] Dynamic switch to Pocket TTS via /tts 'model' field succeeded")


def test_safe_silence_trim_and_lead_in():
  # 1. Test audio starting immediately with speech at sample 0
  sr = 24000
  tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, False)).astype(np.float32) * 0.5
  trimmed = manager._trim_silence(tone, sr=sr, thresh_db=-65.0, lead_ms=250.0, trail_ms=180.0)

  peak = np.max(np.abs(trimmed))
  idx = np.where(np.abs(trimmed) > 0.0005 * peak)[0]
  lead_silence_ms = idx[0] / sr * 1000.0
  trail_silence_ms = (len(trimmed) - (idx[-1] + 1)) / sr * 1000.0

  assert lead_silence_ms >= 240.0, f"Expected >= 240ms lead-in padding, got {lead_silence_ms:.1f}ms"
  assert trail_silence_ms >= 170.0, f"Expected >= 170ms trailing padding, got {trail_silence_ms:.1f}ms"
  print(f"[PASS] Guaranteed lead-in padding verified: {lead_silence_ms:.1f}ms lead, {trail_silence_ms:.1f}ms trail")

  # 2. Test audio with 1 second leading silence (must be trimmed down to 250ms)
  long_lead = np.concatenate([np.zeros(sr, dtype=np.float32), tone])
  trimmed_long = manager._trim_silence(long_lead, sr=sr, thresh_db=-65.0, lead_ms=250.0, trail_ms=180.0)
  idx_long = np.where(np.abs(trimmed_long) > 0.0005 * np.max(np.abs(trimmed_long)))[0]
  lead_long_ms = idx_long[0] / sr * 1000.0
  assert 240.0 <= lead_long_ms <= 260.0, f"Expected ~250ms lead-in, got {lead_long_ms:.1f}ms"
  print(f"[PASS] Excessive leading silence safely normalized to: {lead_long_ms:.1f}ms")


def test_chunking_and_cleanup():
  # 1. Multi-sentence packing for pocket_tts up to 120 words
  s1 = "A lead-in silence of 180ms is implemented. The observed effect is that CoreAudio, responsible for audio output, ramps up during the silence. This silent lead-in allows the audio to begin immediately with perfect clarity, without dropped syllables or phonemes. A 150ms trailing silence is also implemented to ensure the audio finishes cleanly."
  s2 = "The audio output behavior must be examined to prevent clipping at the end of audio streams and at the boundaries between chunks. Verification that padding implemented in the server is sufficient for smooth playback by the application is necessary. If the application handles multiple text chunks, the chunking and buffering strategies will also need to be analyzed."
  s3 = "Chunking configurations across the application and server are to be reviewed, especially related to model-specific parameters."
  full_text = f"{s1}\n\n{s2}\n\n{s3}"

  chunks = manager._chunk_text(full_text, max_words=120, strategy="multi_sentence")
  assert len(chunks) == 2, f"Expected 2 chunks with max_words=120, got {len(chunks)}"
  assert "The audio output behavior" in chunks[0], "Expected 'The audio' paragraph to be packed in chunk 0"
  print("[PASS] Multi-sentence packing kept Paragraph 1 and Paragraph 2 in unified chunk 0 without splitting")

  # 2. Quotation mark boundary splitting
  quote_text = '"Commas (,) are never touched or split on: Every clause stays intact." The model speaks the full sentence.'
  q_chunks = manager._chunk_text(quote_text, max_words=15, strategy="single_sentence")
  assert len(q_chunks) == 2, f"Expected 2 chunks across quote boundary, got {q_chunks}"
  assert q_chunks[0] == '"Commas (,) are never touched or split on: Every clause stays intact."'
  assert q_chunks[1] == "The model speaks the full sentence."
  print("[PASS] Sentence boundary regex correctly splits quote-enclosed sentences")

  # 3. Switch back to supratts to verify cleanup & bidirectional switching
  load_resp = client.post("/load_model", json={"repo_id": "supratts"})
  assert load_resp.status_code == 200
  assert manager.type == "supratts"
  resp = client.post("/tts", json={"text": "Clean return to SupraTTS."})
  assert resp.status_code == 200
  print("[PASS] Bidirectional model switching back to SupraTTS succeeded")


def test_text_normalization():
  # 1. Flow diagram text with unicode arrows and symbols
  diagram_raw = """Presign → PUT WebP → GraphQL ingest
→ validate upload → commit PROCESSING intake → publish event
→ VLM → SAM cutout → display upload → Ember embedding → REVIEW
→ user approval → AVAILABLE item + DONE intake + usage ledger"""

  norm_diagram = normalize_text_for_speech(diagram_raw)
  assert "→" not in norm_diagram, "Arrows must be converted"
  assert " plus " in norm_diagram, "Math symbols must be converted"
  assert "REVIEW." in norm_diagram or "REVIEW," in norm_diagram, "Sentence boundary after REVIEW must be present"
  assert "Then user approval" in norm_diagram, "Line break transition must be cleanly punctuated"
  print("[PASS] Diagram text normalized cleanly:", repr(norm_diagram))

  # 2. Markdown Table conversion
  table_raw = """| Model | Parameters | License | RTF |
| :--- | :--- | :--- | :--- |
| Pocket TTS | 100M | MIT | 0.71 |
| Kokoro | 82M | Apache 2.0 | 0.66 |"""

  norm_table = normalize_text_for_speech(table_raw)
  assert "|" not in norm_table, "Pipes must be stripped"
  assert "---" not in norm_table, "Separators must be stripped"
  assert "Pocket TTS" in norm_table
  assert "Parameters 100M" in norm_table
  print("[PASS] Table normalized cleanly:", repr(norm_table))

  # 3. Test through endpoint with Pocket TTS
  load_resp = client.post("/load_model", json={"repo_id": "pocket-tts"})
  assert load_resp.status_code == 200
  tts_resp = client.post("/tts", json={"text": diagram_raw})
  assert tts_resp.status_code == 200
  audio, sr = sf.read(io.BytesIO(tts_resp.content))
  assert sr == 24000
  assert len(audio) > 0
  print(f"[PASS] Pocket TTS endpoint synthesized normalized diagram ({len(audio) / sr:.2f}s)")


if __name__ == "__main__":
  test_status_and_voices()
  test_supratts_synthesis()
  test_qwen3_load_and_synthesis()
  test_pocket_tts_load_and_synthesis()
  test_safe_silence_trim_and_lead_in()
  test_chunking_and_cleanup()
  test_text_normalization()
  print("\nALL SERVER TESTS PASSED SUCCESSFULLY!")
