#!/usr/bin/env python3
"""On-Demand Voice Cloner from YouTube.

Extracts a targeted audio snippet (5-7 seconds) from a timestamped YouTube URL,
normalizes the audio, and precomputes model conditioning state (.safetensors/.pt)
grouped under ~/.cache/tts-ext/voices/ as:
  - audio__{voicename}.wav
  - pocket__{voicename}.safetensors
  - qwen3__{voicename}.pt (if requested/available)
"""

import argparse
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


CACHE_VOICES_DIR = Path.home() / ".cache" / "tts-ext" / "voices"
CACHE_VOICES_DIR.mkdir(parents=True, exist_ok=True)


def parse_timestamp(url: str, override_start: float | None = None) -> float:
  """Extract starting timestamp in seconds from URL or return override."""
  if override_start is not None:
    return float(override_start)

  parsed = urllib.parse.urlparse(url)
  qs = urllib.parse.parse_qs(parsed.query)

  # Check t or start parameter
  t_val = qs.get("t", qs.get("start", [None]))[0]
  if not t_val:
    return 0.0

  # If purely digits: seconds
  if re.match(r"^\d+s?$", t_val):
    return float(t_val.rstrip("s"))

  # If format like 1m42s or 2h10m5s
  match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?", t_val)
  if match:
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + seconds)

  return 0.0


def download_and_slice(url: str, start_sec: float, duration_sec: float = 6.0, raw_output_path: Path | None = None) -> Path:
  """Use yt-dlp to download only the specified audio range."""
  end_sec = start_sec + duration_sec
  if raw_output_path is None:
    raw_output_path = Path("/tmp") / f"yt_slice_{int(start_sec)}_{int(end_sec)}.wav"

  yt_dlp_bin = "/opt/homebrew/bin/yt-dlp" if os.path.exists("/opt/homebrew/bin/yt-dlp") else "yt-dlp"

  section_arg = f"*{start_sec:.1f}-{end_sec:.1f}"
  cmd = [
    yt_dlp_bin,
    "--download-sections", section_arg,
    "-x",
    "--audio-format", "wav",
    "-o", str(raw_output_path.with_suffix("")) + ".%(ext)s",
    url,
    "--force-overwrites",
    "--no-warnings",
    "--quiet"
  ]

  print(f"[*] Downloading snippet from {start_sec:.1f}s to {end_sec:.1f}s...")
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"yt-dlp download failed: {result.stderr or result.stdout}")

  actual_wav = raw_output_path.with_suffix(".wav")
  if not actual_wav.exists():
    candidates = list(raw_output_path.parent.glob(f"{raw_output_path.stem}*"))
    if candidates:
      actual_wav = candidates[0]
    else:
      raise FileNotFoundError(f"Expected downloaded audio file not found: {raw_output_path}")

  return actual_wav


def normalize_audio(input_path: Path, output_path: Path, sample_rate: int = 24000) -> Path:
  """Convert to 24kHz mono PCM WAV with loudnorm for clean voice cloning."""
  ffmpeg_bin = "/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "ffmpeg"

  cmd = [
    ffmpeg_bin,
    "-y",
    "-i", str(input_path),
    "-ar", str(sample_rate),
    "-ac", "1",
    "-c:a", "pcm_s16le",
    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
    str(output_path),
  ]

  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    raise RuntimeError(f"ffmpeg normalization failed: {res.stderr}")

  return output_path


def precompute_pocket_state(audio_path: Path, safetensors_path: Path):
  """Extract and export Pocket-TTS voice state (.safetensors)."""
  try:
    from pocket_tts import TTSModel, export_model_state
    print(f"[*] Generating Pocket-TTS voice cache -> {safetensors_path.name}...")
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt(str(audio_path.resolve()))
    export_model_state(state, str(safetensors_path.resolve()))
    print(f"[✓] Cached Pocket-TTS state: {safetensors_path} ({safetensors_path.stat().st_size} bytes)")
  except Exception as e:
    print(f"[!] Pocket-TTS state caching failed: {e}")


def clone_voice(
  url: str,
  name: str,
  start: float | None = None,
  duration: float = 6.0,
  precompute: bool = True
) -> dict[str, str]:
  """Full pipeline: Download -> Normalize -> Cache to ~/.cache/tts-ext/voices/"""
  start_sec = parse_timestamp(url, start)
  safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.lower().strip())

  print(f"\n=== Cloning Voice: '{safe_name}' from YouTube ===")
  print(f"URL: {url}")
  print(f"Timestamp: {start_sec}s | Duration: {duration}s")

  # File targets using grouping convention: {model}__{voicename}.{ext}
  target_audio = CACHE_VOICES_DIR / f"audio__{safe_name}.wav"
  target_pocket = CACHE_VOICES_DIR / f"pocket__{safe_name}.safetensors"
  target_qwen = CACHE_VOICES_DIR / f"qwen3__{safe_name}.pt"

  # 1. Download
  raw_temp = Path("/tmp") / f"raw_dl_{safe_name}_{int(start_sec)}.wav"
  dl_path = download_and_slice(url, start_sec, duration, raw_temp)

  # 2. Normalize
  normalize_audio(dl_path, target_audio)
  print(f"[✓] Normalized audio saved -> {target_audio}")

  # Clean up temp raw download
  if dl_path.exists() and dl_path != target_audio:
    dl_path.unlink(missing_ok=True)

  # 3. Precompute caches
  if precompute:
    precompute_pocket_state(target_audio, target_pocket)

  results = {
    "name": safe_name,
    "audio": str(target_audio),
    "pocket_cache": str(target_pocket) if target_pocket.exists() else None,
    "qwen_cache": str(target_qwen) if target_qwen.exists() else None,
  }

  print(f"\n[✓] Voice '{safe_name}' successfully cloned and ready for instant playback!")
  return results


def main():
  parser = argparse.ArgumentParser(description="Clone a voice on demand from a YouTube timestamped URL.")
  parser.add_argument("url", help="YouTube URL with timestamp (e.g. https://youtu.be/...&t=42)")
  parser.add_argument("--name", "-n", required=True, help="Name for the cloned voice (e.g. mark, lex)")
  parser.add_argument("--start", "-s", type=float, default=None, help="Start time in seconds (overrides URL timestamp)")
  parser.add_argument("--duration", "-d", type=float, default=6.0, help="Duration in seconds to extract (default: 6.0s)")
  parser.add_argument("--no-cache", action="store_true", help="Skip precomputing model states")

  args = parser.parse_args()
  clone_voice(
    url=args.url,
    name=args.name,
    start=args.start,
    duration=args.duration,
    precompute=not args.no_cache
  )


if __name__ == "__main__":
  main()
