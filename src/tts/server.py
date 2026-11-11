import argparse
import base64
import gc
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import MISSING, dataclass, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# Ensure Homebrew bin is in PATH for tools like sox and ffmpeg
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
  os.environ["PATH"] = f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"

try:
  import setproctitle

  setproctitle.setproctitle("tts-server")
except Exception:
  pass

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "supratts"))


CACHE_DIR = Path.home() / ".cache" / "tts-ext" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR = REPO_ROOT / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VOICES_DIR = Path.home() / ".cache" / "tts-ext" / "voices"
CACHE_VOICES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ALIASES = {
  "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen3-tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen-tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen3-0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen3-mlx": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen3-4bit": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
  "qwen3-pt": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "qwen3-mps": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
  "supratts": "SupraLabs/SupraTTS-1.0-Preview",
  "supra": "SupraLabs/SupraTTS-1.0-Preview",
  "microtts": "SupraLabs/SupraTTS-1.0-Preview",
  "pocket-tts": "kyutai/pocket-tts",
  "pocket_tts": "kyutai/pocket-tts",
  "pocket": "kyutai/pocket-tts",
  "pockettts": "kyutai/pocket-tts",
  "kyutai/pocket-tts": "kyutai/pocket-tts",
  "pocket-18m": "kyutai/pocket-tts-18m",
  "pocket-tts-18m": "kyutai/pocket-tts-18m",
  "kyutai/pocket-tts-18m": "kyutai/pocket-tts-18m",
  "chatterbox": "mlx-community/Chatterbox-Turbo-TTS-4bit",
  "chatterbox-turbo": "mlx-community/Chatterbox-Turbo-TTS-4bit",
  "chatterbox_turbo": "mlx-community/Chatterbox-Turbo-TTS-4bit",
}


from tts.speech_chunking import SpeechChunk, limit_speech_chunks  # noqa: E402
from tts.text_processor import (  # noqa: E402
  MODEL_CONFIGS,
  normalize_text_for_speech,
  prepare_speech_chunks,
)


class ModelManager:
  def __init__(self, ttl_seconds: int = 0):
    self.lock = threading.RLock()
    self.model_id = "kyutai/pocket-tts"
    self.type = "pocket_tts"
    self.backend = "pytorch"
    self.model = None
    self.selected_voice = "alba"
    self.prompt_cache = {}
    self.ttl_seconds = int(os.environ.get("TTS_MODEL_TTL", str(ttl_seconds)))
    self.last_active_time = time.time()
    self.load(self.model_id)

    # Background sweeper: offloads model weights to free RAM after ttl_seconds of inactivity
    if self.ttl_seconds > 0:
      threading.Thread(target=self._ttl_sweeper, daemon=True).start()

  def _ttl_sweeper(self):
    while True:
      time.sleep(15)
      if self.ttl_seconds <= 0:
        continue
      now = time.time()
      if now - self.last_active_time > self.ttl_seconds:
        with self.lock:
          if self.model is not None and (time.time() - self.last_active_time > self.ttl_seconds):
            idle_for = int(time.time() - self.last_active_time)
            print(f"[*] Server idle for {idle_for}s (> {self.ttl_seconds}s TTL). Offloading model weights to reclaim RAM...")
            self.unload()

  def unload(self):
    with self.lock:
      if self.model is not None:
        del self.model
        self.model = None
        self.prompt_cache.clear()
        gc.collect()
        if torch.backends.mps.is_available():
          torch.mps.empty_cache()
        elif torch.cuda.is_available():
          torch.cuda.empty_cache()
        # Do not import MLX while unloading a CPU model. Importing it solely
        # for cleanup can initialize its native runtime repeatedly on switches.
        mx = sys.modules.get("mlx.core")
        if self.backend == "mlx" and mx is not None:
          try:
            mx.clear_cache()
          except Exception:
            pass
        print("[*] Model weights successfully offloaded. Memory freed to OS.")

  def ensure_loaded(self):
    with self.lock:
      self.last_active_time = time.time()
      if self.model is None:
        print(f"[*] On-demand reloading '{self.model_id}'...")
        self.load(self.model_id)

  def load(self, repo_id: str, device: str | None = None):
    with self.lock:
      self._load_unlocked(repo_id, device=device)

  def _load_unlocked(self, repo_id: str, device: str | None = None):
    repo_id = repo_id.strip()
    canonical_id = MODEL_ALIASES.get(repo_id.lower(), repo_id)
    print(f"[*] Loading model: {canonical_id} (requested: {repo_id})...")

    # Free prior model resources and MPS/CUDA cache
    self.unload()

    if "supratts" in canonical_id.lower() or "microtts" in canonical_id.lower():
      try:
        from microtts import MicroTTS
      except ImportError as exc:
        raise RuntimeError("SupraTTS requires the optional microtts package.") from exc
      local_supra = REPO_ROOT / "supratts"
      if (local_supra / "config.json").exists():
        self.model = MicroTTS.load(str(local_supra))
      else:
        from huggingface_hub import snapshot_download

        m_dir = CACHE_DIR / canonical_id.replace("/", "--")
        if not (m_dir / "config.json").exists():
          snapshot_download(canonical_id, local_dir=str(m_dir))
        self.model = MicroTTS.load(str(m_dir))
      self.type = "supratts"

    elif "pocket" in canonical_id.lower() or "kyutai" in canonical_id.lower():
      from pocket_tts import TTSModel

      target_dev = device or "cpu"
      print(f"[*] Loading PocketTTS from {canonical_id} on {target_dev}...")
      language = "english_18m_2026-10" if canonical_id == "kyutai/pocket-tts-18m" else "english_2026-09"
      self.model = TTSModel.load_model(language=language)
      self.backend = "pytorch"
      if target_dev != "cpu":
        try:
          self.model.to(target_dev)
        except Exception as e:
          print(f"[!] Warning: moving pocket-tts to {target_dev} failed ({e}), remaining on cpu")
      # Anti-degeneration safeguards for FlowLM:
      # Raising EOS threshold from -4.0 to -2.0 prevents endless non-EOS loops & trailing babble.
      # Lowering temp from 0.3 to 0.2 and clamping noise prevents latent drift.
      self.model.eos_threshold = float(os.environ.get("POCKET_TTS_EOS_THRESHOLD", "-2.0"))
      self.model.temp = float(os.environ.get("POCKET_TTS_TEMP", "0.2"))
      self.model.noise_clamp = float(os.environ.get("POCKET_TTS_NOISE_CLAMP", "2.5"))
      self.type = "pocket_tts"

    elif "chatterbox" in canonical_id.lower():
      from mlx_audio.tts.utils import load_model as mlx_load_model

      print(f"[*] Loading Chatterbox Turbo from {canonical_id}...")
      self.model = mlx_load_model(canonical_id)
      self.type = "chatterbox"
      self.backend = "mlx"
      self.model_id = canonical_id
      self._chatterbox_cached_voice = None
      # Pre-warm conditionals for default voice if available
      default_voice_path = VOICES_DIR / "default.wav"
      if default_voice_path.exists():
        try:
          print(f"[*] Pre-conditioning Chatterbox with {default_voice_path.name}...")
          self.model.prepare_conditionals(str(default_voice_path.resolve()))
          self._chatterbox_cached_voice = str(default_voice_path.resolve())
        except Exception as e:
          print(f"[!] Warning: Chatterbox pre-conditioning failed: {e}")
      print(f"[*] Successfully loaded Chatterbox: {self.model_id} (type: {self.type}, backend: {self.backend})")
      return

    elif "qwen3" in canonical_id.lower() or "qwen" in canonical_id.lower():
      use_mlx = (
        "mlx" in canonical_id.lower()
        or canonical_id == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
        or (os.environ.get("TTS_USE_MLX", "1") == "1" and not canonical_id.endswith("-pt") and not canonical_id.endswith("-mps"))
      )
      if use_mlx:
        try:
          from mlx_audio.tts.utils import load_model as mlx_load_model
          mlx_target = canonical_id if "mlx" in canonical_id.lower() else "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
          print(f"[*] Loading high-speed MLX Qwen3TTS from {mlx_target}...")
          self.model = mlx_load_model(mlx_target)
          self.type = "qwen3"
          self.backend = "mlx"
          self.model_id = canonical_id
          print(f"[*] Successfully loaded MLX Qwen3TTS: {self.model_id} (type: {self.type}, backend: {self.backend})")
          return
        except Exception as e:
          print(f"[!] Failed to load MLX Qwen3TTS ({e}), falling back to PyTorch qwen_tts...")

      from qwen_tts import Qwen3TTSModel

      self.backend = "pytorch"
      target_dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
      m_dir = CACHE_DIR / canonical_id.replace("/", "--")
      load_target = str(m_dir) if (m_dir / "config.json").exists() else canonical_id
      print(f"[*] Loading Qwen3TTS from {load_target} on {target_dev} (bfloat16)...")
      load_kwargs = {"device_map": target_dev}
      if target_dev == "mps" and hasattr(torch, "bfloat16"):
        load_kwargs["dtype"] = torch.bfloat16
      try:
        self.model = Qwen3TTSModel.from_pretrained(load_target, **load_kwargs)
      except Exception as e:
        print(f"[!] Failed to load on {target_dev} with bfloat16 ({e}), trying standard...")
        try:
          self.model = Qwen3TTSModel.from_pretrained(load_target, device_map=target_dev)
        except Exception as e2:
          print(f"[!] Falling back to CPU: {e2}")
          self.model = Qwen3TTSModel.from_pretrained(load_target, device_map="cpu")
      self.type = "qwen3"

    else:
      from transformers import pipeline

      target_dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
      self.model = pipeline("text-to-speech", model=canonical_id, device=target_dev)
      self.type = "transformers"

    self.model_id = canonical_id
    print(f"[*] Successfully loaded: {self.model_id} (type: {self.type})")

  @staticmethod
  def list_voices() -> list[str]:
    voices = {"alba"}
    for d in [VOICES_DIR, CACHE_VOICES_DIR]:
      if d.exists():
        for f in d.iterdir():
          if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            stem = f.stem
            if stem.startswith("audio__"):
              voices.add(stem[len("audio__"):])
            else:
              voices.add(f.name)
    return sorted(voices)

  def resolve_voice_ref(
    self,
    voice: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
  ) -> tuple[str | None, str | None]:
    target_audio = ref_audio
    target_text = ref_text

    # If explicit voice name given, look up in voices/ or local path
    if voice:
      voice_clean = voice.strip()
      candidates = [
        CACHE_VOICES_DIR / f"audio__{voice_clean}.wav",
        CACHE_VOICES_DIR / f"{voice_clean}.wav",
        VOICES_DIR / voice_clean,
        VOICES_DIR / f"{voice_clean}.wav",
        VOICES_DIR / f"{voice_clean}.mp3",
        VOICES_DIR / f"{voice_clean}.flac",
        Path(voice_clean),
      ]
      found = False
      for c in candidates:
        if c.exists() and c.is_file():
          target_audio = str(c.resolve())
          txt_path = c.with_suffix(".txt")
          if txt_path.exists() and not target_text:
            target_text = txt_path.read_text(encoding="utf-8").strip()
          found = True
          break
      if not found:
        raise FileNotFoundError(f"Voice '{voice}' not found in {VOICES_DIR}. Available voices: {self.list_voices()}")

    # Fallback to default voice ONLY if neither voice nor ref_audio provided
    elif not target_audio:
      default_candidates = [
        VOICES_DIR / "default.wav",
        VOICES_DIR / "default.mp3",
      ]
      for c in default_candidates:
        if c.exists() and c.is_file():
          target_audio = str(c.resolve())
          txt_path = c.with_suffix(".txt")
          if txt_path.exists() and not target_text:
            target_text = txt_path.read_text(encoding="utf-8").strip()
          break

    # If ref_audio was passed directly as a local file, try companion .txt if no text given
    if target_audio and not target_text:
      p = Path(target_audio)
      if p.exists() and p.is_file():
        txt_path = p.with_suffix(".txt")
        if txt_path.exists():
          target_text = txt_path.read_text(encoding="utf-8").strip()

    return target_audio, target_text

  def get_voice_clone_prompt(
    self,
    ref_audio: str,
    ref_text: str | None = None,
    x_vector_only_mode: bool = False,
  ):
    if not ref_text:
      x_vector_only_mode = True

    ref_path = Path(ref_audio).resolve() if Path(ref_audio).exists() else None
    cache_key = (
      str(ref_path) if ref_path else ref_audio,
      "" if x_vector_only_mode else (ref_text or ""),
      bool(x_vector_only_mode),
    )

    # 1. In-memory RAM cache (0ms)
    if cache_key in self.prompt_cache:
      return self.prompt_cache[cache_key]

    # 2. On-disk persistent cache (.pt)
    pt_path = None
    if ref_path:
      mode_suffix = "_xvec" if x_vector_only_mode else "_prompt"
      pt_path = ref_path.with_name(f"{ref_path.stem}{mode_suffix}.pt")
      if pt_path.exists():
        try:
          prompt = torch.load(str(pt_path), weights_only=False)
          self.prompt_cache[cache_key] = prompt
          return prompt
        except Exception as e:
          print(f"[!] Error loading prompt cache {pt_path}: {e}")

    # 3. Compute and cache
    prompt = self.model.create_voice_clone_prompt(
      ref_audio=ref_audio,
      ref_text=ref_text,
      x_vector_only_mode=x_vector_only_mode,
    )
    self.prompt_cache[cache_key] = prompt

    if pt_path:
      try:
        torch.save(prompt, str(pt_path))
        print(f"[*] Cached voice clone prompt to disk: {pt_path.name}")
      except Exception as e:
        print(f"[!] Error saving prompt cache {pt_path}: {e}")

    return prompt

  def resolve_pocket_voice(
    self,
    voice: str | None = None,
    ref_audio: str | None = None,
  ) -> str:
    # 1. Explicit ref_audio (local path, URL, or hf://)
    if ref_audio:
      p = Path(ref_audio)
      if p.exists() and p.is_file():
        return str(p.resolve())
      return ref_audio

    # 2. Explicit voice name
    if voice:
      voice_clean = voice.strip()
      candidates = [
        CACHE_VOICES_DIR / f"audio__{voice_clean}.wav",
        CACHE_VOICES_DIR / f"{voice_clean}.wav",
        VOICES_DIR / voice_clean,
        VOICES_DIR / f"{voice_clean}.wav",
        VOICES_DIR / f"{voice_clean}.mp3",
        VOICES_DIR / f"{voice_clean}.flac",
        Path(voice_clean),
      ]
      for c in candidates:
        if c.exists() and c.is_file():
          return str(c.resolve())

      from pocket_tts.models.tts_model import _ORIGINS_OF_PREDEFINED_VOICES

      if voice_clean.lower() in _ORIGINS_OF_PREDEFINED_VOICES:
        return voice_clean.lower()

      raise FileNotFoundError(f"Voice '{voice}' not found in {VOICES_DIR} or predefined pocket-tts voices. Available voices: {self.list_voices()}")

    # 3. Default fallback to local voice in voices/ if present
    default_candidates = [
      VOICES_DIR / "default.wav",
      VOICES_DIR / "default.mp3",
    ]
    for c in default_candidates:
      if c.exists() and c.is_file():
        return str(c.resolve())

    # 4. Fallback to predefined voice 'alba'
    return "alba"

  def get_pocket_voice_state(self, voice_conditioning: str):
    cache_key = (self.model_id, voice_conditioning)
    # 1. In-memory RAM cache (0ms)
    if cache_key in self.prompt_cache:
      return self.prompt_cache[cache_key]

    # 2. On-disk persistent safetensors cache for local audio files
    v_path = Path(voice_conditioning) if Path(voice_conditioning).exists() else None
    st_path = None
    if v_path and v_path.is_file() and v_path.suffix.lower() in [".wav", ".mp3", ".flac"]:
      raw_name = v_path.stem.replace("audio__", "")
      is_small = self.model_id == "kyutai/pocket-tts-18m"
      if is_small:
        st_candidates = [
          v_path.with_name(f"{raw_name}_pocket_18m_2026-10.safetensors"),
          CACHE_VOICES_DIR / f"pocket18m__{raw_name}.safetensors",
        ]
      else:
        st_candidates = [
          v_path.with_name(f"pocket__{raw_name}.safetensors"),
          v_path.with_name(f"{v_path.stem}_pocket.safetensors"),
          CACHE_VOICES_DIR / f"pocket__{raw_name}.safetensors",
        ]
      for cand in st_candidates:
        if cand.exists():
          st_path = cand
          break
      if not st_path:
        prefix = "pocket18m" if is_small else "pocket"
        st_path = CACHE_VOICES_DIR / f"{prefix}__{raw_name}.safetensors"

      if st_path.exists():
        try:
          state = self.model.get_state_for_audio_prompt(st_path)
          self.prompt_cache[cache_key] = state
          return state
        except Exception as e:
          print(f"[!] Error loading pocket voice cache {st_path}: {e}")

    # 3. Compute and cache
    state = self.model.get_state_for_audio_prompt(voice_conditioning)
    self.prompt_cache[cache_key] = state

    if st_path:
      try:
        from pocket_tts import export_model_state

        export_model_state(state, st_path)
        print(f"[*] Cached pocket voice state to disk: {st_path.name}")
      except Exception as e:
        print(f"[!] Error saving pocket voice cache {st_path}: {e}")

    return state

  @staticmethod
  def _prepare_speech_chunks(
    text: str,
    max_words: int = 28,
    strategy: str = "single_sentence",
  ) -> list[SpeechChunk]:
    return prepare_speech_chunks(text, max_words=max_words, strategy=strategy)

  def prepare_chunks_for_model(self, text: str) -> list[SpeechChunk]:
    """Prepare the same bounded chunks for complete and incremental playback."""
    cfg = MODEL_CONFIGS.get(self.type, MODEL_CONFIGS["pocket_tts"])
    parse_words = 1000000 if self.type == "pocket_tts" else cfg["max_words"]
    chunks = self._prepare_speech_chunks(text, max_words=parse_words, strategy=cfg["strategy"])
    token_count = None
    if self.type == "pocket_tts":
      tokenizer = self.model.flow_lm.conditioner.tokenizer

      def token_count(value):
        return len(tokenizer.encode(value))
    return limit_speech_chunks(chunks, max_words=cfg["max_words"], token_count=token_count)

  def synthesize(
    self,
    text: str,
    noise_scale: float = 0.0,
    voice: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    language: str = "English",
    x_vector_only_mode: bool = False,
  ) -> tuple[np.ndarray, int]:
    with self.lock:
      self.ensure_loaded()
      chunks = self.prepare_chunks_for_model(text)
      if not chunks:
        chunks = [SpeechChunk(text=normalize_text_for_speech(text) or "")]
        if not chunks[0].text:
          return np.zeros(24000, dtype=np.float32), 24000

      wavs_with_pause, sr = [], 24000

      try:
        if self.type == "qwen3":
          audio_path, transcript = self.resolve_voice_ref(voice, ref_audio, ref_text)
          if not audio_path:
            raise ValueError("No reference voice available for Qwen3-TTS. Please add audio to voices/ or specify 'voice' / 'ref_audio'.")

          if getattr(self, "backend", None) == "mlx":
            from mlx_audio.utils import load_audio
            sr = getattr(self.model, "sample_rate", 24000)
            ref_audio_arr = load_audio(audio_path, sample_rate=sr)

            # Pass reference audio and transcript with 100% exact alignment (no partial slicing)
            ref_txt = None if x_vector_only_mode else (transcript.strip() if transcript else None)

            for chunk in chunks:
              clean_text = chunk.text.strip()
              if not clean_text:
                continue

              # Generous max_tokens ceiling (512-2048) so the model naturally concludes at EOS
              # without cutting off words or trail decay
              num_words = len(clean_text.split())
              max_tok = min(2048, max(512, num_words * 25))

              chunk_sub_wavs = []
              for res in self.model.generate(
                text=clean_text,
                ref_audio=ref_audio_arr,
                ref_text=ref_txt,
                temperature=0.35,
                top_k=40,
                repetition_penalty=1.35,
                stream=True,
                streaming_interval=0.32,
                max_tokens=max_tok,
              ):
                chunk_sub_wavs.append(np.array(res.audio, dtype=np.float32))

              if chunk_sub_wavs:
                chunk_wav = np.concatenate(chunk_sub_wavs)
                wavs_with_pause.append((chunk_wav, chunk.pause_after_ms))

          else:
            prompt = self.get_voice_clone_prompt(
              ref_audio=audio_path,
              ref_text=transcript,
              x_vector_only_mode=x_vector_only_mode,
            )

            with torch.inference_mode():
              for chunk in chunks:
                chunk_wavs, sr = self.model.generate_voice_clone(
                  text=chunk.text,
                  language=language,
                  voice_clone_prompt=prompt,
                )
                w = chunk_wavs[0]
                if hasattr(w, "squeeze"):
                  w = w.squeeze()
                wavs_with_pause.append((np.array(w, dtype=np.float32), chunk.pause_after_ms))
                if torch.backends.mps.is_available():
                  torch.mps.synchronize()

        elif self.type == "chatterbox":
          sr = getattr(self.model, "sample_rate", 24000)
          audio_path, _ = self.resolve_voice_ref(voice, ref_audio, ref_text)
          if audio_path and getattr(self, "_chatterbox_cached_voice", None) != audio_path:
            self.model.prepare_conditionals(audio_path)
            self._chatterbox_cached_voice = audio_path

          for chunk in chunks:
            clean_text = chunk.text.strip()
            if not clean_text:
              continue
            results = list(self.model.generate(text=clean_text))
            if results:
              chunk_audio = np.concatenate([np.array(r.audio) for r in results])
              wavs_with_pause.append((np.array(chunk_audio, dtype=np.float32), chunk.pause_after_ms))

        elif self.type == "pocket_tts":
          sr = getattr(self.model, "sample_rate", 24000)
          voice_conditioning = self.resolve_pocket_voice(voice, ref_audio)
          voice_state = self.get_pocket_voice_state(voice_conditioning)

          with torch.no_grad():
            for chunk in chunks:
              clean_text = chunk.text.strip()
              if not clean_text:
                continue
              # Guarantee terminal punctuation and capitalization to prevent non-EOS autoregressive drift
              if clean_text[-1] not in ".!?…\"'”’":
                clean_text = clean_text.rstrip(",;:-–— ") + "."
              if clean_text and clean_text[0].islower():
                clean_text = clean_text[0].upper() + clean_text[1:]

              chunk_wav = self.model.generate_audio(
                voice_state,
                clean_text,
                copy_state=True,
              )
              w = chunk_wav.detach().cpu().numpy()
              if hasattr(w, "squeeze"):
                w = w.squeeze()
              wavs_with_pause.append((np.array(w, dtype=np.float32), chunk.pause_after_ms))

        elif self.type == "supratts":
          if voice:
            self.resolve_voice_ref(voice, ref_audio, ref_text)
          sr = 24000
          for chunk in chunks:
            w = self.model.synthesize(chunk.text, noise_scale=noise_scale)
            wavs_with_pause.append((np.array(w, dtype=np.float32), chunk.pause_after_ms))

        else:
          if voice:
            self.resolve_voice_ref(voice, ref_audio, ref_text)
          for chunk in chunks:
            out = self.model(chunk.text)
            w = out["audio"]
            if hasattr(w, "squeeze"):
              w = w.squeeze()
            sr = int(out["sampling_rate"])
            wavs_with_pause.append((np.array(w, dtype=np.float32), chunk.pause_after_ms))

        combined = []
        for i, (w, pause_ms) in enumerate(wavs_with_pause):
          is_last = i == len(wavs_with_pause) - 1
          # Internal padding contributes to the desired pause; do not add it twice.
          w_clean = self._trim_silence(w, sr=sr, trail_ms=200.0 if is_last else 60.0)
          combined.append(w_clean)
          if not is_last:
            pause_samples = int((max(0.0, pause_ms - 100.0) / 1000.0) * sr)
            if pause_samples > 0:
              combined.append(np.zeros(pause_samples, dtype=np.float32))
        final_audio = np.concatenate(combined) if combined else np.zeros(sr, dtype=np.float32)
        # Universal Spotify-style loudness normalization across all models (-26 dBFS target)
        normalized_audio = self._match_target_loudness(final_audio, target_db=-26.0, peak_limit=0.90)
        return normalized_audio, sr

      finally:
        self.last_active_time = time.time()
        if torch.backends.mps.is_available():
          torch.mps.empty_cache()

  @staticmethod
  def _match_target_loudness(
    audio: np.ndarray,
    target_db: float = -26.0,
    peak_limit: float = 0.90,
  ) -> np.ndarray:
    """Universal loudness normalizer (EBU R128 / Spotify style active-speech normalization).

    Ensures Pocket-TTS, Qwen3-TTS, SupraTTS, and any future models have unified, comfortable volume.
    """
    if len(audio) == 0:
      return audio
    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    if peak < 1e-4:
      return audio

    # Compute RMS over active speech samples (>1% peak) to ignore inter-sentence pauses
    active_samples = audio[abs_audio > (0.01 * peak)]
    rms = float(np.sqrt(np.mean(active_samples**2))) if len(active_samples) > 0 else float(np.sqrt(np.mean(audio**2)))
    current_db = 20.0 * np.log10(max(rms, 1e-7))

    gain_db = target_db - current_db
    gain_linear = 10.0 ** (gain_db / 20.0)

    scaled = audio * gain_linear
    max_peak = float(np.max(np.abs(scaled)))
    if max_peak > peak_limit:
      scaled = scaled * (peak_limit / max_peak)

    return scaled.astype(np.float32)

  @staticmethod
  def _trim_silence(
    w: np.ndarray,
    sr: int = 24000,
    thresh_db: float = -65.0,
    lead_ms: float = 40.0,
    trail_ms: float = 200.0,
  ) -> np.ndarray:
    if len(w) == 0:
      return w
    peak = np.max(np.abs(w))
    if peak < 1e-4:
      return w

    # Linear amplitude floor at -65dB with safety check against ultra-soft fricatives (/th/, /s/, /f/)
    linear_thresh = max(10.0 ** (thresh_db / 20.0), 0.0002)
    cutoff = min(linear_thresh, 0.0005 * peak)

    idx = np.where(np.abs(w) > cutoff)[0]
    if len(idx) == 0:
      return w

    speech_start = int(idx[0])
    speech_end = int(idx[-1] + 1)

    lead_samples = int((lead_ms / 1000.0) * sr)
    trail_samples = int((trail_ms / 1000.0) * sr)

    # Guarantee at least lead_samples of pre-roll silence before speech starts
    if speech_start >= lead_samples:
      start_idx = speech_start - lead_samples
      lead_pad = np.array([], dtype=w.dtype)
    else:
      start_idx = 0
      lead_pad = np.zeros(lead_samples - speech_start, dtype=w.dtype)

    # Guarantee at least trail_samples of post-roll silence after speech ends
    avail_trail = len(w) - speech_end
    if avail_trail >= trail_samples:
      end_idx = speech_end + trail_samples
      trail_pad = np.array([], dtype=w.dtype)
    else:
      end_idx = len(w)
      trail_pad = np.zeros(trail_samples - avail_trail, dtype=w.dtype)

    core = w[start_idx:end_idx]
    parts = []
    if len(lead_pad) > 0:
      parts.append(lead_pad)
    parts.append(core)
    if len(trail_pad) > 0:
      parts.append(trail_pad)

    return np.concatenate(parts) if len(parts) > 1 else core

  @staticmethod
  def _chunk_text(text: str, max_words: int = 120, strategy: str = "multi_sentence") -> list[str]:
    text = text.strip()
    if not text:
      return []

    # Split on sentence boundaries (. ? ! \n). Also handle sentences closing with quotes.
    # NEVER split on commas (,).
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+|(?<=[.!?][\"'\u201d\u2019])\s+", text) if s.strip()]
    if not sentences:
      sentences = [text]

    if strategy == "single_sentence":
      chunks = []
      for sent in sentences:
        sent_words = sent.split()
        if len(sent_words) <= max_words:
          chunks.append(sent)
        else:
          clauses = [c.strip() for c in re.split(r"(?<=[;:—])\s+", sent) if c.strip()]
          cur, cur_cnt = [], 0
          for cl in clauses:
            cl_words = cl.split()
            if len(cl_words) > max_words:
              if cur:
                chunks.append(" ".join(cur))
                cur, cur_cnt = [], 0
              for i in range(0, len(cl_words), max_words):
                chunks.append(" ".join(cl_words[i : i + max_words]))
              continue
            if cur_cnt + len(cl_words) <= max_words:
              cur.append(cl)
              cur_cnt += len(cl_words)
            else:
              if cur:
                chunks.append(" ".join(cur))
              cur = [cl]
              cur_cnt = len(cl_words)
          if cur:
            chunks.append(" ".join(cur))
      return chunks or [text]

    # multi_sentence strategy (packs full sentences up to max_words without breaking mid-sentence)
    chunks = []
    cur_chunk = []
    cur_count = 0

    for sent in sentences:
      sent_words = sent.split()
      if not sent_words:
        continue

      if len(sent_words) > max_words:
        if cur_chunk:
          chunks.append(" ".join(cur_chunk))
          cur_chunk, cur_count = [], 0
        clauses = [c.strip() for c in re.split(r"(?<=[;:—])\s+", sent) if c.strip()]
        for cl in clauses:
          cl_words = cl.split()
          if cur_count + len(cl_words) <= max_words:
            cur_chunk.append(cl)
            cur_count += len(cl_words)
          else:
            if cur_chunk:
              chunks.append(" ".join(cur_chunk))
              cur_chunk, cur_count = [], 0
            if len(cl_words) <= max_words:
              cur_chunk = [cl]
              cur_count = len(cl_words)
            else:
              for i in range(0, len(cl_words), max_words):
                chunks.append(" ".join(cl_words[i : i + max_words]))
      else:
        if cur_count + len(sent_words) <= max_words:
          cur_chunk.append(sent)
          cur_count += len(sent_words)
        else:
          if cur_chunk:
            chunks.append(" ".join(cur_chunk))
          cur_chunk = [sent]
          cur_count = len(sent_words)

    if cur_chunk:
      chunks.append(" ".join(cur_chunk))

    return chunks or [text]


manager = ModelManager()


def normalize(wav: np.ndarray, rms: float = 0.07) -> np.ndarray:
  scaled = (wav / (np.sqrt(np.mean(wav**2)) + 1e-9) * rms).astype(np.float32)
  return np.clip(scaled, -0.99, 0.99)


@dataclass
class LoadModelReq:
  repo_id: str
  device: str | None = None


@dataclass
class TTSReq:
  text: str
  model: str | None = None
  rms: float = 0.07
  noise_scale: float = 0.0
  voice: str | None = None
  ref_audio: str | None = None
  ref_text: str | None = None
  language: str = "English"
  x_vector_only_mode: bool = False


def parse_dataclass(cls, data: dict):
  if not isinstance(data, dict):
    raise ValueError("Request body must be a JSON object")
  init_kwargs = {}
  for f in fields(cls):
    if f.name in data:
      init_kwargs[f.name] = data[f.name]
    elif f.default is not MISSING:
      init_kwargs[f.name] = f.default
    elif f.default_factory is not MISSING:
      init_kwargs[f.name] = f.default_factory()
    else:
      raise ValueError(f"Missing required field: '{f.name}'")
  return cls(**init_kwargs)


def get_cors_headers(origin: str | None = None) -> dict[str, str]:
  return {
    "Access-Control-Allow-Origin": origin if origin else "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS, HEAD",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Private-Network": "true",
    "Access-Control-Max-Age": "86400",
  }


def handle_request(method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
  parsed_url = urllib.parse.urlparse(path)
  clean_path = urllib.parse.unquote(parsed_url.path).rstrip("/") or "/"
  origin = None
  for k, v in headers.items():
    if k.lower() == "origin":
      origin = v
      break
  cors = get_cors_headers(origin)

  if method == "OPTIONS":
    return 204, cors, b""

  if method == "GET":
    if clean_path == "/":
      with manager.lock:
        is_loaded = manager.model is not None
      idle_sec = round(time.time() - manager.last_active_time, 1)
      resp_data = {
        "status": "ok",
        "current_model": manager.model_id,
        "model_type": manager.type,
        "backend": getattr(manager, "backend", "pytorch"),
        "selected_voice": getattr(manager, "selected_voice", "alba"),
        "available_voices": manager.list_voices(),
        "model_config": MODEL_CONFIGS.get(manager.type, MODEL_CONFIGS["pocket_tts"]),
        "model_loaded": is_loaded,
        "idle_seconds": idle_sec,
        "ttl_seconds": manager.ttl_seconds,
      }
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 200, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/voices":
      resp_data = {"voices": manager.list_voices()}
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 200, resp_headers, json.dumps(resp_data).encode("utf-8")

    else:
      resp_data = {"status": "error", "message": f"Endpoint '{clean_path}' not found"}
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 404, resp_headers, json.dumps(resp_data).encode("utf-8")

  elif method == "POST":
    if clean_path == "/offload":
      with manager.lock:
        manager.unload()
      resp_data = {
        "status": "ok",
        "message": f"Model '{manager.model_id}' weights offloaded from memory",
        "model_loaded": False,
      }
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 200, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/shutdown":
      print("[*] Immediate shutdown requested. Exiting server process...")
      threading.Thread(target=lambda: (time.sleep(0.05), os._exit(0)), daemon=True).start()
      resp_data = {"status": "ok", "message": "Server shutting down"}
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 200, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/load_model":
      try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        req = parse_dataclass(LoadModelReq, payload)
      except Exception as e:
        resp_data = {"status": "error", "message": f"Invalid request body: {e}"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      if not req.repo_id or not str(req.repo_id).strip():
        resp_data = {"status": "error", "message": "Field 'repo_id' cannot be empty"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      try:
        manager.load(req.repo_id, device=req.device)
        resp_data = {
          "status": "ok",
          "current_model": manager.model_id,
          "model_type": manager.type,
        }
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 200, resp_headers, json.dumps(resp_data).encode("utf-8")
      except Exception as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 500, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/set_voice":
      try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        v = payload.get("voice")
        if v:
          manager.selected_voice = str(v).strip()
        resp_data = {"status": "ok", "selected_voice": manager.selected_voice}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 200, resp_headers, json.dumps(resp_data).encode("utf-8")
      except Exception as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/speak":
      try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        req = parse_dataclass(TTSReq, payload)
      except Exception as e:
        resp_data = {"status": "error", "message": f"Invalid request body: {e}"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      text_to_speak = str(req.text).strip() if req.text is not None else ""
      if not text_to_speak:
        resp_data = {"status": "error", "message": "Text cannot be empty"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      try:
        if req.model:
          model_lower = req.model.lower()
          if "pocket" in model_lower or "kyutai" in model_lower:
            canonical_req_type = "pocket_tts"
          elif "chatterbox" in model_lower:
            canonical_req_type = "chatterbox"
          elif "qwen" in model_lower:
            canonical_req_type = "qwen3"
          elif "supra" in model_lower or "microtts" in model_lower:
            canonical_req_type = "supratts"
          else:
            canonical_req_type = "transformers"

          canonical_id = MODEL_ALIASES.get(req.model.strip().lower(), req.model.strip())
          if manager.type != canonical_req_type or manager.model_id != canonical_id:
            print(f"[*] Request specified model '{req.model}', switching from '{manager.type}' ({manager.model_id})...")
            manager.load(req.model)

        rms_val = req.rms if (req.rms is not None and req.rms > 0) else 0.07
        voice_to_use = req.voice or getattr(manager, "selected_voice", "alba")
        wav, sample_rate = manager.synthesize(
          text_to_speak,
          noise_scale=req.noise_scale,
          voice=voice_to_use,
          ref_audio=req.ref_audio,
          ref_text=req.ref_text,
          language=req.language,
          x_vector_only_mode=req.x_vector_only_mode,
        )
        norm_wav = normalize(wav, rms=rms_val)

        tmp_speak_path = f"/tmp/speak_{int(time.time() * 1000)}.wav"
        sf.write(tmp_speak_path, norm_wav, sample_rate, format="WAV", subtype="PCM_24")

        # Play asynchronously through macOS audio output
        def _play_and_cleanup(p):
          try:
            subprocess.run(["afplay", p], check=True)
          except Exception as pe:
            print(f"[!] Playback error: {pe}")
          finally:
            try:
              if os.path.exists(p):
                os.remove(p)
            except Exception:
              pass

        threading.Thread(target=_play_and_cleanup, args=(tmp_speak_path,), daemon=True).start()

        resp_data = {
          "status": "ok",
          "message": "Speaking message",
          "model": manager.model_id,
          "sample_rate": sample_rate,
          "duration_seconds": round(len(norm_wav) / sample_rate, 2),
        }
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 200, resp_headers, json.dumps(resp_data).encode("utf-8")
      except Exception as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 500, resp_headers, json.dumps(resp_data).encode("utf-8")

    elif clean_path == "/tts":
      try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        req = parse_dataclass(TTSReq, payload)
      except Exception as e:
        resp_data = {"status": "error", "message": f"Invalid request body: {e}"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      text = str(req.text).strip() if req.text is not None else ""
      if not text:
        resp_data = {"status": "error", "message": "Text cannot be empty"}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")

      try:
        if req.model:
          model_lower = req.model.lower()
          if "pocket" in model_lower or "kyutai" in model_lower:
            canonical_req_type = "pocket_tts"
          elif "chatterbox" in model_lower:
            canonical_req_type = "chatterbox"
          elif "qwen" in model_lower:
            canonical_req_type = "qwen3"
          elif "supra" in model_lower or "microtts" in model_lower:
            canonical_req_type = "supratts"
          else:
            canonical_req_type = "transformers"

          canonical_id = MODEL_ALIASES.get(req.model.strip().lower(), req.model.strip())
          if manager.type != canonical_req_type or manager.model_id != canonical_id:
            print(f"[*] Request specified model '{req.model}', switching from '{manager.type}' ({manager.model_id})...")
            manager.load(req.model)

        rms_val = req.rms if (req.rms is not None and req.rms > 0) else 0.07
        wav, sample_rate = manager.synthesize(
          text,
          noise_scale=req.noise_scale,
          voice=req.voice,
          ref_audio=req.ref_audio,
          ref_text=req.ref_text,
          language=req.language,
          x_vector_only_mode=req.x_vector_only_mode,
        )
        norm_wav = normalize(wav, rms=rms_val)
        buf = io.BytesIO()
        sf.write(buf, norm_wav, sample_rate, format="WAV", subtype="PCM_24")
        wav_bytes = buf.getvalue()
        resp_headers = {**cors, "Content-Type": "audio/wav"}
        return 200, resp_headers, wav_bytes
      except FileNotFoundError as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 404, resp_headers, json.dumps(resp_data).encode("utf-8")
      except ValueError as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 400, resp_headers, json.dumps(resp_data).encode("utf-8")
      except Exception as e:
        resp_data = {"status": "error", "message": str(e)}
        resp_headers = {**cors, "Content-Type": "application/json"}
        return 500, resp_headers, json.dumps(resp_data).encode("utf-8")

    else:
      resp_data = {"status": "error", "message": f"Endpoint '{clean_path}' not found"}
      resp_headers = {**cors, "Content-Type": "application/json"}
      return 404, resp_headers, json.dumps(resp_data).encode("utf-8")

  else:
    resp_data = {"status": "error", "message": f"Method '{method}' not allowed"}
    resp_headers = {**cors, "Content-Type": "application/json"}
    return 405, resp_headers, json.dumps(resp_data).encode("utf-8")


class TTSRequestHandler(BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  def log_message(self, format, *args):
    # Suppress default HTTP access logging to stderr
    pass

  def do_OPTIONS(self):
    self._dispatch("OPTIONS", b"")

  def do_HEAD(self):
    self._dispatch("HEAD", b"")

  def do_GET(self):
    self._dispatch("GET", b"")

  def do_POST(self):
    try:
      length = int(self.headers.get("Content-Length", 0))
      length = max(0, length)
    except (ValueError, TypeError):
      length = 0
    try:
      body = self.rfile.read(length) if length > 0 else b""
    except Exception:
      body = b""
    if urllib.parse.urlparse(self.path).path.rstrip("/") == "/tts/stream":
      self._stream_tts(body)
    else:
      self._dispatch("POST", body)

  def _stream_tts(self, body: bytes):
    """Send complete, independently playable WAV chunks as newline-delimited JSON."""
    try:
      req = parse_dataclass(TTSReq, json.loads(body.decode("utf-8")) if body else {})
      text = str(req.text).strip() if req.text is not None else ""
      if not text:
        raise ValueError("Text cannot be empty")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
      payload = json.dumps({"status": "error", "message": str(exc)}).encode()
      self.send_response(400)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(payload)))
      self.end_headers()
      self.wfile.write(payload)
      return

    self.send_response(200)
    self.send_header("Content-Type", "application/x-ndjson")
    self.send_header("Connection", "close")
    self.end_headers()
    self.close_connection = True

    def send_frame(frame):
      self.wfile.write(json.dumps(frame, separators=(",", ":")).encode() + b"\n")
      self.wfile.flush()

    try:
      # Model switches and the full stream are serialized for the single user.
      # A disconnected client releases the lock after the current chunk.
      with manager.lock:
        if req.model:
          canonical_id = MODEL_ALIASES.get(req.model.strip().lower(), req.model.strip())
          if manager.model_id != canonical_id:
            manager.load(req.model)
        manager.ensure_loaded()
        chunks = manager.prepare_chunks_for_model(text)
        if not chunks:
          raise ValueError("No speakable text found")
        for index, chunk in enumerate(chunks):
          wav, sr = manager.synthesize(
            chunk.text, noise_scale=req.noise_scale, voice=req.voice,
            ref_audio=req.ref_audio, ref_text=req.ref_text, language=req.language,
            x_vector_only_mode=req.x_vector_only_mode,
          )
          is_last = index == len(chunks) - 1
          wav = manager._trim_silence(wav, sr=sr, trail_ms=200.0 if is_last else 60.0)
          if not is_last:
            pause = int(max(0.0, chunk.pause_after_ms - 100.0) * sr / 1000.0)
            if pause:
              wav = np.concatenate((wav, np.zeros(pause, dtype=np.float32)))
          rms = req.rms if req.rms is not None and req.rms > 0 else 0.07
          audio = io.BytesIO()
          sf.write(audio, normalize(wav, rms=rms), sr, format="WAV", subtype="PCM_24")
          send_frame({"type": "audio", "wav": base64.b64encode(audio.getvalue()).decode("ascii")})
        send_frame({"type": "end"})
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
      pass
    except Exception as exc:
      try:
        send_frame({"type": "error", "message": str(exc)})
      except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass

  def __getattr__(self, name: str):
    if name.startswith("do_"):
      method = name[3:]
      return lambda: self._dispatch(method, b"")
    raise AttributeError(name)

  def _dispatch(self, method: str, body: bytes):
    headers_dict = {k: v for k, v in self.headers.items()}
    req_method = "GET" if method == "HEAD" else method
    status, resp_headers, resp_body = handle_request(req_method, self.path, headers_dict, body)
    try:
      self.send_response(status)
      for k, v in resp_headers.items():
        self.send_header(k, v)
      self.send_header("Content-Length", str(len(resp_body)))
      self.end_headers()
      if method != "HEAD" and resp_body:
        self.wfile.write(resp_body)
      self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
      pass


class TTSHTTPServer(ThreadingHTTPServer):
  allow_reuse_address = True
  daemon_threads = True

  def server_bind(self):
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
      try:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
      except Exception:
        pass
    super().server_bind()


async def app(scope, receive, send):
  """Pure ASGI application adapter for compatibility with ASGI test runners."""
  if scope["type"] == "lifespan":
    while True:
      message = await receive()
      if message["type"] == "lifespan.startup":
        await send({"type": "lifespan.startup.complete"})
      elif message["type"] == "lifespan.shutdown":
        await send({"type": "lifespan.shutdown.complete"})
        return
  if scope["type"] != "http":
    return

  body = bytearray()
  while True:
    message = await receive()
    if message["type"] == "http.request":
      body.extend(message.get("body", b""))
      if not message.get("more_body", False):
        break

  headers_dict = {}
  for k, v in scope.get("headers", []):
    headers_dict[k.decode("latin1")] = v.decode("latin1")

  path = scope.get("path", "/")
  method = scope.get("method", "GET")
  status, resp_headers, resp_body = handle_request(method, path, headers_dict, bytes(body))

  header_list = [(k.encode("latin1"), v.encode("latin1")) for k, v in resp_headers.items()]
  header_list.append((b"content-length", str(len(resp_body)).encode("latin1")))

  await send(
    {
      "type": "http.response.start",
      "status": status,
      "headers": header_list,
    }
  )
  await send(
    {
      "type": "http.response.body",
      "body": resp_body,
    }
  )


def run_server(host: str = "127.0.0.1", port: int = 5050):
  server = TTSHTTPServer((host, port), TTSRequestHandler)
  print(f"[*] TTS Server running on http://{host}:{port} (PID: {os.getpid()})")

  shutdown_event = threading.Event()

  def handle_signal(sig, frame):
    if not shutdown_event.is_set():
      shutdown_event.set()
      print(f"\n[*] Received signal {sig}, shutting down server gracefully...")
      threading.Thread(target=server.shutdown, daemon=True).start()

  signal.signal(signal.SIGINT, handle_signal)
  signal.signal(signal.SIGTERM, handle_signal)

  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
    print("[*] Server shutdown complete.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Local TTS Server")
  parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
  parser.add_argument("--port", type=int, default=5050, help="Port to bind")
  parser.add_argument("--reload", action="store_true", help="Ignored (for CLI compatibility)")
  args = parser.parse_args()
  run_server(host=args.host, port=args.port)
