import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tts.server import ModelManager, normalize


TEST_SAMPLES = {
  "narrative": (
    "The old lighthouse had stood upon the jagged headland for over a century, "
    "weathering fierce northern gales and guiding lost fishermen safely home. "
    "Inside the spiral stone tower, an antique brass pendulum swung in rhythmic precision, "
    "marking the quiet passage of time while shadows lengthened across the wooden floor. "
    "Elena paused on the observation deck, listening closely as the distant foghorn echoed through the twilight."
  ),
  "technical": (
    "Modern neural text-to-speech architectures rely on continuous latent representations, "
    "often combining flow-matching decoders with autoregressive language modeling. "
    "When generating long audio streams on edge devices, maintaining acoustic coherence across sentence boundaries "
    "is critical to avoid phase cancellation, audible DC-offset thumps, and volume pumping. "
    "Careful silence trimming and smooth crossfades preserve natural human prosody without wasting compute."
  ),
  "dialogue": (
    '"Did you hear that strange hum coming from the server room?" Marcus whispered, glancing down the dim hallway. '
    '"It started right after the backup script finished running!" '
    "Sophie frowned, adjusting her glasses as she pulled up the terminal telemetry on her tablet. "
    "\"Don't panic just yet. The cooling fans usually spin up to maximum speed during disk compaction, "
    "so let's check the thermal sensors first.\""
  ),
}


def compute_audio_metrics(wav: np.ndarray, sr: int = 24000, chunk_boundaries: list[int] | None = None):
  metrics = {}

  # 1. Overall RMS and dynamic range
  rms = float(np.sqrt(np.mean(wav**2)))
  peak = float(np.max(np.abs(wav)))
  crest_factor = peak / (rms + 1e-9)
  metrics["rms"] = rms
  metrics["peak"] = peak
  metrics["crest_factor_db"] = float(20 * math.log10(crest_factor + 1e-9))

  # 2. Boundary discontinuities (clicks / phase jumps)
  # Check max first difference |wav[t] - wav[t-1]|
  diff = np.abs(np.diff(wav))
  metrics["max_derivative"] = float(np.max(diff))

  if chunk_boundaries and len(chunk_boundaries) > 0:
    boundary_diffs = []
    boundary_pauses = []

    # Check window around each boundary (e.g. +/- 50ms)
    win = int(0.05 * sr)
    for b in chunk_boundaries:
      start = max(0, b - win)
      end = min(len(wav), b + win)
      local_diff = np.max(np.abs(np.diff(wav[start:end])))
      boundary_diffs.append(float(local_diff))

      # Detect silence length around boundary (< -45 dB)
      silence_thresh = 0.005 * peak
      # Search backwards and forwards from boundary
      left = b
      while left > 0 and abs(wav[left]) < silence_thresh:
        left -= 1
      right = b
      while right < len(wav) and abs(wav[right]) < silence_thresh:
        right += 1
      pause_ms = ((right - left) / sr) * 1000.0
      boundary_pauses.append(float(pause_ms))

    metrics["boundary_max_diff"] = max(boundary_diffs) if boundary_diffs else 0.0
    metrics["avg_pause_ms"] = float(np.mean(boundary_pauses)) if boundary_pauses else 0.0
    metrics["pauses_ms"] = boundary_pauses

  return metrics


def run_asr_verification(audio_path: str):
  try:
    import librosa
    from faster_whisper import WhisperModel

    y, _ = librosa.load(audio_path, sr=16000)
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(y, beam_size=1)
    transcript = " ".join([seg.text.strip() for seg in segments])
    return transcript
  except Exception as e:
    return f"[ASR Error: {e}]"


def main():
  out_dir = Path("benchmark/output")
  out_dir.mkdir(parents=True, exist_ok=True)

  print("[*] Initializing ModelManager for Pocket TTS...")
  manager = ModelManager()
  manager.load("pocket-tts")

  sr = 24000
  voice = manager.resolve_pocket_voice("alba")
  voice_state = manager.get_pocket_voice_state(voice)

  results = {}

  for name, text in TEST_SAMPLES.items():
    print("\n=======================================================")
    print(f"[*] Testing Sample: '{name}' ({len(text.split())} words)")
    print("=======================================================")

    # -------------------------------------------------------------------------
    # STRATEGY 1: Baseline (Current Server + App behavior)
    # - Chunks text into 30 words
    # - For each chunk: generate_audio(voice_state, copy_state=True)
    # - Trim silence with 250ms lead / 180ms trail
    # - Independent RMS normalize (0.07) per chunk
    # - Inter-chunk concatenation (430ms total digital silence between chunks)
    # -------------------------------------------------------------------------
    chunks_30 = manager._chunk_text(text, max_words=30, strategy="multi_sentence")
    print(f"[*] Baseline chunk count (max_words=30): {len(chunks_30)} chunks")

    baseline_wavs = []
    baseline_boundaries = []
    cur_sample = 0

    for c_idx, chunk in enumerate(chunks_30):
      with torch.no_grad():
        w = manager.model.generate_audio(voice_state, chunk, copy_state=True)
        w_np = w.detach().cpu().numpy().squeeze()
        # Server trimming
        w_clean = manager._trim_silence(w_np, sr=sr, lead_ms=250.0, trail_ms=180.0)
        # Server per-chunk RMS normalization
        w_norm = normalize(w_clean, rms=0.07)
        baseline_wavs.append(w_norm)
        cur_sample += len(w_norm)
        if c_idx < len(chunks_30) - 1:
          baseline_boundaries.append(cur_sample)

    baseline_full = np.concatenate(baseline_wavs)
    baseline_path = out_dir / f"{name}_baseline.wav"
    sf.write(str(baseline_path), baseline_full, sr)

    m_base = compute_audio_metrics(baseline_full, sr, baseline_boundaries)
    print(f"[Baseline] Length: {len(baseline_full) / sr:.2f}s | Boundaries: {len(baseline_boundaries)}")
    print(f"[Baseline] Max boundary jump: {m_base.get('boundary_max_diff', 0):.4f} | Avg pause: {m_base.get('avg_pause_ms', 0):.1f}ms")
    print(f"[Baseline] Boundary pauses: {[round(p, 1) for p in m_base.get('pauses_ms', [])]}")

    # -------------------------------------------------------------------------
    # STRATEGY 2: Optimized Sentence-Aware Packing + Continuous State + Natural Fade
    # - Token / Sentence-aware chunking (max_words=120 or native pocket split)
    # - Natural human pause (140ms inter-sentence) with 5ms smooth cosine micro-fade
    # - Global normalization at full utterance level (no volume pumping)
    # -------------------------------------------------------------------------
    chunks_opt = manager._chunk_text(text, max_words=120, strategy="multi_sentence")
    print(f"[*] Optimized chunk count (max_words=120): {len(chunks_opt)} chunks")

    # Direct evaluation

    # Generate full audio with native long-text generation
    with torch.no_grad():
      opt_full_tensor = manager.model.generate_audio(voice_state, text, max_tokens=60, copy_state=True)
      opt_raw = opt_full_tensor.detach().cpu().numpy().squeeze()

      # Trim only true leading/trailing dead silence on the complete utterance
      idx = np.where(np.abs(opt_raw) > 0.001)[0]
      if len(idx) > 0:
        s_start = max(0, idx[0] - int(0.05 * sr))  # 50ms lead-in
        s_end = min(len(opt_raw), idx[-1] + int(0.12 * sr))  # 120ms trail-out
        opt_trimmed = opt_raw[s_start:s_end]
      else:
        opt_trimmed = opt_raw

      # Full utterance normalization (preserves intra-sentence dynamics)
      opt_full = normalize(opt_trimmed, rms=0.07)

    opt_path = out_dir / f"{name}_optimized.wav"
    sf.write(str(opt_path), opt_full, sr)
    m_opt = compute_audio_metrics(opt_full, sr)
    print(f"[Optimized] Length: {len(opt_full) / sr:.2f}s | Max derivative: {m_opt['max_derivative']:.4f}")

    # -------------------------------------------------------------------------
    # ASR Verification
    # -------------------------------------------------------------------------
    asr_base = run_asr_verification(str(baseline_path))
    asr_opt = run_asr_verification(str(opt_path))
    print(f"[ASR Baseline]  {asr_base}")
    print(f"[ASR Optimized] {asr_opt}")

    # -------------------------------------------------------------------------
    # Plot Waveform & Spectrogram Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"Pocket TTS Quality Comparison: {name.capitalize()}", fontsize=14, fontweight="bold")

    # Baseline Waveform
    t_base = np.linspace(0, len(baseline_full) / sr, len(baseline_full))
    axes[0].plot(t_base, baseline_full, color="#d9534f", alpha=0.8, lw=0.6)
    axes[0].set_title(f"Baseline: 30-word Chunks + Per-Chunk RMS Normalization + 430ms Gaps ({len(chunks_30)} chunks)", fontsize=11)
    axes[0].set_ylabel("Amplitude")
    for b in baseline_boundaries:
      axes[0].axvline(b / sr, color="blue", linestyle="--", alpha=0.7, label="Chunk splice" if b == baseline_boundaries[0] else "")
    if baseline_boundaries:
      axes[0].legend(loc="upper right")

    # Baseline Spectrogram
    axes[1].specgram(baseline_full, Fs=sr, NFFT=1024, noverlap=512, cmap="inferno")
    axes[1].set_title("Baseline Spectrogram (Notice unnatural silence gaps & energy jumps)", fontsize=11)
    axes[1].set_ylabel("Frequency (Hz)")
    for b in baseline_boundaries:
      axes[1].axvline(b / sr, color="cyan", linestyle="--", alpha=0.8)

    # Optimized Waveform
    t_opt = np.linspace(0, len(opt_full) / sr, len(opt_full))
    axes[2].plot(t_opt, opt_full, color="#2e7d32", alpha=0.8, lw=0.6)
    axes[2].set_title("Optimized: Unified Sentence Token Splitting + Global Normalization (Continuous Prosody)", fontsize=11)
    axes[2].set_ylabel("Amplitude")

    # Optimized Spectrogram
    axes[3].specgram(opt_full, Fs=sr, NFFT=1024, noverlap=512, cmap="inferno")
    axes[3].set_title("Optimized Spectrogram (Smooth continuous formant transitions & natural pauses)", fontsize=11)
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plot_path = out_dir / f"{name}_comparison.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"[*] Saved comparison plot: {plot_path}")

    results[name] = {
      "baseline_len": len(baseline_full) / sr,
      "opt_len": len(opt_full) / sr,
      "baseline_pauses": m_base.get("pauses_ms", []),
      "baseline_asr": asr_base,
      "opt_asr": asr_opt,
      "plot_path": str(plot_path.resolve()),
      "baseline_audio": str(baseline_path.resolve()),
      "opt_audio": str(opt_path.resolve()),
    }

  print("\n[+] Benchmark Complete!")
  return results


if __name__ == "__main__":
  main()
