"""Compare installed Pocket versions with the same English/reference voice inputs.

Run in separate processes using each package version and a model-specific voice
cache. Set HF_HOME to the cache containing the selected model when offline.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pocket_tts import TTSModel  # noqa: E402

from tts.speech_chunking import limit_speech_chunks  # noqa: E402
from tts.text_processor import prepare_speech_chunks  # noqa: E402


SAMPLES = {
  "short": "Artificial intelligence is changing how people work.",
  "technical": (
    "The model reads this sentence clearly, including punctuation and technical words. "
    "The next sentence should follow without breaking the flow."
  ),
  "long": (
    "This is a long sentence that shows how a speech system can stop suddenly in the "
    "middle of a thought when the text has many words and the chunking algorithm "
    "cuts it in an unnatural place before the sentence is finished."
  ),
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", required=True)
  parser.add_argument("--voice-cache", type=Path, required=True)
  parser.add_argument("--language")
  parser.add_argument("--repetitions", type=int, default=3)
  args = parser.parse_args()

  model = TTSModel.load_model(**({"language": args.language} if args.language else {}))
  model.temp = 0.2
  model.eos_threshold = -2.0
  model.noise_clamp = 2.5
  voice_state = model.get_state_for_audio_prompt(str(args.voice_cache.resolve()))
  tokenizer = model.flow_lm.conditioner.tokenizer
  output_dir = ROOT / "benchmark" / "output" / "pocket-comparison" / args.label
  output_dir.mkdir(parents=True, exist_ok=True)

  # Warm the model and voice state before measuring.
  model.generate_audio(voice_state, "This is a warm up sentence.", copy_state=True)
  for name, text in SAMPLES.items():
    parsed = prepare_speech_chunks(text, max_words=1000000, strategy="multi_sentence")
    chunks = limit_speech_chunks(parsed, 26, lambda value: len(tokenizer.encode(value)))
    durations = []
    for _ in range(args.repetitions):
      start = time.perf_counter()
      pieces = []
      for chunk in chunks:
        prompt = chunk.text.strip()
        if prompt[-1] not in ".!?…\"'”’":
          prompt = prompt.rstrip(",;:-–— ") + "."
        if prompt[0].islower():
          prompt = prompt[0].upper() + prompt[1:]
        pieces.append(model.generate_audio(voice_state, prompt, copy_state=True).detach().cpu().numpy().squeeze())
      durations.append(time.perf_counter() - start)
    audio = np.concatenate(pieces)
    path = output_dir / f"{name}.wav"
    sf.write(path, audio, model.sample_rate)
    print(json.dumps({"model": args.label, "sample": name, "chunks": [c.text for c in chunks],
                      "median_generation_s": round(statistics.median(durations), 3),
                      "audio_s": round(len(audio) / model.sample_rate, 3), "audio_path": str(path)}), flush=True)


if __name__ == "__main__":
  main()
