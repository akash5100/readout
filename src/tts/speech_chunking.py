"""Reusable speech chunk boundaries and input filtering."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class SpeechChunk:
  text: str
  pause_after_ms: float = 300.0
  chunk_type: str = "sentence"  # "header", "paragraph", "list_item", "sentence", "clause"


def remove_code_snippets(text: str) -> str:
  """Remove fenced and inline Markdown code before parsing structural blocks."""
  text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1[ \t]*(?:\n|$)", "\n", text)
  text = re.sub(r"(?m)^(?: {4}|\t)[ \t]*\S.*(?:\n|$)", "", text)
  return re.sub(r"`+[^`\n]+`+", " ", text)


def split_sentences(text: str) -> list[str]:
  """Split sentence boundaries without turning common abbreviations into sentences."""
  result = []
  start = 0
  abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e."}
  for match in re.finditer(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’])\s+", text):
    prefix = text[start:match.start()].rstrip()
    last = prefix.split()[-1].strip("\"'“‘()[]") if prefix.split() else ""
    following = text[match.end():]
    if last.lower() in abbreviations or (re.fullmatch(r"(?:[A-Za-z]\.){2,}", last) and following[:1].islower()):
      continue
    if prefix:
      result.append(prefix)
    start = match.end()
  tail = text[start:].strip()
  if tail:
    result.append(tail)
  return result


def limit_speech_chunks(chunks: list[SpeechChunk], max_words: int, token_count=None, max_tokens: int = 45) -> list[SpeechChunk]:
  """Apply limits to every block, including headings and list items.

  Count the final period too, since Pocket adds one to incomplete prompts.
  A single unusually long word stays intact rather than being corrupted.
  """
  output = []
  for chunk in chunks:
    words = chunk.text.split()
    parts = []
    while words:
      end = 0
      for i in range(1, min(len(words), max_words) + 1):
        candidate = " ".join(words[:i])
        prompt = candidate if candidate[-1] in ".!?…" else candidate.rstrip(",;:-–— ") + "."
        if token_count is not None and token_count(prompt) > max_tokens:
          break
        end = i
      if end == 0:
        logger.warning("A single word exceeds the speech token budget: %.40s", words[0])
        end = 1
      if end < len(words):
        # Prefer a nearby natural pause over cutting in the middle of a clause.
        natural = [i for i in range(max(1, end // 2), end + 1) if words[i - 1].endswith((",", ";", ":", "—"))]
        if natural:
          end = natural[-1]
      parts.append(" ".join(words[:end]))
      words = words[end:]
    for i, part in enumerate(parts):
      last = i == len(parts) - 1
      output.append(SpeechChunk(part, chunk.pause_after_ms if last else 100.0, chunk.chunk_type if last else "clause"))
  return output
