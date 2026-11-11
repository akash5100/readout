"""Text normalization, semantic Markdown chunking, and cadence pause budgeting for TTS.

Implements the 3-tier speech preprocessing pipeline:
1. Block & AST parsing (headers, lists, tables, paragraphs)
2. Normalization & technical token sanitization (<ref> -> ref, merge-base -> merge base)
3. Cadence pause budgeting (550ms headers, 400ms paragraphs, 300ms sentences, 120ms clauses)
"""

from __future__ import annotations

import re
import unicodedata


try:
  from num2words import num2words
except ImportError:
  num2words = None

from tts.speech_chunking import SpeechChunk, remove_code_snippets, split_sentences


# Model configurations aligned with research specifications.
# CRITICAL ARCHITECTURAL INVARIANT: DO NOT INCREASE max_words FOR pocket_tts!
# Kyutai Pocket-TTS FlowLM has a strict hard limit of MAX_TOKEN_PER_CHUNK = 50 tokens.
# 25-28 words maps to ~40-48 SentencePiece tokens. Exceeding 50 tokens causes cross-attention
# positional embeddings to collapse, triggering word skipping, phonetic looping, and degenerated speech.
MODEL_CONFIGS = {
  "pocket_tts": {
    "max_words": 26,  # Optimal knapsack packing ceiling: safe < 45 tokens
    "strategy": "multi_sentence",
    "chain_streaming": True,
    "default_voice": "alba",
  },
  "supratts": {
    "max_words": 150,
    "strategy": "multi_sentence",
    "chain_streaming": False,
    "default_voice": None,
  },
  "qwen3": {
    "max_words": 100,
    "strategy": "multi_sentence",
    "chain_streaming": True,
    "default_voice": None,
  },
  "chatterbox": {
    "max_words": 65,
    "strategy": "multi_sentence",
    "chain_streaming": True,
    "default_voice": None,
  },
  "transformers": {
    "max_words": 60,
    "strategy": "multi_sentence",
    "chain_streaming": True,
    "default_voice": None,
  },
}


def get_model_config(model_alias: str) -> dict:
  m = (model_alias or "").lower()
  if "pocket" in m or "kyutai" in m:
    return MODEL_CONFIGS["pocket_tts"]
  elif "qwen" in m:
    return MODEL_CONFIGS["qwen3"]
  elif "supra" in m or "micro" in m:
    return MODEL_CONFIGS["supratts"]
  elif "chatterbox" in m:
    return MODEL_CONFIGS["chatterbox"]
  return MODEL_CONFIGS.get(m, MODEL_CONFIGS["pocket_tts"])




_ONES = [
  "",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALE = ["", "thousand", "million", "billion"]


def _fallback_int_to_words(n: int) -> str:
  if n == 0:
    return "zero"
  if n < 0:
    return "minus " + _fallback_int_to_words(-n)
  parts = []
  chunk_idx = 0
  while n > 0:
    rem = n % 1000
    if rem != 0:
      c = []
      h, t = rem // 100, rem % 100
      if h > 0:
        c.append(_ONES[h] + " hundred")
      if t > 0:
        if t < 20:
          c.append(_ONES[t])
        else:
          c.append(_TENS[t // 10] + ("-" + _ONES[t % 10] if t % 10 > 0 else ""))
      if chunk_idx < len(_SCALE) and _SCALE[chunk_idx]:
        c.append(_SCALE[chunk_idx])
      parts.insert(0, " ".join(c))
    n //= 1000
    chunk_idx += 1
  return ", ".join(parts)


def int_to_words(n: int) -> str:
  if num2words is not None:
    try:
      return num2words(n)
    except Exception:
      pass
  return _fallback_int_to_words(n)


def convert_markdown_tables(text: str) -> str:
  """Converts markdown ASCII tables into conversational spoken row sentences."""
  lines = text.splitlines()
  table_lines = []
  output_lines = []

  def flush_table(tbl: list[str]) -> list[str]:
    if not tbl:
      return []
    rows = []
    for row_line in tbl:
      raw_cells = [c.strip() for c in row_line.strip().strip("|").split("|")]
      # Skip separator line (e.g. :--- | ---:)
      if all(re.match(r"^:?-+:?$", c) for c in raw_cells if c):
        continue
      rows.append(raw_cells)

    if not rows:
      return []

    headers = rows[0]
    data_rows = rows[1:]

    res = []
    if data_rows:
      for row in data_rows:
        parts = []
        for i, val in enumerate(row):
          if not val:
            continue
          header = headers[i] if i < len(headers) and headers[i] else ""
          if i == 0:
            parts.append(val)
          else:
            if header:
              parts.append(f"{header} {val}")
            else:
              parts.append(val)
        res.append(", ".join(parts) + ".")
    else:
      res.append(", ".join(headers) + ".")
    return res

  for line in lines:
    stripped = line.strip()
    if stripped.startswith("|") or (stripped.endswith("|") and "|" in stripped[1:-1]):
      table_lines.append(stripped)
    else:
      if table_lines:
        output_lines.extend(flush_table(table_lines))
        table_lines = []
      output_lines.append(line)

  if table_lines:
    output_lines.extend(flush_table(table_lines))

  return "\n".join(output_lines)


def verbalize_numbers(text: str) -> str:
  """Expands digits, currencies, ordinals, and units into spoken words."""
  if not text:
    return ""

  # 1. Currencies: $50 -> 50 dollars, €20 -> 20 euros, £15 -> 15 pounds, ¥100 -> 100 yen
  def replace_currency(m: re.Match) -> str:
    sym = m.group(1)
    amt_str = m.group(2).replace(",", "")
    try:
      amt = float(amt_str) if "." in amt_str else int(amt_str)
      word = int_to_words(int(amt)) if isinstance(amt, int) else num2words(amt) if num2words else amt_str
      currency_names = {"$": "dollar", "€": "euro", "£": "pound", "¥": "yen"}
      c_name = currency_names.get(sym, "dollar")
      if amt != 1 and c_name != "yen":
        c_name += "s"
      return f"{word} {c_name}"
    except Exception:
      return m.group(0)

  text = re.sub(r"([$€£¥])\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", replace_currency, text)

  # 2. Time: 10:30 or 10:30am/pm
  def replace_time(m: re.Match) -> str:
    hr, mn = int(m.group(1)), int(m.group(2))
    mod = m.group(3) or ""
    try:
      hr_w = int_to_words(hr)
      if mn == 0:
        mn_w = "o'clock" if not mod else ""
      elif mn < 10:
        mn_w = f"oh {int_to_words(mn)}"
      else:
        mn_w = int_to_words(mn)
      res = f"{hr_w} {mn_w}".strip()
      if mod:
        res += " " + " ".join(mod.lower())
      return res
    except Exception:
      return m.group(0)

  text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b", replace_time, text)

  # 3. Soften colons between words: "foo: bar" -> "foo, bar"
  text = re.sub(r"(?<=\w):(?=\s)", ",", text)

  # 4. Ordinals: 1st, 2nd, 3rd, 4th
  if num2words is not None:
    text = re.sub(
      r"\b(\d+)(?:st|nd|rd|th)\b",
      lambda m: num2words(int(m.group(1)), to="ordinal"),
      text,
      flags=re.IGNORECASE,
    )

  # 5. Decimals: 3.14 -> three point one four
  def replace_decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    frac_words = " ".join(int_to_words(int(d)) for d in frac)
    return f"{int_to_words(int(whole))} point {frac_words}"

  text = re.sub(r"\b(\d+)\.(\d+)\b", replace_decimal, text)

  # 6. Standalone integers
  def replace_int(m: re.Match) -> str:
    val = m.group(0).replace(",", "")
    try:
      return int_to_words(int(val))
    except Exception:
      return m.group(0)

  text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", replace_int, text)
  return text


def clean_technical_syntax(text: str) -> str:
  """Sanitizes technical markdown, inline commands, and code placeholders for natural speech."""
  # Expand the common initialism explicitly; leave lowercase "ai" and words such as
  # "AIX" alone. Apply after URL/link removal below so URLs are not altered.
  text = remove_code_snippets(text)

  # 2. Markdown links: [anchor](url) -> anchor
  # CRITICAL: Must be executed before touching parentheses or brackets!
  text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

  # 3. Raw URLs (http, https, file, ftp)
  text = re.sub(r"(?:https?|file|ftp)://\S+", "", text)

  # Filesystem paths are usually distracting in spoken technical prose.
  # Leave standalone filenames alone; only remove slash-separated paths.
  text = re.sub(r"(?<![\w/])(?:/|(?:\.{1,2}/)|(?:~/))?(?:[\w.+~-]+/)+(?:[\w.+~-]+)?", " ", text)

  text = re.sub(r"\bAI\b", "artificial intelligence", text)

  # 4. Markdown formatting: bold, italic, strikethrough
  text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
  text = re.sub(r"\*([^*]+)\*", r"\1", text)
  text = re.sub(r"~~([^~]+)~~", r"\1", text)

  # 5. Angle bracket placeholders: <comparison-ref> -> comparison-ref, <merge-base-sha> -> merge-base-sha
  text = re.sub(r"<([A-Za-z0-9_-]+)>", r"\1", text)

  # Read snake_case and isolated underscores as ordinary word boundaries.
  text = re.sub(r"_+", " ", text)
  text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)

  # 4. Technical compound tokens: merge-base -> merge base, base-branch -> base branch
  # Matches letters followed by hyphen and letters (avoids negative numbers)
  text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)

  # 5. Programming operators & symbols
  text = re.sub(r"\s*&&\s*", " and ", text)
  text = re.sub(r"\s*\|\|\s*", " or ", text)
  text = re.sub(r"\s*!=\s*", " not equal ", text)
  text = re.sub(r"\s*==\s*", " equals ", text)
  text = re.sub(r"\s*>=\s*", " greater than or equal to ", text)
  text = re.sub(r"\s*<=\s*", " less than or equal to ", text)
  text = re.sub(r"\s*->\s*", ", leads to ", text)
  text = re.sub(r"/\*", " folder", text)
  text = re.sub(r"\s*--\s*", ", ", text)
  text = re.sub(r"\.{2,}", " ", text)

  # Function calls with empty/ellipsis arguments: fn(...) -> fn, fn() -> fn
  text = re.sub(r"\s*\(\s*(?:\.{3}|…)?\s*\)", "", text)

  # Closing parenthesis before sentence-terminal punctuation: (foo). -> foo.
  text = re.sub(r"\s*\)\s*(?=[.!?…\n]|$)", "", text)

  # General parentheses: (foo) -> , foo,
  text = re.sub(r"\s*[\(\)]\s*", ", ", text)

  # File extensions: server.py -> server dot py
  text = re.sub(r"\b([A-Za-z0-9_-]+)\.(md|py|json|ts|js|rs|cpp|h|yaml|yml|html|css|txt)\b", r"\1 dot \2", text)
  text = re.sub(r"(?<=\S)\s*\+\s*(?=\S)", " plus ", text)
  text = re.sub(r"\s*&\s*", " and ", text)
  text = re.sub(r"(?<=\d)%", " percent", text)
  text = re.sub(r"%\s*", " percent ", text)

  # Semicolons:
  text = text.replace(";", ",")

  # Clean duplicate commas / trailing punctuation
  text = re.sub(r",\s*,+", ", ", text)
  text = re.sub(r",\s*([.!?…])", r"\1", text)

  return text








def normalize_text_for_speech(text: str) -> str:
  """Backwards-compatible complete text normalizer returning a clean text string."""
  if not text:
    return ""

  text = unicodedata.normalize("NFKC", text)
  text = remove_code_snippets(text)
  text = convert_markdown_tables(text)
  text = clean_technical_syntax(text)

  # Process line by line: handle bullets, arrows, and punctuation
  lines = text.splitlines()
  processed_lines = []
  terminal_punc = set(".!?…:;,")

  for line in lines:
    item = line.strip()
    if not item:
      continue
    # Check headers
    item = re.sub(r"^#{1,6}\s*", "", item).strip()
    is_arrow_start = bool(re.match(r"^(?:--?>|==?>|[→➔➜▶►])\s*", item))
    item = re.sub(r"^(?:--?>|==?>|[→➔➜▶►]|•|\*|–|—|\-|\+)\s*", "", item).strip()
    item = re.sub(r"^(\d+)[\.\)]\s*", r"\1, ", item).strip()
    item = re.sub(r"[^\S\r\n]*(?:--?>|==?>|[→➔➜▶►])[^\S\r\n]*", ", then ", item)
    if not item:
      continue
    if is_arrow_start:
      item = "Then " + item
    elif item[0].islower():
      item = item[0].upper() + item[1:]
    if item[-1] not in terminal_punc:
      item += "."
    processed_lines.append(item)

  result = " ".join(processed_lines)
  result = verbalize_numbers(result)
  result = re.sub(r"\s+", " ", result)
  result = re.sub(r",\s*,+", ",", result)
  result = re.sub(r"\.\s*\.+", ".", result)
  result = re.sub(r",\s*\.", ".", result)
  return result.strip()


def prepare_speech_chunks(
  raw_text: str,
  max_words: int = 28,
  strategy: str = "single_sentence",
) -> list[SpeechChunk]:
  """Parses raw text into speech chunks annotated with semantic boundary pauses.

  Cadence Pause Budget:
  - Header: 450ms (+100ms acoustic trim pad = ~550ms total silence)
  - Paragraph Break: 300ms (+100ms pad = ~400ms total silence)
  - Sentence: 200ms (+100ms pad = ~300ms natural breath pause)
  - List Item: 180ms (+100ms pad = ~280ms)
  - Sub-sentence Clause: 40ms (+100ms pad = ~140ms comma breath)
  """
  if not raw_text or not raw_text.strip():
    return []

  text = unicodedata.normalize("NFKC", raw_text)
  text = remove_code_snippets(text)
  text = convert_markdown_tables(text)

  # Isolate markdown headers (# Header) by ensuring blank line separation
  text = re.sub(r"(^|\n)(#{1,6}\s+[^\n]+)", r"\1\n\2\n\n", text)

  # Split into structural blocks separated by blank lines (paragraphs / sections)
  raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
  chunks: list[SpeechChunk] = []

  for b_idx, block in enumerate(raw_blocks):
    is_last_block = b_idx == len(raw_blocks) - 1

    # 1. Check if block is a Markdown Header (# Header)
    if re.match(r"^#{1,6}\s+", block):
      clean_h = re.sub(r"^#{1,6}\s*", "", block).strip()
      clean_h = clean_technical_syntax(clean_h)
      clean_h = verbalize_numbers(clean_h)
      clean_h = re.sub(r"\s+", " ", clean_h).strip()
      clean_h = clean_h.rstrip(":;, -–—")
      if clean_h:
        if clean_h[-1] not in ".!?…":
          clean_h += "."
        clean_h = clean_h[0].upper() + clean_h[1:] if len(clean_h) > 1 else clean_h.upper()
        chunks.append(SpeechChunk(text=clean_h, pause_after_ms=450.0, chunk_type="header"))
      continue

    # 2. Check if block is a List (bullets or numbered items)
    lines = block.splitlines()
    is_list_block = any(re.match(r"^\s*(?:•|\*|–|—|\-|\+|\d+[\.\)])\s+", ln) for ln in lines)
    if is_list_block:
      for l_idx, line in enumerate(lines):
        clean_l = line.strip()
        if not clean_l:
          continue
        clean_l = re.sub(r"^(?:•|\*|–|—|\-|\+)\s*", "", clean_l).strip()
        clean_l = re.sub(r"^(\d+)[\.\)]\s*", r"\1, ", clean_l).strip()
        clean_l = clean_technical_syntax(clean_l)
        clean_l = verbalize_numbers(clean_l)
        clean_l = re.sub(r"\s+", " ", clean_l).strip()
        clean_l = clean_l.rstrip(":;, -–—")
        if not clean_l:
          continue
        if clean_l[-1] not in ".!?…":
          clean_l += "."
        clean_l = clean_l[0].upper() + clean_l[1:] if len(clean_l) > 1 else clean_l.upper()

        is_last_item = l_idx == len(lines) - 1
        item_pause = 300.0 if (is_last_item and not is_last_block) else 180.0
        chunks.append(SpeechChunk(text=clean_l, pause_after_ms=item_pause, chunk_type="list_item"))
      continue

    # Preserve single line breaks as audible boundaries instead of packing
    # adjacent lines into the same model prompt.
    if len(lines) > 1:
      for line_index, line in enumerate(lines):
        line_chunks = prepare_speech_chunks(line, max_words=max_words, strategy=strategy)
        if line_chunks:
          if line_index < len(lines) - 1:
            line_chunks[-1].pause_after_ms = 250.0
          elif not is_last_block:
            line_chunks[-1].pause_after_ms = 300.0
          chunks.extend(line_chunks)
      continue

    # 3. Standard Paragraph Text
    clean_block = clean_technical_syntax(block)
    # Process lines: handle arrows and line-break punctuation so diagrams/steps don't run on
    lines = clean_block.splitlines()
    processed_lines = []
    terminal_punc = set(".!?…:;,")

    for line in lines:
      item = line.strip()
      if not item:
        continue
      is_arrow_start = bool(re.match(r"^(?:--?>|==?>|[→➔➜▶►])\s*", item))
      item = re.sub(r"^(?:--?>|==?>|[→➔➜▶►])\s*", "", item).strip()
      item = re.sub(r"[^\S\r\n]*(?:--?>|==?>|[→➔➜▶►])[^\S\r\n]*", ", then ", item)
      if not item:
        continue
      if is_arrow_start:
        item = "Then " + item
      elif item[0].islower():
        item = item[0].upper() + item[1:]
      if item[-1] not in terminal_punc:
        item += "."
      processed_lines.append(item)

    clean_block = " ".join(processed_lines)
    clean_block = verbalize_numbers(clean_block)
    clean_block = re.sub(r"\s+", " ", clean_block).strip()

    # Split into sentences preserving quotes
    sentences = split_sentences(clean_block)
    if not sentences:
      sentences = [clean_block]

    if strategy == "single_sentence":
      for s_idx, sent in enumerate(sentences):
        is_last_sent = s_idx == len(sentences) - 1
        sent_words = sent.split()

        if len(sent_words) <= max_words:
          if sent[-1] not in ".!?…\"'”’":
            sent += "."
          sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
          pause = 300.0 if (is_last_sent and not is_last_block) else 200.0
          chunks.append(SpeechChunk(text=sent, pause_after_ms=pause, chunk_type="sentence"))
        else:
          # Sub-split oversized sentences on clauses to prevent token stalls
          clauses = [c.strip() for c in re.split(r"(?<=[,;:\u2014])\s+", sent) if c.strip()]
          cur_parts, cur_cnt = [], 0
          for cl in clauses:
            cl_words = cl.split()
            # CRITICAL FALLBACK: If an individual unpunctuated clause itself exceeds max_words,
            # slice it by max_words so it can NEVER exceed the Pocket-TTS 50-token window.
            if len(cl_words) > max_words:
              if cur_parts:
                part_text = " ".join(cur_parts).strip()
                part_text = part_text.rstrip(",;:-–— ")
                if part_text and part_text[-1] not in ".!?…\"'”’":
                  part_text += "."
                if part_text and part_text[0].islower():
                  part_text = part_text[0].upper() + part_text[1:]
                chunks.append(SpeechChunk(text=part_text, pause_after_ms=140.0, chunk_type="clause"))
                cur_parts, cur_cnt = [], 0

              for w_idx in range(0, len(cl_words), max_words):
                sub_slice = cl_words[w_idx : w_idx + max_words]
                sub_text = " ".join(sub_slice).strip()
                sub_text = sub_text.rstrip(",;:-–— ")
                if sub_text and sub_text[-1] not in ".!?…\"'”’":
                  sub_text += "."
                if sub_text and sub_text[0].islower():
                  sub_text = sub_text[0].upper() + sub_text[1:]
                chunks.append(SpeechChunk(text=sub_text, pause_after_ms=140.0, chunk_type="clause"))
              continue

            if cur_cnt + len(cl_words) <= max_words:
              cur_parts.append(cl)
              cur_cnt += len(cl_words)
            else:
              if cur_parts:
                part_text = " ".join(cur_parts).strip()
                part_text = part_text.rstrip(",;:-–— ")
                if part_text and part_text[-1] not in ".!?…\"'”’":
                  part_text += "."
                if part_text and part_text[0].islower():
                  part_text = part_text[0].upper() + part_text[1:]
                chunks.append(SpeechChunk(text=part_text, pause_after_ms=140.0, chunk_type="clause"))
              cur_parts = [cl]
              cur_cnt = len(cl_words)

          if cur_parts:
            part_text = " ".join(cur_parts).strip()
            part_text = part_text.rstrip(",;:-–— ")
            if part_text and part_text[-1] not in ".!?…\"'”’":
              part_text += "."
            if part_text and part_text[0].islower():
              part_text = part_text[0].upper() + part_text[1:]
            pause = 300.0 if (is_last_sent and not is_last_block) else 200.0
            chunks.append(SpeechChunk(text=part_text, pause_after_ms=pause, chunk_type="sentence"))

    else:
      # multi_sentence strategy: Optimal Greedy Bin Packing with Clause Fallback
      # 1. Pre-process sentences: normalize punctuation/casing and sub-split oversized sentences by clauses
      atomic_units = []
      for sent in sentences:
        s = sent.strip()
        if not s:
          continue
        if s[-1] not in ".!?…\"'”’":
          s += "."
        s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()

        s_words = s.split()
        if len(s_words) <= max_words:
          atomic_units.append(s)
        else:
          # Sub-split oversized sentences on clauses
          clauses = [c.strip() for c in re.split(r"(?<=[,;:\u2014])\s+", s) if c.strip()]
          cur_cl = []
          cur_cnt = 0
          for cl in clauses:
            cl_words = cl.split()
            # If an individual unpunctuated clause itself exceeds max_words, slice it by max_words
            if len(cl_words) > max_words:
              if cur_cl:
                part = " ".join(cur_cl).strip().rstrip(",;:-–— ")
                if part and part[-1] not in ".!?…\"'”’":
                  part += "."
                if part and part[0].islower():
                  part = part[0].upper() + part[1:]
                atomic_units.append(part)
                cur_cl, cur_cnt = [], 0

              for w_idx in range(0, len(cl_words), max_words):
                sub_slice = cl_words[w_idx : w_idx + max_words]
                sub_text = " ".join(sub_slice).strip().rstrip(",;:-–— ")
                if sub_text and sub_text[-1] not in ".!?…\"'”’":
                  sub_text += "."
                if sub_text and sub_text[0].islower():
                  sub_text = sub_text[0].upper() + sub_text[1:]
                atomic_units.append(sub_text)
              continue

            if cur_cnt + len(cl_words) <= max_words:
              cur_cl.append(cl)
              cur_cnt += len(cl_words)
            else:
              if cur_cl:
                part = " ".join(cur_cl).strip().rstrip(",;:-–— ")
                if part and part[-1] not in ".!?…\"'”’":
                  part += "."
                if part and part[0].islower():
                  part = part[0].upper() + part[1:]
                atomic_units.append(part)
              cur_cl = [cl]
              cur_cnt = len(cl_words)

          if cur_cl:
            part = " ".join(cur_cl).strip().rstrip(",;:-–— ")
            if part and part[-1] not in ".!?…\"'”’":
              part += "."
            if part and part[0].islower():
              part = part[0].upper() + part[1:]
            atomic_units.append(part)

      # 2. Greedy Bin Packing: Pack adjacent complete sentences/clauses up to max_words
      cur_chunk_units = []
      cur_count = 0

      for unit in atomic_units:
        u_words = unit.split()
        if cur_count + len(u_words) <= max_words:
          cur_chunk_units.append(unit)
          cur_count += len(u_words)
        else:
          if cur_chunk_units:
            pack_text = " ".join(cur_chunk_units).strip()
            chunks.append(SpeechChunk(text=pack_text, pause_after_ms=200.0, chunk_type="sentence_pack"))
          cur_chunk_units = [unit]
          cur_count = len(u_words)

      if cur_chunk_units:
        pack_text = " ".join(cur_chunk_units).strip()
        pause = 300.0 if not is_last_block else 200.0
        chunks.append(SpeechChunk(text=pack_text, pause_after_ms=pause, chunk_type="paragraph" if not is_last_block else "sentence_pack"))

  return chunks
