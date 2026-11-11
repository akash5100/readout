"""Regression cases for the text that reaches local speech models."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tts.speech_chunking import limit_speech_chunks, split_sentences
from tts.text_processor import normalize_text_for_speech, prepare_speech_chunks


def chunks(text, max_words=26, token_count=None, max_tokens=45):
  parsed = prepare_speech_chunks(text, max_words=1000000, strategy="multi_sentence")
  return limit_speech_chunks(parsed, max_words, token_count, max_tokens)


def test_sentence_split_keeps_abbreviations_inside_sentence():
  assert split_sentences("Dr. Smith lives in the U.S. and works remotely. Next sentence.") == [
    "Dr. Smith lives in the U.S. and works remotely.",
    "Next sentence.",
  ]


def test_long_sentence_preserves_words_and_marks_continuation():
  text = "This long sentence has several words before a natural pause, and continues with a few more words until the thought is complete."
  result = chunks(text, max_words=10)
  assert result[0].text.endswith(",")
  assert result[0].pause_after_ms == 100
  assert " ".join(c.text for c in result) == text
  assert all(len(c.text.split()) <= 10 for c in result)


def test_limits_apply_to_headings_lists_and_token_heavy_text():
  for prefix in ("# ", "- ", ""):
    text = prefix + " ".join(["electroencephalography"] * 12)
    result = chunks(text, max_words=26, token_count=lambda s: sum(len(w) for w in s.split()), max_tokens=100)
    assert len(result) > 1
    assert all(sum(len(w) for w in c.text.split()) <= 100 for c in result)
    assert sum(len(c.text.split()) for c in result) == 12


def test_pronunciation_paths_and_code_are_filtered_before_chunking():
  text = ('AI helps read /Users/akash/file.py and path/to/file.py.\n\n'
          '```python\nprint("should not be spoken")\n\nx = 1\n```\n\n'
          'Keep this sentence, but skip `git status`.')
  spoken = " ".join(c.text for c in chunks(text))
  assert "artificial intelligence" in spoken.lower()
  assert "should not be spoken" not in spoken and "git status" not in spoken
  assert "Users" not in spoken and "path/to" not in spoken
  assert "Keep this sentence" in spoken
  assert "artificial intelligence" in normalize_text_for_speech("AI works.").lower()
  assert normalize_text_for_speech("AIX and ai work.").startswith("AIX and ai")


def test_links_keep_label_and_code_only_input_is_empty():
  assert "documentation" in normalize_text_for_speech("Read [documentation](https://example.com/path).")
  assert prepare_speech_chunks("```python\nprint('hello')\n```", 26, "multi_sentence") == []
  assert "print" not in normalize_text_for_speech("Example:\n\n    print('hello')\n\nDone.")


def test_underscores_become_spaces_in_spoken_text():
  text = "Read foo_bar, __init__, and _ _ things."
  expected = "Read foo bar, init, and things."
  assert normalize_text_for_speech(text) == expected
  assert " ".join(chunk.text for chunk in chunks(text)) == expected
  assert "path" not in normalize_text_for_speech("Skip /path/to/file_name.py please.").lower()


def test_line_breaks_create_a_pause_without_changing_list_boundaries():
  result = chunks("First line.\nSecond line.\nThird line.")
  assert [c.text for c in result] == ["First line.", "Second line.", "Third line."]
  assert [c.pause_after_ms for c in result] == [250.0, 250.0, 200.0]
  items = chunks("- First item\n- Second item")
  assert [c.pause_after_ms for c in items] == [180.0, 180.0]
