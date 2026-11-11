# readout

A small, local text-to-speech app for macOS. Select text in another app and click the floating button to hear it, or type into readout directly. It uses [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) for local speech generation and has a simple menu bar app, voice selection, and playback controls.

This was mostly AI-generated, with my direction and iteration for my own use. I'm sharing it as-is in case it's useful or interesting; I don't plan to maintain or support it. Feel free to adapt it for yourself.

## Getting started

This is a personal project, not a packaged macOS app. It was developed on Apple Silicon and needs some setup:

- macOS 14 or newer, Xcode Command Line Tools, Python 3.12+, and [uv](https://docs.astral.sh/uv/).
- Pocket TTS uses its built-in Alba voice by default. You can select or add a local reference voice in the app. Model files may download on first use.
- Accessibility permission for the selection overlay, so readout can detect text selected in other apps.

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/akash5100/readout.git
cd readout
uv sync
```

The launcher currently has a Python path specific to my machine. Edit `PYTHON` near the top of `tts` to point to the Python in your environment (for example, the full path to this checkout's `.venv/bin/python`). Then build and start the native app:

```bash
cd native/AuraTTS
swift build -c release
cd ../..
./tts
```

Open readout from the menu bar and set the repository folder in Settings to your checkout. Select text in another app and click the button that appears. During playback, the control moves to the bottom of the screen; it disappears when playback finishes or you select different text.

```bash
./tts speak "Hello, this is spoken locally."
./tts speak --status JOB_ID
./tts speak --wait "Wait until this finishes."
./tts status
./tts stop
```

`tts speak` returns a job ID immediately and queues requests in order. Its duration is an estimate; `--status` reports the actual audio duration after playback finishes.

## Speaking from an agent

Make `tts` available on your `PATH` (for example, link this checkout’s `tts` launcher into `~/.local/bin`), then add this to your agent’s global instructions, such as `~/.codex/AGENTS.md` or `~/.gemini/GEMINI.md`:

```markdown
## Speech

Give brief spoken notifications for meaningful progress, completion, or when user input is needed. Run `tts speak "..."` using the selected model and voice; requests queue and return immediately. Keep spoken notifications short and avoid repeating every detail.
```

The setup is tailored to my machine and hasn't been verified on a fresh checkout. You may need to adjust paths or dependencies. Voice samples and model caches are not included, and other models exposed by the interface may require additional setup.

## License

[MIT](LICENSE)
