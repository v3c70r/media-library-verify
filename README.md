# media-library-verify

[![CI](https://github.com/v3c70r/media-library-verify/actions/workflows/ci.yml/badge.svg)](https://github.com/v3c70r/media-library-verify/actions/workflows/ci.yml)

A [Pi](https://pi.dev) skill/package that verifies a downloaded video library really is
the claimed content and is safe to keep — without trusting filenames.

It combines cheap local checks with optional vision-model calls in tiers, so expensive
model time is only spent where it helps.

## What it checks

- **Structure / integrity (T0):** ffprobe every file — container, duration, video/audio
  streams, codec, resolution, audio language, `SxxEe` naming, season/episode coverage.
- **Content sampling (T1):** extracts encoder keyframes (a cheap camera-cut proxy) as
  small JPEGs — no full decode for detection.
- **Local outlier detection (T2):** builds a robust "collection palette" from all frames
  and flags palette outliers and blank/black frames. Runs fully offline.
- **Vision checks (T3, optional):** OCRs episode title cards and fuzzy-matches them to the
  filename; captions flagged and randomly audited frames for human review.

It writes `verify_report.json` next to the media and prints a summary of metadata issues,
outlier frames, title mismatches, and anything a human should glance at.

## Install

As a Pi package (installs the bundled skill):

```bash
pi install git:github.com/<you>/media-library-verify
```

Or just copy `skills/media-library-verify/` into `~/.pi/agent/skills/` (or
`~/.agents/skills/`).

## Requirements

- `ffmpeg` + `ffprobe` on `PATH`
- Python 3.9+ with `numpy` and `Pillow`
- Optional: any OpenAI-compatible vision endpoint (llama.cpp `llama-server`, vLLM, ...)

## Usage

```bash
python3 skills/media-library-verify/scripts/verify_media.py \
    --root "/path/to/library" \
    --show "Bluey" \
    --vlm http://localhost:8080/v1/chat/completions \
    --model openbmb/MiniCPM-V-4.6-gguf \
    --jobs 8

# offline: metadata + local outlier detection only
python3 .../verify_media.py --root "/path/to/library" --no-vlm
```

Common flags: `--audit` (random frames captioned), `--maxflag` (flagged frames captioned),
`--zcut` (outlier threshold), `--width` (keyframe size), `--resume` (reuse cached
keyframes), `--min-duration` / `--max-duration` / `--min-width` / `--lang`.

See [`skills/media-library-verify/SKILL.md`](skills/media-library-verify/SKILL.md) and
[`references/design.md`](skills/media-library-verify/references/design.md) for the full
methodology, interpretation guide, and tuning.

## Design note

Small VLMs are good captioners/OCR engines but poor at forced classification. This tool
only uses the model for OCR and open-ended captions; the automated verdict comes from
metadata checks and local outlier detection, with the model output kept as evidence.

## License

MIT
