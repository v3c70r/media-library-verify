---
name: media-library-verify
description: >
  Verify a downloaded video library (season folders, episode packs, archives) actually
  contains the claimed content and is safe to keep/watch. Tiered pipeline: ffprobe
  metadata checks, keyframe ("camera cut") sampling, local color/blank outlier detection,
  episode-title OCR, and optional VLM captions. Use when a user wants to check a torrent/
  download for fakes, wrong episodes, mislabeled files, corruption, or inappropriate
  content before watching or sharing — especially kids' shows. Also use to verify media
  completeness (missing episodes, wrong seasons, inconsistent encoding).
---

# Media Library Verify

Verify that downloaded episodes/seasons are the claimed content and safe. Do not trust
filenames: check container metadata, sample the actual video, and (when a vision model is
available) OCR episode title cards and caption suspicious frames.

## When to use

- User downloaded a show/movie pack (torrent, DDL archive) and wants to know if it is genuine.
- "Is this really <show>?", "did I get the full season?", "is it safe for kids?"
- Checking for swapped/mislabeled episodes, wrong seasons, corrupt/truncated files,
  resolution/codec mismatches, missing audio, or inserted foreign footage.

## Prerequisites

- `ffmpeg` and `ffprobe` on PATH.
- Python 3 with `numpy` and `Pillow` (`pip install numpy pillow`).
- Optional: any OpenAI-compatible vision endpoint (llama.cpp `llama-server`, vLLM, etc.).
  Small VLMs (MiniCPM-V, Qwen2-VL) work; they are used only for OCR + captions.

## Workflow

1. **Locate the library root** (the folder holding the season subfolders).
2. **Run the pipeline** from this skill directory:

   ```bash
   python3 scripts/verify_media.py --root "/path/to/library" \
       --show "Bluey" \
       --vlm "$VLM_URL" --key "$VLM_KEY" --model "$VLM_MODEL" \
       --jobs 8
   ```

   - No model available? Use `--no-vlm` (metadata + local outlier detection only).
   - Long jobs: add `--resume` to reuse cached keyframes in the workdir.
   - Tunables: `--maxflag`, `--audit`, `--zcut`, `--width`, `--min-duration`,
     `--max-duration`, `--min-width`, `--lang`.
3. **Read `verify_report.json`** (written to the library root) and interpret:
   - `metadata_issues` — unreadable files, odd duration, no audio, wrong codec/res, missing `SxxEe`.
   - `title_mismatch` / `title_missing` — episode identity could not be confirmed from the title card.
   - `local_outlier_frames` — visually unusual frames (title cards, scene text, night scenes,
     transitions, or genuinely foreign content).
   - `vlm_flagged` — captions that mention test patterns, live action, or unsafe content.
4. **Judge, don't over-trust the VLM.** Small VLMs hallucinate and echo templates on forced
   classification. Trust the local outlier list + title OCR. For any `vlm_flagged` or
   `title_mismatch` entry, open the listed keyframe images (paths are in the report) and look
   yourself before declaring a problem.
5. **Report back** with: file count vs expected, season/episode coverage, metadata problems,
   identity confirmations, and a short list of frames/offsets worth a human glance.

## Interpretation guide

| Signal | Likely meaning |
|---|---|
| `metadata_issues` empty, per-season counts match the show | Complete, consistent release |
| Outlier frames are only title cards / black transitions / uncommon palettes | Normal |
| Outlier frame caption names another show, "test pattern", or "real photo" | Investigate that timestamp |
| Many `title_missing` | Title cards may be later than the sampling window, or OCR failed — not automatically bad |
| `title_mismatch` with a plausible title of the *next* episode | Filenames may be shifted; check ordering |
| Duration far from the rest | Truncated file, or a legit special/finale |

## Notes and limits

- Keyframe sampling approximates camera cuts. It is far cheaper than full scene detection
  but can miss content between two long keyframe intervals. Raise coverage with
  `--zcut`-independent means: lower `--zcut` flags more frames for review, and a second run
  with a different `--width`/random seed re-samples the audit set.
- The local detector finds *palette* outliers. Foreign footage with a similar palette may
  pass; the random `--audit` captions are the safety net. Increase `--audit` for untrusted
  sources.
- Everything runs locally except the optional VLM calls. Point `--vlm` at a local server
  (e.g. `llama-server`) to keep the media private.

See `references/design.md` for the tier design and tuning rationale.
