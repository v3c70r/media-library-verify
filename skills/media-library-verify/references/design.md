# Design and tuning

## Why tiers

Verifying a large library with a vision model on every frame is slow and unreliable.
Instead, push work down to the cheapest tier that can answer the question, and only
escalate ambiguous cases.

```
T0 metadata      ffprobe every file                          ~free        seconds
T1 keyframes     extract I-frames only (no full decode)       cheap        ~3s/episode
T2 local score   color-histogram outliers + blank detector    cheap        ~1s/episode
T3 VLM           title OCR + captions on selected frames      expensive    seconds/frame
```

## T0 — metadata

Catches corruption, truncated downloads, missing audio, wrong codecs, resolution
downgrades, and filenames without an `SxxEe` marker. Tune the expected ranges with
`--min-duration`, `--max-duration`, `--min-width`, and `--lang`. This tier alone finds
most structural problems.

## T1 — keyframe sampling as a camera-cut proxy

`verify_media.py` lists keyframe timestamps from packet headers
(`ffprobe -show_entries packet=...`) — this does **not** decode the video and is nearly
free. It then extracts those frames with `ffmpeg -skip_frame nokey`, which decodes only
I-frames.

For a 1080p HEVC TV episode this yields roughly 60–150 frames and takes ~3 s, versus
~20–25 s for full scene detection (`select='gt(scene,...)'`) that walks every frame.
Encoders place I-frames on cuts, so the samples land on scene changes — a good
approximation of "one frame per camera cut".

Byproduct: the same frames feed T3's title-card OCR, so no extra decoding is needed for
identity checks in the common case.

## T2 — local palette outlier detection

For every keyframe:

- resize to 32×32, compute an HSV histogram (12 hue × 4 sat × 4 val = 192 bins),
  normalized;
- compute mean brightness and contrast.

A reference "collection palette" is the **median** histogram across all frames — robust
because the genuine content dominates. Each frame's cosine distance to that reference is
turned into a robust z-score (median absolute deviation). Frames above `--zcut`
(default 6.0) are flagged, plus near-black/low-contrast frames (fade-in/out, transitions).

What this catches: title cards, black frames, night scenes, unrelated footage with a
different palette, inserted ads, test patterns. What it misses: foreign clips that share
the show's palette. That gap is covered by the random `--audit` captions in T3.

## T3 — VLM, used narrowly

Two jobs only:

1. **Title-card OCR.** Pick the flattest, brightest frames in the opening
   `[25 s, 150 s]` window (title cards are plain backgrounds with centered text), OCR
   them, and fuzzy-match against the filename title. Fast path uses cached keyframes;
   episodes that don't match get a 1 fps fallback extraction limited to that window.
2. **Captions** for flagged frames plus `--audit` random frames. Captions are advisory:
   they are scanned for red-flag phrases (test pattern, live action, real photo,
   adult/violent/disturbing terms) and otherwise kept for human review.

### Important lesson: small VLMs are not classifiers

MiniCPM-V 4.6 is a good captioner and OCR engine, but when asked to emit forced labels
such as `BLUEY / OTHER / BLANK` it echoes the template, contradicts itself, or answers
"BLUEY" for a fully black frame. Multi-image classification prompts were worse than
single-image ones. Therefore this pipeline never relies on the VLM for an automatic
verdict; it uses OCR (reliable) and open-ended captions (reliable), and leaves the
decision to T0/T2 plus human review of short lists.

## Tuning checklist

| Symptom | Knob |
|---|---|
| Too few frames reviewed | lower `--zcut`, raise `--audit`, raise `--maxflag` |
| Too much noise in outliers | raise `--zcut` |
| Title cards not found | widen the window in `onefps()` / raise candidates in `low_energy()` |
| Endpoint rate limits | lower `--jobs` |
| Reruns during tuning | keep `--resume` and point `--workdir` at a stable path |

## Reference result (Bluey S1–S3, 154 episodes)

- T0: 0 metadata issues; complete seasons (52/52/50).
- T1: 14,211 keyframes sampled.
- T2: 688 outliers flagged (title cards, transitions, night scenes).
- T3: 149/154 title cards matched filenames; remaining ones were OCR/candidate misses,
  not content problems (confirmed by full-window OCR and visual inspection).
- No foreign or unsafe content found. Full run with cached keyframes: ~6–7 minutes.
