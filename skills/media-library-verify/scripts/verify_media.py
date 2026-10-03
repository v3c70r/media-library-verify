#!/usr/bin/env python3
"""
Tiered verifier for downloaded media libraries (episode folders, seasons, archives).

Goal: cheaply flag files/frames that are NOT the claimed content or are unsafe,
then spend expensive VLM calls only where they add value.

Tiers
-----
T0 Metadata    : ffprobe each file -> container/stream/duration/language checks. (free)
T1 Sampling    : extract encoder keyframes as small JPEGs. Keyframe timestamps come
                 from packet headers (no decode); only keyframes are decoded, so this
                 is ~50-100x cheaper than full scene detection while still giving a
                 good "camera cut" sample (~60-150 frames per episode).
T2 Local score : cheap color-histogram outlier + blank-frame detector over every
                 keyframe. Builds a robust "show palette" reference by median across
                 all frames in the collection. Runs locally, no model needed.
T3 VLM         : (a) OCR the episode title card and compare with the filename to
                 verify identity; (b) caption the flagged/audit frames for human
                 review. Uses any OpenAI-compatible vision endpoint.
T4 Report      : writes verify_report.json and prints a summary.

Design note
-----------
Small VLMs (e.g. MiniCPM-V) are good captioners and OCR engines but poor at forced
single-word classification. This pipeline therefore uses the VLM only for OCR and
open-ended captions, and does the automated judgment with T0 checks + local T2
outlier detection.

Usage
-----
  python3 verify_media.py --root /path/to/library \
      --vlm http://localhost:8080/v1/chat/completions \
      --key TOKEN --model <vision-model> --jobs 8

  # no model available / offline: metadata + local outlier detection only
  python3 verify_media.py --root /path/to/library --no-vlm

Requirements: python3, ffmpeg + ffprobe, numpy, Pillow.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image

EP_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v", ".ts")

DEFAULTS = dict(
    min_duration=60.0,
    max_duration=3600.0,
    min_width=640,
    allowed_vcodec=("hevc", "h264", "av1", "vp9", "mpeg2video", "mpeg4"),
    allowed_alang=("eng", "und", "en"),
)


# ------------------------------------------------------------------ utils
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_json(path, entries):
    r = run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", path])
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def human_time(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# ------------------------------------------------------------------ T0 metadata
def probe(path):
    j = ffprobe_json(path, "format=duration,size,format_name:format_tags:"
                           "stream=index,codec_type,codec_name,width,height,pix_fmt,"
                           "channels,sample_rate:stream_tags=language")
    fmt = j.get("format", {})
    streams = j.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    s = [s for s in streams if s.get("codec_type") == "subtitle"]

    def dur():
        try:
            return float(fmt.get("duration"))
        except Exception:
            return None

    info = {
        "path": path,
        "name": os.path.basename(path),
        "duration": dur(),
        "size": int(fmt.get("size") or 0),
        "container": fmt.get("format_name"),
        "n_video": len(v), "n_audio": len(a), "n_sub": len(s),
        "vcodec": v[0].get("codec_name") if v else None,
        "width": v[0].get("width") if v else None,
        "height": v[0].get("height") if v else None,
        "pix_fmt": v[0].get("pix_fmt") if v else None,
        "acodec": a[0].get("codec_name") if a else None,
        "achannels": a[0].get("channels") if a else None,
        "alang": (a[0].get("tags") or {}).get("language") if a else None,
    }
    m = EP_RE.search(info["name"])
    info["season"] = int(m.group(1)) if m else None
    info["episode"] = int(m.group(2)) if m else None
    info["title"] = None
    if m:
        t = info["name"][m.end():]
        t = re.sub(r"\((?:1080p|720p|2160p|480p)[^)]*\)", "", t)
        t = re.sub(r"\[[^\]]*\]", "", t)
        t = re.sub(r"\.(mkv|mp4|avi|mov|webm|m4v|ts)$", "", t, flags=re.I)
        info["title"] = t.strip(" .-_–") or None
    return info


def tier0_checks(info, cfg):
    issues = []
    d = info["duration"]
    if d is None:
        issues.append("unreadable")
    elif not (cfg["min_duration"] <= d <= cfg["max_duration"]):
        issues.append(f"odd duration {human_time(d)}")
    if info["n_video"] != 1:
        issues.append(f"n_video={info['n_video']}")
    if info["n_audio"] < 1:
        issues.append("no audio")
    if info["vcodec"] not in cfg["allowed_vcodec"]:
        issues.append(f"vcodec={info['vcodec']}")
    if info["width"] and info["width"] < cfg["min_width"]:
        issues.append(f"low res {info['width']}x{info['height']}")
    if info["alang"] and info["alang"] not in cfg["allowed_alang"]:
        issues.append(f"audio lang={info['alang']}")
    if info["season"] is None or info["episode"] is None:
        issues.append("no SxxEe in filename")
    return issues


# ------------------------------------------------------------------ T1 keyframes
def keyframe_times(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path])
    ts = []
    for line in r.stdout.splitlines():
        parts = line.rsplit(",", 1)
        if len(parts) == 2 and "K" in parts[1]:
            try:
                ts.append(float(parts[0]))
            except ValueError:
                pass
    return ts


def extract_keyframes(path, outdir, width=224):
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(outdir, f))
    # -skip_frame nokey decodes only I-frames -> fast, and keyframes land on cuts
    run(["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", path, "-vsync", "0",
         "-vf", f"scale={width}:-2", "-q:v", "5", os.path.join(outdir, "%04d.jpg"), "-y"])
    return sorted(os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".jpg"))


# ------------------------------------------------------------------ T2 local score
def features(img_path):
    im = Image.open(img_path).convert("RGB").resize((32, 32))
    a = np.asarray(im, dtype=np.float32)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    h = np.histogram(hsv[..., 0], bins=12, range=(0, 256))[0]
    s = np.histogram(hsv[..., 1], bins=4, range=(0, 256))[0]
    v = np.histogram(hsv[..., 2], bins=4, range=(0, 256))[0]
    hist = np.concatenate([h, s, v]).astype(np.float64)
    hist = hist / (hist.sum() + 1e-9)
    gray = a.mean(axis=2)
    return {
        "hist": hist.tolist(),
        "bright": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def low_energy(paths, n):
    """Pick the n flattest+brightest frames (title cards/plain backgrounds)."""
    def energy(p):
        try:
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        except Exception:
            return 1e9
        return (np.mean(np.abs(np.diff(g, axis=1))) + np.mean(np.abs(np.diff(g, axis=0)))
                - 0.05 * float(g.mean()))
    return sorted(paths, key=energy)[:n]


# ------------------------------------------------------------------ T3 VLM
class VLM:
    def __init__(self, url, key, model):
        self.url, self.key, self.model = url, key, model

    def ask_images(self, paths, prompt, max_tokens=300, timeout=300):
        content = [{"type": "text", "text": prompt}]
        for p in paths:
            b = base64.b64encode(open(p, "rb").read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b}})
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": content}],
                           "max_tokens": max_tokens, "temperature": 0.0}).encode()
        hdr = {"Content-Type": "application/json"}
        if self.key:
            hdr["Authorization"] = "Bearer " + self.key
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers=hdr)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == 3:
                    return "VLM_ERROR: " + repr(e)
                time.sleep(2)


def ocr_title(vlm, path):
    txt = vlm.ask_images(
        [path],
        "Read the large white text in the centre of this image. Output only that text. "
        "If there is no such text, output NONE.",
        max_tokens=40)
    if txt.strip().upper().startswith("NONE"):
        return None
    return txt.strip()


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def title_match(ocr, expected):
    import difflib
    a, b = norm(ocr), norm(expected)
    if not a or not b:
        return None
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.75


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Tiered downloaded-media verifier.")
    ap.add_argument("--root", default=".", help="folder containing the media files")
    ap.add_argument("--vlm", default=os.environ.get("VLM_URL",
                    "http://localhost:8080/v1/chat/completions"),
                    help="OpenAI-compatible chat completions endpoint")
    ap.add_argument("--key", default=os.environ.get("VLM_KEY", ""),
                    help="bearer token for the VLM endpoint")
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "openbmb/MiniCPM-V-4.6-gguf"))
    ap.add_argument("--show", default="", help="expected show/title, used only in prompts")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--audit", type=int, default=4, help="random audit frames per file")
    ap.add_argument("--zcut", type=float, default=6.0, help="robust-z outlier threshold")
    ap.add_argument("--maxflag", type=int, default=8, help="max flagged frames/file sent to VLM")
    ap.add_argument("--width", type=int, default=224, help="keyframe width in px")
    ap.add_argument("--min-duration", type=float, default=DEFAULTS["min_duration"])
    ap.add_argument("--max-duration", type=float, default=DEFAULTS["max_duration"])
    ap.add_argument("--min-width", type=int, default=DEFAULTS["min_width"])
    ap.add_argument("--lang", default="eng,und,en", help="allowed audio languages, comma separated")
    ap.add_argument("--no-vlm", action="store_true", help="skip all VLM calls")
    ap.add_argument("--resume", action="store_true", help="reuse cached keyframes in workdir")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(min_duration=args.min_duration, max_duration=args.max_duration,
               min_width=args.min_width,
               allowed_alang=tuple(x.strip() for x in args.lang.split(",") if x.strip()))

    root = os.path.abspath(args.root)
    work = args.workdir or os.path.join(tempfile.gettempdir(),
                                        "media_verify_" + (os.path.basename(root) or "root"))
    if not args.resume:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    files = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(VIDEO_EXT):
                files.append(os.path.join(dirpath, n))
    files.sort()
    if not files:
        print(f"No video files found under {root}", file=sys.stderr)
        sys.exit(1)
    print(f"[T0] {len(files)} video files under {root}", flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        infos = list(ex.map(probe, files))
    for i in infos:
        i["t0_issues"] = tier0_checks(i, cfg)
    print(f"[T0] done in {time.time()-t0:.1f}s; "
          f"{sum(1 for i in infos if i['t0_issues'])} files with issues", flush=True)

    # ---- T1 keyframes
    t1 = time.time()

    def do_kf(item):
        idx, info = item
        d = os.path.join(work, "kf", f"{idx:04d}")
        ts = keyframe_times(info["path"])
        cached = (sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
                  if os.path.isdir(d) else [])
        if args.resume and cached:
            return info, ts, cached
        return info, ts, extract_keyframes(info["path"], d, args.width)

    with cf.ThreadPoolExecutor(args.jobs) as ex:
        kf = list(ex.map(do_kf, list(enumerate(infos))))
    for (info, ts, jpgs), _ in zip(kf, infos):
        info["kf_times"], info["kf_jpgs"] = ts, jpgs
    print(f"[T1] keyframes in {time.time()-t1:.1f}s "
          f"({sum(len(i['kf_jpgs']) for i in infos)} frames)", flush=True)

    # ---- T2 local scoring
    t2 = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        flat = [(i, j, p) for i, info in enumerate(infos) for j, p in enumerate(info["kf_jpgs"])]
        results = list(ex.map(lambda x: (x[0], x[1], features(x[2])), flat))
    per_file = [[] for _ in infos]
    for i, j, f in results:
        per_file[i].append((j, f))
    for i, info in enumerate(infos):
        per_file[i].sort()
        info["feats"] = [f for _, f in per_file[i]]

    all_feats = [f for info in infos for f in info["feats"]]
    if all_feats:
        gref = np.median(np.array([f["hist"] for f in all_feats]), axis=0)
        gref = gref / (gref.sum() + 1e-9)
    else:
        gref = None

    for info in infos:
        feats = info["feats"]
        if not feats or gref is None:
            info["flagged"], info["z"] = [], []
            continue
        H = np.array([f["hist"] for f in feats])
        sim = (H @ gref) / (np.linalg.norm(H, axis=1) * np.linalg.norm(gref) + 1e-9)
        dist = 1 - sim
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) + 1e-6
        z = (dist - med) / (1.4826 * mad)
        blanks = [j for j, f in enumerate(feats) if f["bright"] < 18 and f["contrast"] < 10]
        info["z"] = z.tolist()
        info["flagged"] = sorted(set(np.where(z > args.zcut)[0].tolist()) | set(blanks))
    print(f"[T2] scored in {time.time()-t2:.1f}s; "
          f"{sum(len(i['flagged']) for i in infos)} outlier frames flagged", flush=True)

    # ---- T3 VLM: title OCR + captions
    if not args.no_vlm:
        vlm = VLM(args.vlm, args.key, args.model)
        rng = np.random.default_rng(42)
        show_clause = f" The collection is expected to contain '{args.show}'." if args.show else ""

        # T3a: title-card OCR, keyframes first, 1fps fallback for misses
        t3 = time.time()
        kf_cand = []
        for info in infos:
            pairs = [(p, t) for p, t in zip(info["kf_jpgs"], info["kf_times"]) if 25 <= t <= 150]
            if not pairs:
                pairs = list(zip(info["kf_jpgs"], info["kf_times"]))
            kf_cand.append(low_energy([p for p, _ in pairs], 3))
        ktasks = [(i, p) for i, ps in enumerate(kf_cand) for p in ps]
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            ktxt = list(ex.map(lambda x: (x[0], ocr_title(vlm, x[1])), ktasks))
        title = {}
        for i, t in ktxt:
            if t and i not in title and title_match(t, infos[i]["title"]):
                title[i] = t

        need = [i for i, info in enumerate(infos)
                if i not in title and (info["duration"] or 0) > 60]

        def onefps(i):
            info = infos[i]
            d = os.path.join(work, "tc", f"{i:04d}")
            if not (args.resume and os.path.isdir(d)
                    and any(x.endswith(".jpg") for x in os.listdir(d))):
                os.makedirs(d, exist_ok=True)
                end = max(40.0, min(170.0, (info["duration"] or 400) * 0.5))
                run(["ffmpeg", "-v", "error", "-ss", "40", "-t", f"{end-40:.2f}",
                     "-i", info["path"], "-vf", "fps=1,scale=256:-2", "-q:v", "5",
                     os.path.join(d, "%03d.jpg"), "-y"])
            files = [os.path.join(d, x) for x in sorted(os.listdir(d)) if x.endswith(".jpg")]
            return i, low_energy(files, 3)

        with cf.ThreadPoolExecutor(args.jobs) as ex:
            fall = list(ex.map(onefps, need))
        ftasks = [(i, p) for i, ps in fall for p in ps]
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            ftxt = list(ex.map(lambda x: (x[0], ocr_title(vlm, x[1])), ftasks))
        for i, t in ftxt:
            if t and i not in title:
                title[i] = t
        for i, info in enumerate(infos):
            info["title_ocr"] = title.get(i)
            info["title_match"] = title_match(info["title_ocr"], info["title"])
        print(f"[T3] title OCR in {time.time()-t3:.1f}s "
              f"(fast {len(ktasks)} + fallback {len(ftasks)} calls)", flush=True)

        # T3b: caption flagged + audit frames (advisory)
        t3b = time.time()
        tasks = []
        for i, info in enumerate(infos):
            n = len(info["kf_jpgs"])
            picks = set(info["flagged"][:args.maxflag])
            rest = [j for j in range(n) if j not in picks]
            if rest and args.audit > 0:
                for j in rng.choice(rest, size=min(args.audit, len(rest)), replace=False):
                    picks.add(int(j))
            for j in sorted(picks):
                tasks.append((i, j))

        def describe(x):
            i, j = x
            txt = vlm.ask_images(
                [infos[i]["kf_jpgs"][j]],
                "Describe this single image in one short sentence."
                + (" Name the show if you recognize it." if not show_clause else
                   f" Name the show if you recognize it. Expected show:{show_clause}"),
                max_tokens=90)
            return i, j, txt

        with cf.ThreadPoolExecutor(args.jobs) as ex:
            descs = list(ex.map(describe, tasks))
        per = {}
        for i, j, txt in descs:
            per.setdefault(i, {})[j] = txt
        for i, info in enumerate(infos):
            info["vlm_frames"] = [
                {"idx": j,
                 "time": info["kf_times"][j] if j < len(info["kf_times"]) else None,
                 "flagged": j in info["flagged"],
                 "desc": per[i][j]}
                for j in sorted(per.get(i, {}))]
            note = []
            for fr in info["vlm_frames"]:
                d = (fr["desc"] or "").lower()
                if re.search(r"test pattern|color bars|live[- ]action|real (photo|footage)|photograph", d) \
                        or re.search(r"adult|nudit|sexual|violen|blood|gore|disturbing|inappropriate", d):
                    note.append(fr)
            info["vlm_bad"] = note
        print(f"[T3] captioned {len(tasks)} frames in {time.time()-t3b:.1f}s", flush=True)

    # ---- T4 report
    episodes = sorted([i for i in infos if i["season"] is not None],
                      key=lambda x: (x["season"], x["episode"]))
    summary = {
        "root": root,
        "n_files": len(infos),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_issues": [{"file": i["name"], "issues": i["t0_issues"]}
                            for i in infos if i["t0_issues"]],
        "vlm_flagged": [{"file": i["name"], "bad": i.get("vlm_bad", [])}
                        for i in infos if i.get("vlm_bad")],
        "title_mismatch": [{"file": i["name"], "filename_title": i.get("title"),
                            "ocr_title": i.get("title_ocr"), "match": i.get("title_match")}
                           for i in infos if i.get("title_match") is False],
        "title_missing": [i["name"] for i in infos if i.get("title_ocr") is None],
        "local_outlier_frames": [{"file": i["name"], "n_flagged": len(i["flagged"]),
                                  "n_frames": len(i["kf_jpgs"])}
                                 for i in infos if i["flagged"]],
    }
    for i in infos:
        i.pop("feats", None)
    summary["files"] = infos
    out = args.out or os.path.join(root, "verify_report.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)

    print("\n===== SUMMARY =====")
    print(f"files: {summary['n_files']}  episodes: {len(episodes)}")
    print(f"metadata issues: {len(summary['metadata_issues'])}")
    for x in summary["metadata_issues"][:20]:
        print("  -", x["file"], "->", x["issues"])
    print(f"local outlier files: {len(summary['local_outlier_frames'])}")
    for x in summary["local_outlier_frames"][:20]:
        print(f"  - {x['file']}: {x['n_flagged']}/{x['n_frames']} frames")
    print(f"VLM suspicious files: {len(summary['vlm_flagged'])}")
    for x in summary["vlm_flagged"][:20]:
        print(f"  - {x['file']}: {x['bad']}")
    if not args.no_vlm:
        n_ok = sum(1 for i in infos if i.get("title_match"))
        print(f"title cards matched to filename: {n_ok}/{len(infos)}")
        print(f"title mismatch: {len(summary['title_mismatch'])}")
        for x in summary["title_mismatch"][:30]:
            print(f"  - {x['file']}: file='{x['filename_title']}' ocr='{x['ocr_title']}'")
        print(f"title not found: {len(summary['title_missing'])}")
        for x in summary["title_missing"][:30]:
            print("  -", x)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
