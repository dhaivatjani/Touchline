"""
Sample frames from the Final Recordings for annotation, with YOLO pre-labels.

For every match in splits.csv:
  1. Cut the video into fixed windows by timestamp (the recordings are
     variable frame rate, so frame indices are not a reliable clock).
  2. In each window, draw several random candidate timestamps and try them
     in random order. The first candidate that passes is kept:
       - grass test        (same rule as benchmark/camera_benchmark.py)
       - live-play test    (>= 4 on-pitch players, median player height <= 20%
                            of the frame; same rule as benchmark/make_tables.py)
       - not a near-duplicate of the previous kept frame of that match
       - scoreboard visible (replays and cutscenes hide it)
     A window where nothing passes (cutscene, replay, menu) gives no frame.
     Overlays on live play (GOAL card, HALFTIME banner) are kept: the pipeline
     sees them too. Don't box the card art during annotation.
  3. Save the frame and a YOLO-format pre-label file for correction in CVAT.

Pre-label classes (fix these by hand during annotation):
  0 player      <- every on-pitch COCO 'person'; relabel goalkeepers/referees
  1 goalkeeper  <- not pre-labelled
  2 referee     <- not pre-labelled (includes linesmen)
  3 ball        <- COCO 'sports ball' inside the pitch

Output (dataset/frames/):
  images/{train,val,test}/M05_t0123.45.jpg
  labels/{train,val,test}/M05_t0123.45.txt
  data.yaml
  manifest.csv     one row per kept frame (match, time, frame index, counts)
  windows.csv      one row per window, including the empty ones

Usage:
    python dataset/sample_frames.py                 # all matches (resumes)
    python dataset/sample_frames.py --only M01      # one match
    python dataset/sample_frames.py --overwrite     # start again
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from camera_benchmark import (  # noqa: E402  same filter as the benchmark
    MIN_GREEN_FRAC, MIN_HULL_FRAC, UI_TOP_FRAC, on_pitch, pitch_mask,
)

VIDEO_DIR = ROOT / "Screen Recordings" / "Final Recordings"
SPLITS_CSV = HERE / "splits.csv"
OUT_DIR = HERE / "frames"
MODEL = ROOT / "benchmark" / "yolo11m.pt"

SEED = 42
WINDOW_S = {"train": 10.0, "val": 8.0, "test": 8.0}
CANDIDATES_PER_WINDOW = 8

IMGSZ = 1920
PERSON_CONF = 0.30
BALL_CONF = 0.15
LIVE_MIN_PLAYERS = 4
LIVE_MAX_MEDIAN_H = 0.20

# Near-duplicate test: mean absolute difference of small grayscale thumbnails
# (0-255 scale). Below this the frame is treated as the same scene.
DUP_THUMB = (160, 72)
DUP_MAX_DIFF = 4.0         # different live scenes 10 s apart measured 6-22 on M01

# Replays and cutscenes use other cameras and hide the scoreboard. The light-grey
# score panel sits at a fixed place in every recording (2400x1080): live play
# measures 0.83-0.96 light-grey pixels there, replays 0.00-0.03.
SCORE_PANEL = (slice(40, 95), slice(440, 610))
MIN_SCORE_PANEL = 0.5

CLASSES = ["player", "goalkeeper", "referee", "ball"]
JPEG_QUALITY = 95


def thumb(frame):
    small = cv2.resize(frame, DUP_THUMB, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def candidate_times(duration, window_s, rng):
    """Per window: CANDIDATES_PER_WINDOW random timestamps, in try order."""
    windows = []
    n = int(np.ceil(duration / window_s))
    for k in range(n):
        a, b = k * window_s, min((k + 1) * window_s, duration)
        if b - a < 1.0:
            break
        windows.append((k, a, b, [rng.uniform(a, b) for _ in range(CANDIDATES_PER_WINDOW)]))
    return windows


def detect(model, frame):
    """On-pitch players and balls, plus the live-play verdict. None if not gameplay."""
    H, W = frame.shape[:2]
    panel = cv2.cvtColor(frame[SCORE_PANEL], cv2.COLOR_BGR2HSV)
    if ((panel[..., 2] > 170) & (panel[..., 1] < 40)).mean() < MIN_SCORE_PANEL:
        return None, "no_scoreboard"
    _, pitch, green_frac, hull_frac = pitch_mask(frame)
    if green_frac < MIN_GREEN_FRAC or hull_frac < MIN_HULL_FRAC:
        return None, "grass"

    res = model.predict(frame, imgsz=IMGSZ, conf=min(PERSON_CONF, BALL_CONF),
                        classes=[0, 32], verbose=False)[0]
    pitch_inner = cv2.erode(pitch, np.ones((7, 7), np.uint8))
    players, balls = [], []
    for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.cls.cpu().numpy().astype(int),
                                      res.boxes.conf.cpu().numpy()):
        if y2 < H * UI_TOP_FRAC:
            continue
        if c == 0 and p >= PERSON_CONF and on_pitch(pitch, (x1 + x2) / 2, y2):
            players.append((x1, y1, x2, y2))
        elif c == 32 and p >= BALL_CONF and on_pitch(pitch_inner, (x1 + x2) / 2, (y1 + y2) / 2):
            balls.append((x1, y1, x2, y2))

    if len(players) < LIVE_MIN_PLAYERS:
        return None, "few_players"
    med_h = float(np.median([(y2 - y1) / H for _, y1, _, y2 in players]))
    if med_h > LIVE_MAX_MEDIAN_H:
        return None, "closeup"
    return {"players": players, "balls": balls, "green_frac": green_frac, "med_h": med_h}, "ok"


def yolo_lines(det, W, H):
    lines = []
    for cls, boxes in ((0, det["players"]), (3, det["balls"])):
        for x1, y1, x2, y2 in boxes:
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def process_match(model, row):
    mid, split = row["match_id"], row["split"]
    rng = random.Random(f"{SEED}-{mid}")
    windows = candidate_times(float(row["duration_s"]), WINDOW_S[split], rng)

    # Every candidate timestamp maps to the first decoded frame at or after it.
    wanted = sorted((t, k) for k, _, _, cands in windows for t in cands)
    grabbed = {}                      # window -> list of (target_t, frame_t, frame_idx, frame)
    cap = cv2.VideoCapture(str(VIDEO_DIR / row["file"]))
    i, idx = 0, -1
    pending_windows = [w[0] for w in windows]
    kept_rows, window_rows = [], []
    last_thumb = None

    def finish_window(k):
        nonlocal last_thumb
        _, a, b, cands = windows[k]
        order = {t: j for j, t in enumerate(cands)}          # original random try order
        tries = sorted(grabbed.pop(k, []), key=lambda c: order[c[0]])
        reasons = []
        for target, t, fidx, frame in tries:
            det, why = detect(model, frame)
            if det is None:
                reasons.append(why)
                continue
            th = thumb(frame)
            diff = float(np.abs(th - last_thumb).mean()) if last_thumb is not None else 255.0
            if diff < DUP_MAX_DIFF:
                reasons.append("duplicate")
                continue
            last_thumb = th
            H, W = frame.shape[:2]
            name = f"{mid}_t{t:07.2f}"
            cv2.imwrite(str(OUT_DIR / "images" / split / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            (OUT_DIR / "labels" / split / f"{name}.txt").write_text(
                "\n".join(yolo_lines(det, W, H)) + "\n", encoding="utf-8")
            kept_rows.append(dict(
                image=f"images/{split}/{name}.jpg", match_id=mid, split=split,
                t=round(t, 3), frame_idx=fidx, window=k, tries=len(reasons) + 1,
                n_players=len(det["players"]), n_ball=len(det["balls"]),
                median_player_h=round(det["med_h"], 4), dup_diff=round(diff, 2)))
            window_rows.append(dict(match_id=mid, split=split, window=k, start=a, end=b,
                                    kept=name, rejected=" ".join(reasons)))
            return
        window_rows.append(dict(match_id=mid, split=split, window=k, start=a, end=b,
                                kept="", rejected=" ".join(reasons)))

    while i < len(wanted) or pending_windows:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if i < len(wanted) and t >= wanted[i][0]:
            ok, frame = cap.retrieve()
            while i < len(wanted) and t >= wanted[i][0]:
                if ok:
                    grabbed.setdefault(wanted[i][1], []).append((wanted[i][0], t, idx, frame))
                i += 1
        # a window is complete once decoding has passed its end
        while pending_windows and t >= windows[pending_windows[0]][2]:
            finish_window(pending_windows.pop(0))
    cap.release()
    for k in pending_windows:          # windows touching the very end of the file
        finish_window(k)
    return kept_rows, window_rows


def append_csv(path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def write_data_yaml():
    text = (f"path: {OUT_DIR.as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    (OUT_DIR / "data.yaml").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="match ids, e.g. M01 M04")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.overwrite and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for sub in ("images", "labels"):
        for split in WINDOW_S:
            (OUT_DIR / sub / split).mkdir(parents=True, exist_ok=True)
    write_data_yaml()

    manifest, windows_csv = OUT_DIR / "manifest.csv", OUT_DIR / "windows.csv"
    done = set()
    if windows_csv.exists():
        with open(windows_csv, encoding="utf-8") as fh:
            done = {r["match_id"] for r in csv.DictReader(fh)}

    with open(SPLITS_CSV, encoding="utf-8") as fh:
        matches = list(csv.DictReader(fh))
    if args.only:
        matches = [m for m in matches if m["match_id"] in args.only]

    model = YOLO(str(MODEL))
    for m in matches:
        if m["match_id"] in done:
            print(f"skip {m['match_id']} (already sampled)")
            continue
        kept, wins = process_match(model, m)
        append_csv(manifest, kept)
        append_csv(windows_csv, wins)
        print(f"{m['match_id']} [{m['split']:5s}] windows={len(wins):3d} kept={len(kept):3d}", flush=True)

    if manifest.exists():
        with open(manifest, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for split in WINDOW_S:
            sel = [r for r in rows if r["split"] == split]
            print(f"total {split:5s}: {len(sel):4d} frames, "
                  f"{sum(int(r['n_players']) for r in sel):6d} player boxes, "
                  f"{sum(int(r['n_ball']) for r in sel):4d} ball boxes")


if __name__ == "__main__":
    main()
