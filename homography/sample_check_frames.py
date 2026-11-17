"""Sample live-play frames for the homography feasibility check.

One frame from each of N equal windows per match, using the same scoreboard and
grass tests as dataset/sample_frames.py (no YOLO: close-ups are rare and get
spotted in the overlays). Output: homography/check_frames/M01_123.4s.jpg ...
"""
import csv
import random
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "dataset"))
from camera_benchmark import MIN_GREEN_FRAC, MIN_HULL_FRAC, pitch_mask  # noqa: E402
from sample_frames import MIN_SCORE_PANEL, SCORE_PANEL, VIDEO_DIR  # noqa: E402

OUT = HERE / "check_frames"
PER_MATCH = 10
TRIES = 8
SEED = 7


def is_live(frame):
    panel = cv2.cvtColor(frame[SCORE_PANEL], cv2.COLOR_BGR2HSV)
    if ((panel[..., 2] > 170) & (panel[..., 1] < 40)).mean() < MIN_SCORE_PANEL:
        return False
    _, _, green, hull = pitch_mask(frame)
    return green >= MIN_GREEN_FRAC and hull >= MIN_HULL_FRAC


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    with open(ROOT / "dataset" / "splits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cap = cv2.VideoCapture(str(VIDEO_DIR / row["file"]))
        dur = float(row["duration_s"])
        got = 0
        for k in range(PER_MATCH):
            a, b = dur * k / PER_MATCH, dur * (k + 1) / PER_MATCH
            for _ in range(TRIES):
                t = rng.uniform(a, b)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if ok and is_live(frame):
                    cv2.imwrite(str(OUT / f"{row['match_id']}_{t:06.1f}s.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    got += 1
                    break
        cap.release()
        print(row["match_id"], got, "frames", flush=True)


if __name__ == "__main__":
    main()
