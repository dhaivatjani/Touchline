"""
Split the Final Recordings into train / val / test at the match level.

Rules:
  - A whole match goes to one split (no frame leakage between splits).
  - Matches are grouped into recording sessions (same day / sitting).
  - Every val and test match comes from a different session, and every
    session keeps at least one match in train.
  - Seeded, so the split is reproducible.

Videos are not copied; the output is a manifest (splits.csv) that later
steps read.

Usage:
    python dataset/make_splits.py
"""

import csv
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "Screen Recordings" / "Final Recordings"
OUT_CSV = Path(__file__).resolve().parent / "splits.csv"

SEED = 42
N_VAL = 3
N_TEST = 3

# Recording sessions, from file timestamps (modified time / name).
SESSIONS = {
    "S1_0907": ["01.mp4"],
    "S2_0914": ["02.mp4", "03.mp4"],
    "S3_0915": ["04.mp4", "05.mp4"],
    "S4_0917": ["06.mp4", "07.mp4", "08.mp4"],
    "S5_0918": ["09.mp4", "10.mp4", "11.mp4"],
    "S6_0919am": ["12.mp4", "13.mp4", "14.mp4"],
    "S7_0919pm": [
        "Screen_Recording_20260919-211537_FC Mobile.mp4",
        "Screen_Recording_20260919-212343_FC Mobile.mp4",
    ],
    "S8_0922": [
        "Screen_Recording_20260922-214527_FC Mobile.mp4",
        "Screen_Recording_20260922-215330_FC Mobile.mp4",
        "Screen_Recording_20260922-220425_FC Mobile.mp4",
        "Screen_Recording_20260922-221402_FC Mobile.mp4",
    ],
}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    dur = float(info["format"]["duration"])
    frames = int(s["nb_frames"])
    return {
        "width": s["width"], "height": s["height"],
        "duration_s": round(dur, 1), "frames": frames,
        "avg_fps": round(frames / dur, 1),
    }


def main():
    on_disk = {p.name for p in VIDEO_DIR.glob("*.mp4")}
    listed = {f for files in SESSIONS.values() for f in files}
    if on_disk != listed:
        raise SystemExit(f"Session table out of date.\n"
                         f"  not listed: {sorted(on_disk - listed)}\n"
                         f"  missing:    {sorted(listed - on_disk)}")

    rng = random.Random(SEED)

    # Only sessions with >= 2 matches can give one away and still keep one in train.
    eligible = [s for s, files in SESSIONS.items() if len(files) >= 2]
    if len(eligible) < N_VAL + N_TEST:
        raise SystemExit("Not enough multi-match sessions for the requested split.")
    rng.shuffle(eligible)
    held_out_sessions = eligible[:N_VAL + N_TEST]

    split_of = {f: "train" for f in listed}
    for i, sess in enumerate(held_out_sessions):
        pick = rng.choice(sorted(SESSIONS[sess]))
        split_of[pick] = "test" if i < N_TEST else "val"

    # Stable IDs: 01-14 keep their number, the Screen_Recording files become 15-20.
    ordered = sorted(listed, key=lambda f: (not f[:2].isdigit(), f))
    rows = []
    for idx, fname in enumerate(ordered, start=1):
        session = next(s for s, files in SESSIONS.items() if fname in files)
        rows.append({
            "match_id": f"M{idx:02d}",
            "file": fname,
            "session": session,
            "split": split_of[fname],
            **probe(VIDEO_DIR / fname),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for split in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == split]
        mins = sum(r["duration_s"] for r in sel) / 60
        print(f"{split:5s} {len(sel):2d} matches  {mins:5.1f} min  "
              + ", ".join(r["match_id"] for r in sel))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
