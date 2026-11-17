"""Read the FC Mobile HUD game clock (MM:SS) by digit template matching. No OCR model needed:
the clock is a fixed green box with dark digits in four fixed slots.

  python roles/clock.py build     make clock_digits.npz from frames whose clock was read by eye
  python roles/clock.py M04 100   read the clock at 100 s of M04
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES = HERE / "clock_digits.npz"
BOX = (slice(95, 160), slice(400, 660))          # clock area in a 2400x1080 frame
SLOTS = [68, 96, 137, 164]                       # digit x in the box; each 24 x 32 from y 16
DARK_V = 90
# frames read by eye (match, video s, clock): together they hold every digit 0-9
LABELLED = [("M04", 41, "0032"), ("M04", 100, "2055"), ("M11", 141, "2521"), ("M12", 280, "5406"),
            ("M12", 70, "0859"), ("M03", 200, "4435"), ("M17", 300, "8717"),
            ("M04", 160, "3853"), ("M11", 60, "0000"), ("M12", 290, "5822"), ("M05", 150, "2716"),
            ("M09", 250, "4500"), ("M20", 200, "5352")]


def video_path(match):
    with open(ROOT / "dataset" / "splits.csv", newline="") as f:
        file = next(r["file"] for r in csv.DictReader(f) if r["match_id"] == match)
    return ROOT / "Screen Recordings" / "Final Recordings" / file


def frame_at(match, t):
    cap = cv2.VideoCapture(str(video_path(match)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    return f if f.shape[:2] == (1080, 2400) else cv2.resize(f, (2400, 1080))


def glyphs(frame):
    """-> (4, 32*24) float binary digit images, or None when the green clock box is not shown."""
    box = cv2.cvtColor(frame[BOX], cv2.COLOR_BGR2HSV)
    inner = box[12:52, 60:200]
    green = ((inner[..., 0] > 60) & (inner[..., 0] < 95) & (inner[..., 1] > 100) & (inner[..., 2] > 150)).mean()
    if green < 0.5:                                # faded full-time card, replay, menu
        return None
    dark = (box[..., 2] < DARK_V).astype(np.float32)
    return np.array([dark[16:48, x:x + 24].ravel() for x in SLOTS])


def build():
    X, y = [], []
    for match, t, txt in LABELLED:
        g = glyphs(frame_at(match, t))
        assert g is not None, (match, t)
        X += list(g)
        y += [int(c) for c in txt]
    X, y = np.array(X), np.array(y)
    assert set(y) == set(range(10)), sorted(set(y))
    tmpl = np.array([X[y == d].mean(0) for d in range(10)])
    np.savez(TEMPLATES, tmpl=tmpl)
    # leave-one-out check on the labelled digits
    wrong = 0
    for i in range(len(X)):
        others = [X[(y == d) & (np.arange(len(X)) != i)] for d in range(10)]
        t_ = np.array([o.mean(0) if len(o) else np.full(X.shape[1], 9.0) for o in others])
        wrong += int(np.argmin(((t_ - X[i]) ** 2).sum(1)) != y[i])
    print(f"templates from {len(X)} digits; leave-one-out wrong: {wrong}")


_tmpl = None


def read(frame):
    """-> game minutes as a float (e.g. 54.1 for 54:06), or None if no clock is shown."""
    global _tmpl
    if _tmpl is None:
        _tmpl = np.load(TEMPLATES)["tmpl"]
    g = glyphs(frame)
    if g is None:
        return None
    d = [int(np.argmin(((_tmpl - x) ** 2).sum(1))) for x in g]
    return d[0] * 10 + d[1] + (d[2] * 10 + d[3]) / 60


def half(match, t):
    """1 or 2 from the clock at t (searching up to 5 s on for a frame that shows it), else None."""
    for dt in np.arange(0, 5, 0.5):
        f = frame_at(match, t + dt)
        m = read(f) if f is not None else None
        if m is not None:
            return 1 if m < 45 else 2
    return None


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        m = read(frame_at(sys.argv[1], float(sys.argv[2])))
        print("no clock" if m is None else f"{int(m):02d}:{round((m % 1) * 60):02d}")
