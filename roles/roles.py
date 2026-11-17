"""Role stage prototype: team A / team B / other from kit colour, seeded at kickoff.

1. kit_colour: median Lab colour of the shirt area of a person box (rows 15-50 %,
   middle 60 % of columns), dropping pixels close to the grass sampled in a ring
   around the box (a fixed "remove green" would erase green kits).
2. seed: on the kickoff frames, the halfway line (homography) splits people into
   two halves -> one median colour per half, then a few 2-means passes that ignore
   people far from both (the referee, a stray player across the line).
3. assign: nearest team colour, or "other" (referee, linesmen, keepers) when it is
   farther than the threshold learned from the seed spread.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "homography"))

RING_PX = 6          # grass ring width around the box
GRASS_DE = 12        # Lab distance: shirt pixels this close to the grass colour are dropped
MIN_PX = 15          # fewer shirt pixels than this -> no colour
OTHER_MIN = 18       # "other" threshold never below this Lab distance
TEAM, OTHER = ("A", "B"), "O"


def lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def kit_colour(img_lab, box):
    """box = x1,y1,x2,y2 (px) -> Lab (3,) or None."""
    h, w = img_lab.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    gx1, gy1, gx2, gy2 = max(0, x1 - RING_PX), max(0, y1 - RING_PX), min(w, x2 + RING_PX), min(h, y2 + RING_PX)
    ring = img_lab[gy1:gy2, gx1:gx2].copy()
    inner = np.zeros(ring.shape[:2], bool)
    inner[y1 - gy1:y2 - gy1, x1 - gx1:x2 - gx1] = True
    grass = np.median(ring[~inner], axis=0) if (~inner).any() else None
    shirt = img_lab[y1 + int(0.15 * bh):y1 + int(0.5 * bh), x1 + int(0.2 * bw):x2 - int(0.2 * bw)].reshape(-1, 3)
    if grass is not None:
        shirt = shirt[np.linalg.norm(shirt - grass, axis=1) > GRASS_DE]
    return np.median(shirt, axis=0) if len(shirt) >= MIN_PX else None


def side_of_halfway(H, feet):
    """H model->image px; feet (n,2) px -> bool array, True = beyond halfway (model x > L/2)."""
    import autolabel as A
    p = np.c_[feet, np.ones(len(feet))] @ np.linalg.inv(H).T
    return p[:, 0] / p[:, 2] > A.L / 2


def seed(colours, sides, iters=5):
    """colours (n,3), sides (n,) bool -> (centres (2,3), threshold).

    Team A = the half with sides False in the first kickoff frame given."""
    colours, sides = np.asarray(colours, float), np.asarray(sides, bool)
    c = np.array([np.median(colours[~sides], axis=0), np.median(colours[sides], axis=0)])
    keep = np.ones(len(colours), bool)
    for _ in range(iters):
        d = np.linalg.norm(colours[:, None] - c[None], axis=2)
        lab_ = d.argmin(1)
        dmin = d.min(1)
        keep = dmin <= max(OTHER_MIN, 2.5 * np.median(dmin))   # drops the referee / outliers
        c = np.array([np.median(colours[keep & (lab_ == k)], axis=0) if (keep & (lab_ == k)).any() else c[k]
                      for k in range(2)])
    d = np.linalg.norm(colours[:, None] - c[None], axis=2).min(1)[keep]
    # cap: a loose team spread must not swallow the referee, who sits between or beside the teams
    gap = float(np.linalg.norm(c[0] - c[1]))
    return c, max(OTHER_MIN, min(3 * float(np.percentile(d, 90)), 0.6 * gap))


def assign(colour, centres, thr):
    if colour is None:
        return None
    d = np.linalg.norm(centres - colour, axis=1)
    return TEAM[int(d.argmin())] if d.min() <= thr else OTHER


def lab_to_bgr(c):
    px = cv2.cvtColor(np.float32(c).reshape(1, 1, 3), cv2.COLOR_Lab2BGR)
    return tuple(int(v) for v in np.clip(px[0, 0] * 255, 0, 255))
