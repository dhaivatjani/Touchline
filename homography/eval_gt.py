"""Score autolabel against the clicked ground truth (ground_truth.csv from click_gt.py).

Per clicked landmark: pixel error |H*model - click| and pitch error in metres
|H^-1*click - model|. A frame counts as a wrong "ok" if its median error is over
WRONG_M. Output: eval_gt.csv (one row per landmark) and a printed summary.
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import autolabel as A
from click_gt import LANDMARKS

HERE = Path(__file__).resolve().parent
MODEL = dict(LANDMARKS)
WRONG_M = 2.0               # median error above this = the labeller picked the wrong answer


def landmark_errors(H, clicks):
    """clicks: {landmark name: (x_px, y_px)} -> {name: (px error, metre error)}."""
    out = {}
    Hi = np.linalg.inv(H)
    for name, (x, y) in clicks.items():
        best = None
        mx, my = MODEL[name]
        # box/corner names don't say which goal; H's x=0 end is arbitrary: try both ends
        for m in {(mx, my), (A.L - mx, my)}:
            m = np.array(m, float)
            q = H @ [*m, 1]
            p = Hi @ [x, y, 1]
            e = (float(np.hypot(*(q[:2] / q[2] - (x, y)))), float(np.hypot(*(p[:2] / p[2] - m))))
            if best is None or e[1] < best[1]:
                best = e
        out[name] = best
    return out


def load_clicks():
    frames = defaultdict(dict)
    unusable = set()
    with open(HERE / "ground_truth.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["landmark"] == "UNUSABLE":
                unusable.add(r["frame"])
            elif r["landmark"] != "NONE":
                frames[r["frame"]][r["landmark"]] = (float(r["x_px"]), float(r["y_px"]))
    return frames, unusable


def main():
    frames, unusable = load_clicks()
    rows, per_frame = [], []
    for name, clicks in sorted(frames.items()):
        r = A.label_frame(cv2.imread(str(HERE / "check_frames" / f"{name}.jpg")))
        if r["status"] != "ok":
            per_frame.append((name, r["status"], None))
            print(f"{name}: {r['status']} (missed, {len(clicks)} landmarks clicked)")
            continue
        err = landmark_errors(r["H"], clicks)
        for lm, (epx, em) in err.items():
            rows.append({"frame": name, "landmark": lm, "err_px": round(epx, 2), "err_m": round(em, 3)})
        med = np.median([e[1] for e in err.values()])
        per_frame.append((name, "ok", med))
        print(f"{name}: {len(err)} landmarks, median {np.median([e[0] for e in err.values()]):.1f} px"
              f" / {med:.2f} m, max {max(e[1] for e in err.values()):.2f} m")
    with open(HERE / "eval_gt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["frame", "landmark", "err_px", "err_m"])
        w.writeheader()
        w.writerows(rows)

    n = len(per_frame)
    ok = [p for p in per_frame if p[1] == "ok"]
    right = [p for p in ok if p[2] <= WRONG_M]
    print(f"\n{n} live frames with clicks ({len(unusable)} marked unusable)")
    print(f"labeller ok on {len(ok)}/{n}; right (median <= {WRONG_M} m) {len(right)}, "
          f"wrong ok {len(ok) - len(right)}: {[p[0] for p in ok if p[2] > WRONG_M]}")
    good = [r for r in rows if r["frame"] in {p[0] for p in right}]
    if good:
        px = np.array([r["err_px"] for r in good])
        m = np.array([r["err_m"] for r in good])
        print(f"on right frames, {len(good)} landmarks: median {np.median(px):.1f} px / "
              f"{np.median(m):.2f} m, 95% {np.percentile(px, 95):.1f} px / {np.percentile(m, 95):.2f} m")


if __name__ == "__main__":
    main()
