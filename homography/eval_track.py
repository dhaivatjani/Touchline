"""Score the tracker against the clicked ground truth.

For each clicked test frame at time t: track from t - LEAD s up to t (one anchor at the
start, then tracking unless it loses the pitch), with the clicked JPEG as the last step.
Reports the status reached at t and the landmark errors, like eval_gt.py.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

import track as T
from eval_gt import WRONG_M, landmark_errors, load_clicks

HERE = Path(__file__).resolve().parent
LEAD = 5.0


def main():
    with open(T.ROOT / "dataset" / "splits.csv", newline="") as f:
        files = {r["match_id"]: r["file"] for r in csv.DictReader(f)}
    clicks, _ = load_clicks()
    res = []
    for name, cl in sorted(clicks.items()):
        match, t = name[:3], float(name[4:-1])
        video = T.ROOT / "Screen Recordings" / "Final Recordings" / files[match]
        rng = np.random.default_rng(0)
        H, prev, sts = None, None, []
        for _, img in T.frames(video, t - LEAD, LEAD - 0.05):
            st, H, _, prev = T.step(img, H, prev, rng)
            sts.append(st)
        st, H, sup, _ = T.step(cv2.imread(str(HERE / "check_frames" / f"{name}.jpg")), H, prev, rng)
        if H is None:
            print(f"{name}: {st}")
            res.append((name, st, None))
            continue
        err = landmark_errors(H, cl)
        med = float(np.median([e[1] for e in err.values()]))
        res.append((name, st, med))
        print(f"{name}: {st} (anchors in lead-in {sts.count('anchor')}/{len(sts)}), "
              f"median {med:.2f} m, max {max(e[1] for e in err.values()):.2f} m", flush=True)
    got = [r for r in res if r[2] is not None]
    right = [r for r in got if r[2] <= WRONG_M]
    print(f"\n{len(res)} frames: H for {len(got)}, right {len(right)}, "
          f"wrong {[r[0] for r in got if r[2] > WRONG_M]}")
    print("status at t:", {s: sum(r[1] == s for r in res) for s in {r[1] for r in res}})
    if right:
        print(f"median of per-frame medians on right frames: {np.median([r[2] for r in right]):.2f} m")


if __name__ == "__main__":
    main()
