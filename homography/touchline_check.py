"""Is the far touchline visible? Longest near-flat white segment in the upper frame."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from band_check import white_mask  # noqa: E402

Y_RANGE = (110, 600)        # below the scoreboard strip, upper half of the pitch
MIN_LEN = 600               # px at 2400 wide; a full far touchline spans 1000+
MAX_ANGLE = 12              # degrees from horizontal


def far_touchline(img):
    m = white_mask(img).astype(np.uint8) * 255
    m[: Y_RANGE[0]] = 0
    m[Y_RANGE[1]:] = 0
    segs = cv2.HoughLinesP(m, 1, np.pi / 720, 80, minLineLength=150, maxLineGap=40)
    best = 0
    for x1, y1, x2, y2 in ([] if segs is None else segs[:, 0]):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if min(ang, 180 - ang) < MAX_ANGLE:
            best = max(best, np.hypot(x2 - x1, y2 - y1))
    return best


def main():
    rows = []
    for f in sorted((HERE / "check_frames").glob("*.jpg")):
        L = far_touchline(cv2.imread(str(f)))
        rows.append({"frame": f.stem, "longest_flat_px": round(L), "visible": L >= MIN_LEN})
    with open(HERE / "touchline_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    vis = sum(r["visible"] for r in rows)
    print(f"far touchline found in {vis}/{len(rows)} frames")
    print("not found:", " ".join(r["frame"] for r in rows if not r["visible"]))


if __name__ == "__main__":
    main()
