"""Run autolabel on every check frame: CSV of status + overlay contact sheets."""
import csv
from pathlib import Path

import cv2
import numpy as np

import autolabel as A
import fit_lines as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "autolabel_out"


def main():
    OUT.mkdir(exist_ok=True)
    rows, tiles = [], []
    partial = open(OUT / "results_partial.csv", "w", newline="")  # survives an interrupted run
    for f in sorted((HERE / "check_frames").glob("*.jpg")):
        img = cv2.imread(str(f))
        r = A.label_frame(img)
        rows.append({"frame": f.stem, "status": r["status"], "score": round(float(r["score"]), 1),
                     "n_lines": r["n_lines"], "n_arcs": r["n_arcs"],
                     "line_rms": round(r.get("line_rms", np.nan), 2), "support": r.get("support"),
                     "features": " ".join(sorted(r["feats"]))})
        ov = img if r["H"] is None else F.draw(img, np.diag([1 / F.S, 1 / F.S, 1]) @ r["H"], A.L, A.W)
        t = cv2.resize(ov, (600, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{f.stem} {r['status']}", (5, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(t)
        if len(rows) == 1:
            pw = csv.DictWriter(partial, rows[0].keys())
            pw.writeheader()
        pw.writerow(rows[-1])
        partial.flush()
        print(rows[-1], flush=True)
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for i in range(0, len(tiles), 20):
        page = tiles[i:i + 20] + [np.zeros_like(tiles[0])] * (-len(tiles[i:i + 20]) % 4)
        cv2.imwrite(str(OUT / f"sheet_{i // 20:02d}.jpg"),
                    np.vstack([np.hstack(page[j:j + 4]) for j in range(0, len(page), 4)]))
    st = [r["status"] for r in rows]
    print({s: st.count(s) for s in set(st)})


if __name__ == "__main__":
    main()
