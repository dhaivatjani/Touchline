"""Homography for a stretch of video: track frame to frame, re-anchor when lost.

Per sampled frame (FPS per second of video, by timestamp: recordings are VFR):
  not live (no scoreboard / grass)  -> status not_live, H none, next live frame re-anchors
  have a previous H                 -> predict with optical flow on the grass (camera
                                       motion), refine on the pitch lines (autolabel.refine)
                                       -> status track if support holds
  otherwise / tracking failed       -> full autolabel.label_frame -> status anchor | lost
Line refinement every frame means no drift; flow only gives it a close enough start.

Usage: python track.py M11 60 30   (match, start s, duration s)
Writes tracks/<match>_<start>.csv (t, status, support, H as 9 numbers) and a small
preview mp4 with the model drawn on.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import autolabel as A
import fit_lines as F
from sample_check_frames import is_live

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "tracks"
FPS = 10                    # same rate the benchmark samples full matches at
FLOW_SCALE = 0.5            # optical flow on half-res frames
TRACK_RADIUS = 12           # px: first refine pass picks up markings this far from the prediction


def pitch_small(img):
    """Pitch mask at flow resolution."""
    _, pitch, _ = A.masks(img)
    h, w = img.shape[:2]
    return cv2.resize(pitch.astype(np.uint8), (int(w * FLOW_SCALE), int(h * FLOW_SCALE)),
                      interpolation=cv2.INTER_NEAREST)


def flow_homography(prev_gray, gray, mask):
    """Inter-frame homography (full-res px) from LK flow on pitch features, or None."""
    pts = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if pts is None or len(pts) < 15:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 15:
        return None
    G, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 3.0)  # players move: RANSAC
    if G is None or inl.sum() < 15:
        return None
    S = np.diag([FLOW_SCALE, FLOW_SCALE, 1])
    return np.linalg.inv(S) @ G @ S


def step(img, H_prev, prev, rng):
    """One frame -> (status, H, support, flow state for the next frame)."""
    if not is_live(img):
        return "not_live", None, None, None
    gray = cv2.cvtColor(cv2.resize(img, None, fx=FLOW_SCALE, fy=FLOW_SCALE), cv2.COLOR_BGR2GRAY)
    mask = pitch_small(img)
    if H_prev is not None and prev is not None:
        G = flow_homography(prev[0], gray, prev[1])
        Hp = (G if G is not None else np.eye(3)) @ H_prev
        white, pitch, dist = A.masks(img)
        grass = A.grass_points(pitch, rng)
        r = None
        if len(grass) >= 20:
            # wide search first: a flow prediction a few px off would otherwise drop the
            # thin far touchline, and without it the fit slowly loses depth
            r = A.refine(img, Hp, None, white, pitch, dist, grass, rng, radius=TRACK_RADIUS)
            r = r and A.refine(img, r["H"], None, white, pitch, dist, grass, rng)
        if r and r["support"] >= A.MIN_SUPPORT and r["n_on"] >= A.MIN_ON:
            return "track", r["H"], r["support"], (gray, mask)
    r = A.label_frame(img, rng)
    if r["status"] == "ok":
        return "anchor", r["H"], r["support"], (gray, mask)
    return "lost", None, None, (gray, mask)


def frames(video, start, dur):
    """(t, frame) every 1/FPS s of video time, from start for dur seconds."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    nxt = start
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t > start + dur:
            break
        if t + 1e-6 >= nxt:
            ok, img = cap.retrieve()
            if ok:
                yield t, img
            nxt += 1 / FPS * max(1, round((t - nxt) * FPS) + 1)
    cap.release()


def run(video, start, dur, preview=None):
    rng = np.random.default_rng(0)
    rows, H, prev = [], None, None
    vw = None
    for t, img in frames(video, start, dur):
        t0 = time.time()
        status, H, sup, prev = step(img, H, prev, rng)
        rows.append({"t": round(t, 3), "status": status, "support": sup, "sec": round(time.time() - t0, 2),
                     **{f"h{i}": (None if H is None else H.ravel()[i]) for i in range(9)}})
        if preview:
            ov = img if H is None else F.draw(img, np.diag([1 / F.S, 1 / F.S, 1]) @ H, A.L, A.W)
            ov = cv2.resize(ov, (960, 432), interpolation=cv2.INTER_AREA)
            cv2.putText(ov, f"{t:7.1f}s {status}", (8, 424), 0, 0.6, (0, 255, 255), 2)
            if vw is None:
                vw = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 432))
            vw.write(ov)
    if vw:
        vw.release()
    return rows


def main():
    match, start, dur = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    with open(ROOT / "dataset" / "splits.csv", newline="") as f:
        file = next(r["file"] for r in csv.DictReader(f) if r["match_id"] == match)
    OUT.mkdir(exist_ok=True)
    stem = f"{match}_{start:g}"
    rows = run(ROOT / "Screen Recordings" / "Final Recordings" / file, start, dur, OUT / f"{stem}.mp4")
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    st = [r["status"] for r in rows]
    secs = [r["sec"] for r in rows]
    print(f"{len(rows)} frames: " + ", ".join(f"{s} {st.count(s)}" for s in sorted(set(st))))
    for s in sorted(set(st)):
        sel = [x for x, y in zip(secs, st) if y == s]
        print(f"  {s}: median {np.median(sel):.2f} s/frame")
    print(f"total {sum(secs):.0f} s for {dur:g} s of video")


if __name__ == "__main__":
    main()
