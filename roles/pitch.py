"""Players -> pitch metres, and a side-by-side minimap video.

  python roles/pitch.py M11 60 10

Joins roles/out/<match>_live<start>.csv (boxes + roles, every decoded frame) with the
homography track homography/tracks/<match>_<start>.csv (H at 10 fps; made with
homography/track.py if missing). Each box's feet (bottom centre) go through the nearest
H in time (within MAX_DT) into pitch coordinates:
  X 0..105 m from the goal line on the LEFT of the screen (the camera never changes side),
  Y -34..34 m from the far touchline (top of screen) to the near one.
The homography model measures x from whichever goal line it anchored on, so X flips it
when model +x points left on screen.
Writes roles/out/<stem>_pitch.csv and <stem>_pitch.mp4.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "homography"))
OUT = HERE / "out"
L, W = 105.0, 68.0
MAX_DT = 0.1         # s: farther than this from a homography sample -> no pitch position (VFR gaps reach 0.17 s)
MAP_PX = 10          # minimap pixels per metre
MARGIN = 4           # m of run-off drawn around the minimap
HEAD_PX = 48         # legend strip between the frame and the minimap


def to_pitch(H, feet):
    """H model->image px, feet (n,2) px -> (n,2) global X, Y in metres."""
    Hi = np.linalg.inv(H)

    def back(px):
        p = np.c_[px, np.ones(len(px))] @ Hi.T
        return p[:, :2] / p[:, 2:]

    xy = back(np.asarray(feet, float))
    # orientation from the middle of the screen (always visible; far-off markings can project
    # through the horizon): screen-right must be +X, screen-down (nearer) must be +Y. The model
    # measures x from whichever goal it anchored on, and the markings are symmetric about y = 0,
    # so an anchor can also land on the mirror image.
    c, r, d = back(np.array([[1200.0, 540], [1300, 540], [1200, 640]]))
    if r[0] < c[0]:
        xy[:, 0] = L - xy[:, 0]
    if d[1] < c[1]:
        xy[:, 1] = -xy[:, 1]
    return xy


LINE_M = 4           # linesman: within this of a touchline
BOX_M = 2            # keeper: inside the penalty box plus this margin
OFFICIAL = {"referee": (255, 0, 255), "linesman": (0, 165, 255), "keeper": (255, 255, 0)}


def split_other(X, Y):
    """Median pitch position of an "other" track -> referee | linesman | keeper.
    shortcut: position only, so a referee standing in a box (corners, penalties) reads as keeper;
    add the officials' shared kit colour if that shows up."""
    if abs(Y) > W / 2 - LINE_M:
        return "linesman"
    if (X < 16.5 + BOX_M or X > L - 16.5 - BOX_M) and abs(Y) < 20.16 + BOX_M:
        return "keeper"
    return "referee"


def load_track(match, start, dur):
    path = ROOT / "homography" / "tracks" / f"{match}_{start:g}.csv"
    if path.exists():   # an older, shorter run under the same name must not be reused
        with open(path, newline="") as f:
            ts = [float(r["t"]) for r in csv.DictReader(f)]
        if not ts or ts[-1] < start + dur - 0.5:
            path.unlink()
    if not path.exists():
        import track as T
        with open(ROOT / "dataset" / "splits.csv", newline="") as f:
            file = next(r["file"] for r in csv.DictReader(f) if r["match_id"] == match)
        rows = T.run(ROOT / "Screen Recordings" / "Final Recordings" / file, start, dur)
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["h0"]]
    t = np.array([float(r["t"]) for r in rows])
    Hs = [np.array([float(r[f"h{i}"]) for i in range(9)]).reshape(3, 3) for r in rows]
    keep = consistent(Hs)
    print(f"  homography: {len(Hs)} samples, {int((~keep).sum())} dropped as inconsistent with their neighbours")
    return t[keep], [H for H, k in zip(Hs, keep) if k]


def consistent(Hs, win=3, tol_m=4.0):
    """Drop one-off wrong fits: a sample whose screen points (centre, lower centre) land more than
    tol_m from the median of its +-win neighbours. The camera moves well under that in 0.3 s."""
    probe = np.array([[1200.0, 540], [1200, 900]])
    pts = np.array([to_pitch(H, probe).ravel() for H in Hs])
    keep = np.ones(len(Hs), bool)
    for i in range(len(Hs)):
        nb = np.r_[pts[max(0, i - win):i], pts[i + 1:i + 1 + win]]
        if len(nb) >= 2:
            d = np.abs(pts[i] - np.median(nb, axis=0)).reshape(2, 2)
            keep[i] = np.hypot(d[:, 0], d[:, 1]).max() <= tol_m
    return keep


def minimap():
    w, h = int((L + 2 * MARGIN) * MAP_PX), int((W + 2 * MARGIN) * MAP_PX)
    img = np.full((h, w, 3), (40, 110, 40), np.uint8)
    P = lambda x, y: (int((x + MARGIN) * MAP_PX), int((y + W / 2 + MARGIN) * MAP_PX))
    white = (230, 230, 230)
    cv2.rectangle(img, P(0, -W / 2), P(L, W / 2), white, 2)
    cv2.line(img, P(L / 2, -W / 2), P(L / 2, W / 2), white, 2)
    cv2.circle(img, P(L / 2, 0), int(9.15 * MAP_PX), white, 2)
    for x0, s in ((0, 1), (L, -1)):
        cv2.rectangle(img, P(x0, -20.16), P(x0 + s * 16.5, 20.16), white, 2)
        cv2.rectangle(img, P(x0, -9.16), P(x0 + s * 5.5, 9.16), white, 2)
    return img, P


def ball_track(stem, rows, ht, Hs):
    """<stem>_ball.csv candidates -> {frame: dict(t, u, v, kind, X, Y, air)}, also written to
    <stem>_balltrack.csv. Empty if the role pass predates ball candidates."""
    import ball as B
    path = OUT / f"{stem}_ball.csv"
    if not path.exists():
        print("  no ball candidates (rerun run_roles.py live)")
        return {}
    cands = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cands.setdefault(r["frame"], []).append(
                (float(r["conf"]), (float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"])))
    times = {r["frame"]: float(r["t"]) for r in rows}
    with open(path, newline="") as f:
        times.update({r["frame"]: float(r["t"]) for r in csv.DictReader(f)})
    order = sorted(times, key=times.get)
    res = B.track([(times[n], cands.get(n, [])) for n in order])
    out = {}
    for n, (t, u, v, kind) in zip(order, res):
        X = Y = ""
        if kind:
            i = int(np.abs(ht - t).argmin()) if len(ht) else -1
            if i >= 0 and abs(ht[i] - t) <= MAX_DT:
                X, Y = (round(float(c), 2) for c in to_pitch(Hs[i], np.array([[u, v]]))[0])
        out[n] = dict(frame=n, t=t, u=u, v=v, kind=kind, X=X, Y=Y, air=False)
    placed = [o for o in out.values() if o["X"] != ""]
    if placed:
        air = B.airborne([o["t"] for o in placed], [[o["X"], o["Y"]] for o in placed])
        for o, a in zip(placed, air):
            o["air"] = bool(a)
    with open(OUT / f"{stem}_balltrack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["frame", "t", "u", "v", "kind", "X", "Y", "air"])
        w.writeheader()
        w.writerows(out[n] for n in order)
    k = [o["kind"] for o in out.values()]
    n = len(k)
    n_cand = sum(1 for n_ in order if cands.get(n_))
    print(f"  ball: {k.count('det') / n:.1%} detected, {k.count('fill') / n:.1%} filled, "
          f"{k.count(None) / n:.1%} none ({n_cand / n:.1%} of frames had a candidate); "
          f"{sum(o['air'] for o in placed)} placed samples flagged airborne")
    return out


def main():
    match, start = sys.argv[1], float(sys.argv[2])
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    stem = f"{match}_live{start:g}"
    with open(OUT / f"{stem}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    colour = {k: tuple(v) for k, v in json.loads((OUT / f"{stem}_teams.json").read_text()).items()}
    ht, Hs = load_track(match, start, dur)

    for r in rows:
        t = float(r["t"])
        i = int(np.abs(ht - t).argmin()) if len(ht) else -1
        r["X"] = r["Y"] = ""
        if i >= 0 and abs(ht[i] - t) <= MAX_DT:
            feet = np.array([[(float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"])]])
            X, Y = to_pitch(Hs[i], feet)[0]
            r["X"], r["Y"] = round(float(X), 2), round(float(Y), 2)

    # "other" tracks -> referee / linesman / keeper from the track's median position
    oth = {}
    for r in rows:
        if r["role"] == "O" and r["X"] != "" and r["track"] != "-1":
            oth.setdefault(r["track"], []).append((r["X"], r["Y"]))
    kind = {k: split_other(*np.median(np.array(v, float), axis=0)) for k, v in oth.items()}
    for r in rows:
        if r["role"] != "O":
            r["role2"] = r["role"]
        elif r["track"] == "-1":     # not confirmed by the tracker: classify by its own position
            r["role2"] = split_other(float(r["X"]), float(r["Y"])) if r["X"] != "" else "other"
        else:
            r["role2"] = kind.get(r["track"], "other")
    print(f"{stem}: other tracks -> " + ", ".join(f"{k} {list(kind.values()).count(k)}" for k in OFFICIAL))
    with open(OUT / f"{stem}_pitch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # sanity: share inside the pitch (+3 m run-off), and per-track speed over 1 s
    # (frame to frame, ~0.2 m of box jitter alone reads as several m/s)
    pos = [r for r in rows if r["X"] != ""]
    xy = np.array([[r["X"], r["Y"]] for r in pos], float)
    inside = ((xy[:, 0] > -3) & (xy[:, 0] < L + 3) & (np.abs(xy[:, 1]) < W / 2 + 3)).mean() if len(xy) else 0
    tracks = {}
    for r in pos:
        if r["track"] != "-1":       # untracked boxes are different people, not one track
            tracks.setdefault(r["track"], []).append((float(r["t"]), r["X"], r["Y"]))
    sp = []
    for v in tracks.values():
        v = np.array(sorted(v))
        j = np.searchsorted(v[:, 0], v[:, 0] + 1.0)
        for i in np.flatnonzero((j < len(v))):
            dt = v[j[i], 0] - v[i, 0]
            if dt < 1.1:
                sp.append(np.hypot(*(v[j[i], 1:] - v[i, 1:])) / dt)
    sp = np.array(sp)
    print(f"{stem}: {len(pos)}/{len(rows)} boxes placed; {inside:.1%} inside the pitch; "
          f"X {xy[:, 0].min():.1f}..{xy[:, 0].max():.1f} m, Y {xy[:, 1].min():.1f}..{xy[:, 1].max():.1f} m")
    if len(sp):
        print(f"  speed over 1 s: median {np.median(sp):.1f} m/s, 95% {np.percentile(sp, 95):.1f}, "
              f"max {sp.max():.1f}, {(sp > 10).mean():.1%} over 10 m/s")

    ball = ball_track(stem, rows, ht, Hs)

    # video: broadcast frame on top, minimap below, same width
    from run_roles import video_clip
    base, P = minimap()
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    names, frames = video_clip(match, start, dur)
    span = names[-1][1] - names[0][1]
    fps = (len(names) - 1) / span if span > 0 else 25.0
    vw_w = base.shape[1]
    top_h = int(1080 * vw_w / 2400)
    vw = cv2.VideoWriter(str(OUT / f"{stem}_pitch.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (vw_w, top_h + HEAD_PX + base.shape[0]))
    for name, t, img in frames():
        mm = base.copy()
        for r in by_frame.get(name, []):
            c = colour.get(r["role2"]) or OFFICIAL.get(r["role2"], (128, 128, 128))
            p1, p2 = (int(float(r["x1"])), int(float(r["y1"]))), (int(float(r["x2"])), int(float(r["y2"])))
            cv2.rectangle(img, p1, p2, c, 4)
            if r["X"] != "":
                q = P(r["X"], r["Y"])
                cv2.circle(mm, q, 10, (0, 0, 0), -1)
                cv2.circle(mm, q, 8, c, -1)
        b = ball.get(name)
        if b and b["kind"]:
            u, v = int(b["u"]), int(b["v"])
            cv2.circle(img, (u, v - 8), 22, (0, 255, 255), 3 if b["kind"] == "det" else 1)
            trail = [ball[n] for n, tt in names if t - 1.0 <= tt <= t and n in ball and ball[n]["X"] != ""]
            for p, q in zip(trail, trail[1:]):
                cv2.line(mm, P(p["X"], p["Y"]), P(q["X"], q["Y"]), (255, 255, 255), 2)
            if b["X"] != "":
                q = P(b["X"], b["Y"])
                solid = b["kind"] == "det" and not b["air"]
                cv2.circle(mm, q, 8, (0, 0, 0), -1)
                cv2.circle(mm, q, 6, (255, 255, 255), -1 if solid else 2)
        head = np.full((HEAD_PX, vw_w, 3), 30, np.uint8)
        cv2.putText(head, f"{match}  {t:6.2f}s", (12, 32), 0, 0.9, (255, 255, 255), 2)
        x = 300
        for lab, c in [("team A", colour["A"]), ("team B", colour["B"]), *OFFICIAL.items(), ("ball", (255, 255, 255))]:
            cv2.circle(head, (x, 24), 9, (0, 0, 0), -1)
            cv2.circle(head, (x, 24), 7, c, -1)
            cv2.putText(head, lab, (x + 14, 31), 0, 0.6, (255, 255, 255), 2)
            x += 30 + 12 * len(lab)
        vw.write(np.vstack([cv2.resize(img, (vw_w, top_h), interpolation=cv2.INTER_AREA), head, mm]))
    vw.release()
    print(f"  -> out/{stem}_pitch.mp4")


if __name__ == "__main__":
    main()
