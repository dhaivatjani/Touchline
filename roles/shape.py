"""Team shape and formation per stretch, from roles/out/<stem>_pitch.csv (+ _balltrack.csv).

  python roles/shape.py M04 47.5

Attack direction (which goal each team defends) comes from the offside line: the defending
team's deepest outfield player is almost always nearer its own goal than the attackers'. Per
frame, on the side of the pitch the play is on, the team holding the deeper extreme gets the vote
for defending that goal; majority over the stretch. (Sides swap at half time, so no kickoff prior.)

Shape, per team per frame (outfield only, >= MIN_SHAPE visible), in attack-oriented metres:
  centroid, width (Y p90 - p10), depth (X p90 - p10), defensive line (mean of the two deepest,
  as distance from own goal), hull area (convex hull of visible players).

Formation: frames with >= MIN_FORM outfield visible; positions centred and scaled (per-axis std),
matched to standardised templates with an optimal assignment (Hungarian, visible players onto a
subset of the 10 slots); lowest cost wins the frame; majority over the stretch, also over the
frames where the team is defending (ball in its own half). Scored against dataset/formations.csv.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
L, W = 105.0, 68.0
MIN_SHAPE, MIN_FORM = 5, 8
FORMATIONS = ["4-4-2", "4-3-3", "4-2-3-1", "4-2-4", "4-1-4-1", "4-5-1", "3-5-2", "3-4-3", "5-3-2"]


def template(name):
    """Formation -> (10, 2) slots: lines evenly spaced back (x=0) to front (x=1), players spread
    across y in [-1, 1] (two-man lines at +-0.4, a lone player central)."""
    lines = [int(c) for c in name.split("-")]
    pts = []
    for i, n in enumerate(lines):
        x = i / (len(lines) - 1)
        ys = [0.0] if n == 1 else [-0.4, 0.4] if n == 2 else np.linspace(-1, 1, n)
        pts += [(x, y) for y in ys]
    return np.array(pts)


def standardise(p):
    p = np.asarray(p, float) - np.mean(p, axis=0)
    return p / np.maximum(p.std(axis=0), 1e-6)


TEMPLATES = {f: standardise(template(f)) for f in FORMATIONS}


def match_formation(pts):
    """Attack-oriented outfield positions (n,2), n <= 10 -> (best formation, {formation: cost})."""
    p = standardise(pts)
    costs = {}
    for f, t in TEMPLATES.items():
        d = ((p[:, None, :] - t[None]) ** 2).sum(2)
        r, c = linear_sum_assignment(d)
        costs[f] = float(d[r, c].mean())
    return min(costs, key=costs.get), costs


def directions(frames):
    """frames: {frame: {"A": (n,2) xy, "B": ...}} -> {"A": +1 | -1}: +1 = defends the left goal
    (X = 0) and attacks towards +X."""
    votes = Counter()
    for f in frames.values():
        a, b = f.get("A"), f.get("B")
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            continue
        mid = np.r_[a[:, 0], b[:, 0]].mean()
        if mid < L / 2:     # play in the left half: deeper (smaller X) team defends left
            votes["A" if a[:, 0].min() < b[:, 0].min() else "B"] += 1
        else:               # right half: the team with the larger max X defends right
            votes["B" if a[:, 0].max() > b[:, 0].max() else "A"] += 1
    a_left = votes["A"] >= votes["B"]
    return {"A": 1 if a_left else -1, "B": -1 if a_left else 1}, votes


def kickoff_left(match):
    """Team ("A"/"B") standing in the screen-left half at the first-half kickoff, or None.
    Cached in out/<match>_kickoff.json (needs the detector and a homography on the kickoff frame)."""
    path = OUT / f"{match}_kickoff.json"
    if path.exists():
        return json.loads(path.read_text())["left"]
    import glob
    import roles as R          # puts homography/ on sys.path
    import autolabel as A
    import run_roles as RR
    from pitch import to_pitch
    seeded = RR.seed_match(match)
    if seeded is None:
        return None
    centres, thr, _ = seeded
    for f in sorted(glob.glob(str(RR.KICK / f"{match}_kickoff_*.jpg"))):
        img = cv2.imread(f)
        r = A.label_frame(img)
        if r["H"] is None or r["status"] != "ok":
            continue
        boxes, _ = RR.persons(RR.model().predict(img, imgsz=RR.IMGSZ, conf=RR.CONF, quantize=16, verbose=False)[0])
        lab = R.lab(img)
        xs = {"A": [], "B": []}
        for b in boxes:
            role = R.assign(R.kit_colour(lab, b), centres, thr)
            if role in xs:
                xs[role].append(to_pitch(r["H"], np.array([[(b[0] + b[2]) / 2, b[3]]]))[0, 0])
        if xs["A"] and xs["B"]:
            left = "A" if np.median(xs["A"]) < np.median(xs["B"]) else "B"
            path.write_text(json.dumps({"left": left, "A_x": float(np.median(xs["A"])),
                                        "B_x": float(np.median(xs["B"]))}))
            return left
    return None


def orient(xy, d):
    """Pitch xy -> attack-oriented (x = distance from own goal line, y with the attack to the left)."""
    xy = np.asarray(xy, float).copy()
    if d < 0:
        xy[:, 0], xy[:, 1] = L - xy[:, 0], -xy[:, 1]
    return xy


def shape(o):
    """Attack-oriented outfield positions (n,2), n >= 3 -> metrics dict."""
    x, y = o[:, 0], o[:, 1]
    hull = ConvexHull(o).volume if len(o) >= 3 and np.ptp(x) > 0.1 and np.ptp(y) > 0.1 else 0.0
    return dict(n=len(o), cx=x.mean(), cy=y.mean(), width=np.percentile(y, 90) - np.percentile(y, 10),
                depth=np.percentile(x, 90) - np.percentile(x, 10), line=np.sort(x)[:2].mean(), hull=hull)


def is_red(bgr):
    b, g, r = bgr
    return r > g + 40 and r > b + 40


def main():
    match, start = sys.argv[1], float(sys.argv[2])
    stem = f"{match}_live{start:g}"
    with open(OUT / f"{stem}_pitch.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["X"] != ""]
    colour = {k: tuple(v) for k, v in json.loads((OUT / f"{stem}_teams.json").read_text()).items()}
    ball = {}
    bpath = OUT / f"{stem}_balltrack.csv"
    if bpath.exists():
        with open(bpath, newline="") as f:
            ball = {r["frame"]: float(r["X"]) for r in csv.DictReader(f) if r["X"] != ""}

    frames, times = {}, {}
    for r in rows:
        if r["role2"] in ("A", "B"):
            frames.setdefault(r["frame"], {}).setdefault(r["role2"], []).append((float(r["X"]), float(r["Y"])))
            times[r["frame"]] = float(r["t"])
    frames = {k: {t: np.array(v) for t, v in f.items()} for k, f in frames.items()}
    # attack direction: kickoff half, swapped in the second half (clock); the offside vote only as fallback
    import clock
    half, left = clock.half(match, start), kickoff_left(match)
    vdirs, votes = directions(frames)
    if half and left:
        a_left = (left == "A") == (half == 1)
        dirs = {"A": 1 if a_left else -1, "B": -1 if a_left else 1}
        how = f"kickoff ({left} started left) + clock (half {half})"
    else:
        dirs, how = vdirs, "offside vote (no clock or kickoff)"
    (OUT / f"{stem}_dirs.json").write_text(json.dumps(dirs))     # +1 = defends the left goal
    agree = "agrees" if vdirs == dirs else "DISAGREES"
    print(f"{stem}: A defends {'left' if dirs['A'] > 0 else 'right'} goal from {how}; "
          f"offside vote {agree} (A {votes['A']} / B {votes['B']})")

    # shape + formation per frame per team
    out, form = [], {"A": Counter(), "B": Counter()}
    form_def = {"A": Counter(), "B": Counter()}
    for fr in sorted(frames, key=times.get):
        for team, xy in frames[fr].items():
            if len(xy) < MIN_SHAPE:
                continue
            o = orient(xy, dirs[team])
            row = dict(frame=fr, t=times[fr], team=team, **{k: round(float(v), 2) for k, v in shape(o).items()})
            row["formation"] = ""
            if MIN_FORM <= len(o) <= 10:
                best, _ = match_formation(o)
                row["formation"] = best
                form[team][best] += 1
                bx = ball.get(fr)
                if bx is not None and orient([[bx, 0]], dirs[team])[0, 0] < L / 2:
                    form_def[team][best] += 1
            out.append(row)
    with open(OUT / f"{stem}_shape.csv", "w", newline="") as f:
        w = csv.DictWriter(f, out[0].keys())
        w.writeheader()
        w.writerows(out)

    gt = {}
    with open(ROOT / "dataset" / "formations.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["match"] == match:
                gt[r["team"]] = r["formation"]
    for team in ("A", "B"):
        rs = [r for r in out if r["team"] == team]
        if not rs:
            continue
        med = {k: np.median([r[k] for r in rs]) for k in ("n", "width", "depth", "line", "hull")}
        who = "user" if is_red(colour[team]) else "opponent"
        truth = gt.get(who, "?")
        top = form[team].most_common(3)
        topd = form_def[team].most_common(3)
        n_f, n_d = sum(form[team].values()), sum(form_def[team].values())
        print(f"  team {team} ({who}, truth {truth}): {len(rs)} frames, median {med['n']:.0f} visible, "
              f"width {med['width']:.1f} m, depth {med['depth']:.1f} m, line {med['line']:.1f} m from goal, "
              f"hull {med['hull']:.0f} m2")
        fmt = lambda c, n: ", ".join(f"{k} {v / n:.0%}" for k, v in c) if n else "-"
        print(f"    formation, all {n_f} frames: {fmt(top, n_f)}"
              f"{'  <- right' if top and top[0][0] == truth else ''}")
        print(f"    formation, defending {n_d} frames: {fmt(topd, n_d)}"
              f"{'  <- right' if topd and topd[0][0] == truth else ''}")
    charts(stem, out, colour)
    video(stem, rows, frames, times, dirs, colour, form)


def charts(stem, out, colour):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = {k: tuple(c / 255 for c in v[::-1]) for k, v in colour.items()}
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    panels = [("line", "Defensive line (m from own goal)"), ("width", "Width (m)"),
              ("depth", "Depth (m)"), ("hull", "Hull area (m²)")]
    for ax, (k, title) in zip(axs, panels):
        for team in ("A", "B"):
            rs = [r for r in out if r["team"] == team]
            if not rs:
                continue
            t = np.array([r["t"] for r in rs])
            v = np.array([r[k] for r in rs], float)
            # 1 s rolling median: per-frame values jump as players enter/leave the view
            sm = np.array([np.median(v[np.abs(t - ti) <= 0.5]) for ti in t])
            ax.plot(t, sm, lw=2, color=rgb[team], label=f"Team {team}")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(loc="upper right", frameon=False)
    axs[-1].set_xlabel("video time (s)")
    fig.suptitle(f"{stem}: team shape (visible outfield players, 1 s median)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}_shape.png", dpi=110)
    plt.close(fig)


def video(stem, rows, frames, times, dirs, colour, form):
    """Minimap-only video: hull and defensive line per team, formation of the frame."""
    from pitch import minimap
    base, P = minimap()
    order = sorted(frames, key=times.get)
    span = times[order[-1]] - times[order[0]]
    fps = (len(order) - 1) / span if span > 0 else 25.0
    vw = cv2.VideoWriter(str(OUT / f"{stem}_shape.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (base.shape[1], base.shape[0]))
    by = {}
    for r in rows:
        by.setdefault(r["frame"], []).append(r)
    for fr in order:
        mm = base.copy()
        layer = mm.copy()
        labels = []
        for team, xy in frames[fr].items():
            c = colour[team]
            if len(xy) >= 3:
                try:
                    h = ConvexHull(xy)
                    cv2.fillPoly(layer, [np.array([P(*xy[i]) for i in h.vertices], np.int32)], c)
                except Exception:
                    pass   # collinear players
            if len(xy) >= MIN_SHAPE:
                o = orient(xy, dirs[team])
                line = np.sort(o[:, 0])[:2].mean()
                X = line if dirs[team] > 0 else L - line
                cv2.line(mm, P(X, -W / 2), P(X, W / 2), c, 3)
                if MIN_FORM <= len(o) <= 10:
                    labels.append(f"{team}: {match_formation(o)[0]}")
        mm = cv2.addWeighted(layer, 0.25, mm, 0.75, 0)
        for r in by.get(fr, []):
            if r["role2"] in ("A", "B"):
                q = P(float(r["X"]), float(r["Y"]))
                cv2.circle(mm, q, 9, (0, 0, 0), -1)
                cv2.circle(mm, q, 7, colour[r["role2"]], -1)
        cv2.putText(mm, f"{stem}  {times[fr]:6.2f}s   " + "   ".join(labels), (12, 28), 0, 0.8,
                    (255, 255, 255), 2)
        vw.write(mm)
    vw.release()
    print(f"  -> out/{stem}_shape.png, out/{stem}_shape.mp4, out/{stem}_shape.csv")


if __name__ == "__main__":
    main()
