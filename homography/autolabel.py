"""Automatic pitch-line labelling + homography for one frame (no hand segments).

1. Candidates: white pixels on the (eroded) pitch -> Hough segments merged into
   lines, split by angle into the touchline family (flat) and goal-line family
   (steep); leftover long curved blobs become ellipse (arc) candidates.
2. Hypotheses, each giving H in closed form:
   - near a goal: 2 steep lines labelled from {goal, six_front, box_front, halfway}
     + the far touchline + 1 more flat line from the box / near-touchline set (line DLT);
   - midfield: touchline + 1 steep line as halfway + an arc as the centre circle
     (4 points: corner, the two circle crossings, the far side of the arc).
3. Score = model points that land on white minus half of those that land on grass,
   after a plausibility test (grass must back-project inside the pitch).
4. The best H names the visible markings -> fit_lines.fit (105 x 68 fixed) refines it.
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

import fit_lines as F
from band_check import white_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from camera_benchmark import pitch_mask  # noqa: E402

L, W = 105.0, 68.0
FLAT_DEG = 25               # flatter than this = touchline family
SUPPORT_PX = 4              # model point counts as "on a line" within this distance
MIN_SUPPORT = 0.6           # accepted if this share of visible model points lands on white
MIN_ON = 60                 # ... and at least this many do
MAX_LINES = 12
ARC_CUT_LEN = 500
TOP_K = 6                   # best stage-1 hypotheses refined, then re-scored
N_TOUCH = 3                 # highest long flat lines tried as the far touchline (ad boards)
RUNOFF_M = 12               # grass this far outside the lines still counts as pitch           # only lines at least this long are removed before arc search              # strongest line candidates kept; bounds the hypothesis count

X_SLOTS = {"goal": 0.0, "six_front": 5.5, "box_front": 16.5, "halfway": L / 2}
Y_SLOTS = {"box_side": -20.16, "six_side": -9.16, "six_near": 9.16, "box_near": 20.16,
           "near_touch": W / 2}


def _model_samples():
    """Points every 0.5 m on every marking of the full pitch, tagged by feature name."""
    pts, tag = [], []

    def seg(a, b, name):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.5))
        t = np.linspace(0, 1, n)[:, None]
        pts.append(np.array(a) + t * (np.array(b) - np.array(a)))
        tag.extend([name] * n)

    def arc(c, a0, a1, name):
        a = np.arange(a0, a1, 0.5 / F.R_CIRCLE)
        pts.append(np.c_[c[0] + F.R_CIRCLE * np.cos(a), c[1] + F.R_CIRCLE * np.sin(a)])
        tag.extend([name] * len(a))

    w = W / 2
    seg((0, -w), (L, -w), "touch")
    seg((0, w), (L, w), "near_touch")
    seg((L / 2, -w), (L / 2, w), "halfway")
    arc((L / 2, 0), 0, 2 * np.pi, "centre")
    for x0, s in ((0, 1), (L, -1)):          # both ends; names refer to the half-model
        sfx = "" if x0 == 0 else "@far"
        seg((x0, -w), (x0, w), "goal" + sfx)
        seg((x0, -20.16), (x0 + s * 16.5, -20.16), "box_side" + sfx)
        seg((x0, 20.16), (x0 + s * 16.5, 20.16), "box_near" + sfx)
        seg((x0 + s * 16.5, -20.16), (x0 + s * 16.5, 20.16), "box_front" + sfx)
        seg((x0, -9.16), (x0 + s * 5.5, -9.16), "six_side" + sfx)
        seg((x0, 9.16), (x0 + s * 5.5, 9.16), "six_near" + sfx)
        seg((x0 + s * 5.5, -9.16), (x0 + s * 5.5, 9.16), "six_front" + sfx)
        a = np.arccos(5.5 / F.R_CIRCLE)       # D: part of the arc outside the box
        arc((x0 + s * 11, 0), -a, a, "d_arc" + sfx) if s == 1 else \
            arc((x0 + s * 11, 0), np.pi - a, np.pi + a, "d_arc" + sfx)
    return np.vstack(pts), np.array(tag)


MODEL_PTS, MODEL_TAG = _model_samples()


def masks(img):
    """White-on-pitch mask, eroded pitch mask (full res), distance to nearest white px."""
    h, w = img.shape[:2]
    _, pitch, _, _ = pitch_mask(img)
    pitch = cv2.erode(pitch, np.ones((3, 3), np.uint8))      # 1/4 scale: drops ad-board edge
    pitch = cv2.resize(pitch, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    white = white_mask(img) & pitch
    dist = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 3)
    return white, pitch, dist


def line_candidates(white):
    """Merged straight lines as (a, b, c, p_start, p_end, n_pixels), a^2+b^2=1."""
    m = white.astype(np.uint8) * 255
    segs = cv2.HoughLinesP(m, 1, np.pi / 720, 60, minLineLength=100, maxLineGap=20)
    if segs is None:
        return []
    segs = segs[:, 0].astype(float)
    segs = segs[np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))]
    ys, xs = np.nonzero(white)
    pts = np.c_[xs, ys].astype(float)
    lines, used = [], np.zeros(len(segs), bool)
    for i, (x1, y1, x2, y2) in enumerate(segs):
        if used[i]:
            continue
        d = np.array([x2 - x1, y2 - y1]) / np.hypot(x2 - x1, y2 - y1)
        nrm = np.array([-d[1], d[0]])
        rho = nrm @ [x1, y1]
        # absorb near-collinear segments
        for j in range(i, len(segs)):
            if used[j]:
                continue
            a, b = segs[j, :2], segs[j, 2:]
            if abs(nrm @ a - rho) < 8 and abs(nrm @ b - rho) < 8:
                used[j] = True
        near = np.abs(pts @ nrm - rho) < 4
        p = pts[near]
        if len(p) < 80:
            continue
        vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d, nrm = np.array([vx, vy]), np.array([-vy, vx])
        rho = nrm @ [x0, y0]
        p = pts[np.abs(pts @ nrm - rho) < 3]
        t = np.sort(p @ d)
        # keep the longest run without gaps > 40 px (a line, not scattered white)
        gaps = np.flatnonzero(np.diff(t) > 40)
        runs = np.split(t, gaps + 1)
        run = max(runs, key=len)
        if run[-1] - run[0] < 100 or len(run) < 80:
            continue
        # a pitch line has grass beside it; goal net and ad-board text are white all round
        off = np.abs(pts @ nrm - rho)
        along = pts @ d
        side = ((off > 6) & (off < 16) & (along >= run[0]) & (along <= run[-1])).sum()
        if side > len(run):
            continue
        p0 = d * run[0] + nrm * rho
        p1 = d * run[-1] + nrm * rho
        lines.append((nrm[0], nrm[1], -rho, p0, p1, len(run)))
    lines.sort(key=lambda l: -l[5])
    return lines[:MAX_LINES]


def arc_candidates(white, lines):
    """Ellipses fitted to long curved white blobs not explained by any line."""
    h, w = white.shape
    rest = white.copy()
    ys, xs = np.nonzero(rest)
    for a, b, c, p0, p1, _ in lines:
        if np.hypot(*(p1 - p0)) < ARC_CUT_LEN:
            continue                      # short "lines" are often chords of the circle itself
        on = np.abs(a * xs + b * ys + c) < 6
        rest[ys[on], xs[on]] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), 8)
    arcs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if np.hypot(bw, bh) < 200 or area < 150 or area > 0.15 * bw * bh:
            continue                      # small, or a filled blob (player, text) rather than a curve
        yy, xx = np.nonzero(lab == i)
        arcs.append(np.c_[xx, yy].astype(np.float32))
    arcs.sort(key=len, reverse=True)
    return arcs[:3]


def flat(line):
    a, b = line[:2]
    return abs(np.degrees(np.arctan2(abs(a), abs(b)))) < FLAT_DEG   # normal near vertical


def dlt(pairs):
    """H (model -> image px) from (image line, model line) pairs: l_m ~ H^T l_i."""
    rows = []
    for li, lm in pairs:
        for a, b in ((1, 2), (2, 0), (0, 1)):
            r = np.zeros(9)
            r[3 * b:3 * b + 3] += lm[a] * li
            r[3 * a:3 * a + 3] -= lm[b] * li
            rows.append(r)
    return np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3).T


def mline(axis, off):
    return np.array([1.0, 0, -off]) if axis == "x" else np.array([0, 1.0, -off])


def score(H, white, pitch, dist, grass_pts):
    """(score, visible model points, of which on white). -inf if H is implausible."""
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        return -np.inf, 0, 0
    Hi = np.linalg.inv(H)
    g = np.c_[grass_pts, np.ones(len(grass_pts))] @ Hi.T
    if not (np.all(g[:, 2] > 0) or np.all(g[:, 2] < 0)):
        return -np.inf, 0, 0              # pitch would straddle the horizon
    g = g[:, :2] / g[:, 2:]
    m = RUNOFF_M
    inside = (g[:, 0] > -m) & (g[:, 0] < L + m) & (np.abs(g[:, 1]) < W / 2 + m)
    if inside.mean() < 0.9:
        return -np.inf, 0, 0
    q = np.c_[MODEL_PTS, np.ones(len(MODEL_PTS))] @ H.T
    ok = q[:, 2] * np.sign(np.c_[grass_pts[:1], 1] @ Hi.T)[0, 2] > 0
    q = q[:, :2] / q[:, 2:]
    h, w = pitch.shape
    ok &= (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    qi = q[ok].astype(int)
    vis = pitch[qi[:, 1], qi[:, 0]]
    qi = qi[vis]
    on = dist[qi[:, 1], qi[:, 0]] <= SUPPORT_PX
    return on.sum() - 0.5 * (~on).sum(), len(qi), int(on.sum())


def touch_candidates(lines):
    """Far-touchline candidates: the highest long flat lines (a green ad board can sit above)."""
    return sorted((l for l in lines if flat(l) and np.hypot(*(l[4] - l[3])) > 400),
                  key=lambda l: l[3][1] + l[4][1])[:N_TOUCH]


def hypotheses(lines, arcs, img):
    """Yield (H model->image px, touchline used)."""
    for touch in touch_candidates(lines):
        yield from _hypotheses(lines, arcs, img, touch)


def _hypotheses(lines, arcs, img, touch):
    flats = [l for l in lines if flat(l)]
    steeps = [l for l in lines if not flat(l)]
    others = [l for l in flats if l is not touch and l[3][1] + l[4][1] > touch[3][1] + touch[4][1]]
    T_seg = (touch[3], touch[4])
    T = np.array(touch[:3])
    mt = mline("y", -W / 2)
    for s1, s2 in itertools.combinations(steeps, 2):
        for (k1, x1), (k2, x2) in itertools.permutations(X_SLOTS.items(), 2):
            for o in others:
                for ky, y in Y_SLOTS.items():
                    yield dlt([(T, mt), (np.array(s1[:3]), mline("x", x1)),
                               (np.array(s2[:3]), mline("x", x2)),
                               (np.array(o[:3]), mline("y", y))]), touch
    # midfield: halfway + centre circle. Arc pixels next to the halfway line were removed
    # with the line pixels, so the crossings come from an ellipse fitted to the arc(s).
    arc_sets = arcs + [np.vstack(p) for p in itertools.combinations(arcs, 2)]
    for s in steeps:
        S_ = np.array(s[:3])
        corner = np.cross(T, S_)
        if abs(corner[2]) < 1e-9:
            continue
        corner = corner[:2] / corner[2]
        u = np.array([-S_[1], S_[0]])                       # direction along the line
        for arc in arc_sets:
            if len(arc) < 20:
                continue
            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(arc)
            t = np.radians(ang)
            e = np.linspace(0, 2 * np.pi, 720)
            ell = np.c_[cx + ax1 / 2 * np.cos(e) * np.cos(t) - ax2 / 2 * np.sin(e) * np.sin(t),
                        cy + ax1 / 2 * np.cos(e) * np.sin(t) + ax2 / 2 * np.sin(e) * np.cos(t)]
            d = ell @ S_[:2] + S_[2]
            cross = np.flatnonzero(np.sign(d) != np.sign(np.roll(d, 1)))
            if len(cross) != 2:
                continue
            pa, pb = ell[cross[0]], ell[cross[1]]
            if np.hypot(*(pa - corner)) > np.hypot(*(pb - corner)):
                pa, pb = pb, pa                             # nearer the touchline = y -9.15
            # diamond: both crossings + the arc point farthest from the line on each side
            # (~ (L/2 +- r, 0); refinement fixes the approximation). No 3 points collinear.
            fa = ell[d > 0][np.argmax(d[d > 0])]
            fb = ell[d < 0][np.argmin(d[d < 0])]
            src = np.float32([[L / 2, -F.R_CIRCLE], [L / 2, F.R_CIRCLE],
                              [L / 2 + F.R_CIRCLE, 0], [L / 2 - F.R_CIRCLE, 0]])
            try:
                H0 = cv2.getPerspectiveTransform(src, np.float32([pa, pb, fa, fb])).astype(float)
            except cv2.error:
                continue
            # too rough to score as is (depth scale is off): refine on touch + halfway + circle
            lo, hi = arc.min(0) - 10, arc.max(0) + 10
            pix = F.collect(img, {"touch": (*T_seg[0], *T_seg[1]), "halfway": (*s[3], *s[4]),
                                  "centre": (*lo, *hi)}, np.random.default_rng(0))
            if len(pix) < 3:
                continue
            _, _, Hf = F.fit({"f": pix}, LW=(L, W), H0={"f": np.diag([1 / F.S, 1 / F.S, 1]) @ H0})
            yield np.diag([F.S, F.S, 1]) @ Hf["f"], touch


def candidate_points(lines, arcs):
    """Detected line runs every 15 px plus arc pixels (subsampled): what a good H must explain."""
    pts = []
    for *_, p0, p1, _ in lines:
        n = max(2, int(np.hypot(*(p1 - p0)) / 15))
        pts.append(p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0))
    for a in arcs:
        pts.append(a[::max(1, len(a) // 60)])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def explained(H, cand, shape):
    """Candidate points within ~SUPPORT_PX of the projected model (rendered at 1/4 scale)."""
    h, w = shape[0] // 4, shape[1] // 4
    q = np.c_[MODEL_PTS, np.ones(len(MODEL_PTS))] @ (np.diag([0.25, 0.25, 1]) @ H).T
    good = q[:, 2] > 0 if np.median(q[:, 2]) > 0 else q[:, 2] < 0
    q = q[:, :2] / q[:, 2:]
    canvas = np.zeros((h, w), np.uint8)
    for name in np.unique(MODEL_TAG):
        sel = (MODEL_TAG == name) & good & (np.abs(q).max(1) < 1e5)
        if sel.sum() > 1:
            cv2.polylines(canvas, [q[sel].astype(np.int32)], False, 255, 2)
    c = (cand / 4).astype(int)
    ok = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
    return int((canvas[c[ok, 1], c[ok, 0]] > 0).sum())


def visible_features(H, pitch, dist, radius=6):
    """Rough segments / circle boxes of the markings H says are visible and supported."""
    q = np.c_[MODEL_PTS, np.ones(len(MODEL_PTS))] @ H.T
    q = q[:, :2] / q[:, 2:]
    h, w = pitch.shape
    feats = {}
    for name in np.unique(MODEL_TAG):
        sel = MODEL_TAG == name
        p = q[sel]
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        ok[ok] &= pitch[p[ok, 1].astype(int), p[ok, 0].astype(int)]
        if ok.sum() < 8:
            continue
        on = ok.copy()
        on[ok] = dist[p[ok, 1].astype(int), p[ok, 0].astype(int)] <= radius
        if on.sum() < 8 or on.sum() < 0.4 * ok.sum():
            continue
        key = name.replace("@far", "")
        if key in ("centre", "d_arc"):
            # generous box: if H is a bit small, a tight box clips the real circle and the
            # fit shrinks it further (seen as a drifting centre circle in tracking)
            lo, hi = p[on].min(0), p[on].max(0)
            pad = 10 + 0.3 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            feats[key] = (*lo, *hi)
        else:
            idx = np.flatnonzero(on)
            feats[key] = (*p[idx[0]], *p[idx[-1]])
    return feats


def grass_points(pitch, rng):
    """Up to 400 pitch pixels (16 px grid) for the plausibility test."""
    gy, gx = np.nonzero(pitch[::16, ::16])
    grass = np.c_[gx, gy].astype(float) * 16
    return grass[rng.choice(len(grass), 400, replace=False)] if len(grass) > 400 else grass


def label_frame(img, rng=None):
    """-> dict(H=model->image px or None, feats, score, n_lines, n_arcs, status)."""
    rng = rng or np.random.default_rng(0)
    white, pitch, dist = masks(img)
    lines = line_candidates(white)
    arcs = arc_candidates(white, lines)
    grass = grass_points(pitch, rng)
    out = dict(H=None, feats={}, score=-np.inf, n_lines=len(lines), n_arcs=len(arcs))
    if len(grass) < 20:
        return {**out, "status": "no_pitch"}
    scored = [(score(H, white, pitch, dist, grass)[0], i, H, t)
              for i, (H, t) in enumerate(hypotheses(lines, arcs, img))]
    scored = sorted((x for x in scored if np.isfinite(x[0])), key=lambda x: -x[0])[:TOP_K]
    if not scored:
        return {**out, "status": "no_hypothesis"}
    cand = candidate_points(lines, arcs)
    best = None
    for s1, i, H, touch in scored:
        r = refine(img, H, touch, white, pitch, dist, grass, rng)
        if r is None:
            continue
        # a wrong H can show few model lines and so pay few penalties: also reward
        # explaining the detected lines and arcs
        r["score"] = r["score"] + explained(r["H"], cand, pitch.shape)
        if best is None or r["score"] > best["score"]:
            best = r
    if best is None:
        return {**out, "status": "too_few_features"}
    status = "ok" if best["support"] >= MIN_SUPPORT and best["n_on"] >= MIN_ON else "low_support"
    return {**out, **best, "status": status}


def refine(img, H, touch, white, pitch, dist, grass, rng, radius=6):
    """Name the markings H shows, refit on their pixels; None if too few to fit."""
    feats = visible_features(H, pitch, dist, radius)
    if "touch" not in feats and touch is not None:   # rough midfield H can miss the far line
        feats["touch"] = (*touch[3], *touch[4])
    pix = F.collect(img, feats, rng)
    has_x = any(F.LINES.get(k, ("",))[0] == "x" for k in pix) or "centre" in pix
    if len(pix) < 3 or not has_x:
        return None
    _, _, Hf = F.fit({"f": pix}, LW=(L, W), H0={"f": np.diag([1 / F.S, 1 / F.S, 1]) @ H})
    Hr = np.diag([F.S, F.S, 1]) @ Hf["f"]
    s0, s1 = score(H, white, pitch, dist, grass)[0], score(Hr, white, pitch, dist, grass)[0]
    if s1 < s0:                           # refinement made it worse: keep the closed-form H
        Hr, s1 = H, s0
    res = F.residuals_frame(np.diag([1 / F.S, 1 / F.S, 1]) @ Hr, pix, L, W)
    line_res = [r for k, r in res.items() if k in F.LINES]
    rms = float(np.sqrt(np.mean(np.concatenate(line_res) ** 2)) * F.S) if line_res else np.nan
    _, n_vis, n_on = score(Hr, white, pitch, dist, grass)
    # H*[model] has w = 1/(H^-1 p)_w for pitch pixels p: flip so w > 0 (draw() relies on it)
    Hr = Hr * np.sign((np.linalg.inv(Hr) @ [*grass[0], 1])[2])
    return dict(H=Hr, feats=feats, score=s1, line_rms=rms, n_on=n_on,
                support=round(n_on / max(n_vis, 1), 3))

if __name__ == "__main__":
    import time
    for f in sys.argv[1:]:
        t = time.time()
        r = label_frame(cv2.imread(f))
        print(Path(f).stem, r["status"], f"score {r['score']:.0f}", f"lines {r['n_lines']}",
              f"arcs {r['n_arcs']}", sorted(r["feats"]), f"{time.time() - t:.1f}s")
