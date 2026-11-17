"""Least-squares pitch homography from all visible lines and circles.

Per frame, rough named segments / circle boxes (eyeballed for now; the auto line
labeller will supply them later) select the white pixels of each pitch marking.
One joint fit then solves every frame's H (model -> image) plus the shared pitch
length L and width W, minimising pixel distance to the projected model lines and
circles (Sampson distance). Band edges are *not* used in the fit: they are checked
afterwards against k*L/18 as independent validation.

Model coords (m): x from the visible goal line towards halfway (mirrored per frame
by H as needed), y from the goal centre towards the near touchline.
Outputs: fit_lines/<frame>_overlay.jpg, printed L, W and per-feature RMS.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from band_check import white_mask

HERE = Path(__file__).resolve().parent
OUT = HERE / "fit_lines"
S = 1000.0                  # image px are divided by this inside the fit (conditioning)
TOL = 12                    # px around a rough segment that count as its pixels
MAX_PTS = 300               # per feature, enough for a line and keeps the fit fast
R_CIRCLE = 9.15
L0, W0 = 101.0, 68.0

# feature -> ("x" | "y", offset) for lines; circles take a centre function of (L, W).
LINES = {
    "goal": ("x", lambda L, W: 0.0), "halfway": ("x", lambda L, W: L / 2),
    "touch": ("y", lambda L, W: -W / 2), "near_touch": ("y", lambda L, W: W / 2),
    "box_side": ("y", lambda L, W: -20.16), "box_near": ("y", lambda L, W: 20.16),
    "box_front": ("x", lambda L, W: 16.5),
    "six_side": ("y", lambda L, W: -9.16), "six_near": ("y", lambda L, W: 9.16),
    "six_front": ("x", lambda L, W: 5.5),
}
CIRCLES = {"centre": lambda L, W: (L / 2, 0.0), "d_arc": lambda L, W: (11.0, 0.0)}

C = HERE / "check_frames"
FRAMES = {  # name: (image, {feature: segment (x1,y1,x2,y2) or circle box (x1,y1,x2,y2)})
    "M03_0281.4s": (C / "M03_0281.4s.jpg", {
        "touch": (620, 182, 2260, 180), "goal": (614, 182, 0, 720),
        "box_side": (464, 318, 1114, 318), "box_front": (1114, 318, 1028, 1080),
        "six_side": (270, 490, 540, 490), "six_front": (540, 490, 66, 1080),
        "halfway": (2268, 182, 2400, 246), "d_arc": (1086, 522, 1308, 1032),
    }),
    "screenshot": (HERE / "screenshot_2.webp", {
        "touch": (0, 172, 1630, 172), "goal": (1630, 172, 2000, 414),
        "box_side": (1275, 270, 1780, 270), "box_front": (1275, 270, 1630, 930),
        "six_side": (1765, 392, 1965, 392), "six_front": (1765, 392, 2000, 585),
        "halfway": (310, 172, 0, 350), "centre": (0, 385, 165, 625),
        "d_arc": (1238, 412, 1530, 745),
    }),
    "M01_0080.3s": (C / "M01_0080.3s.jpg", {
        "touch": (0, 186, 1218, 168), "goal": (1218, 168, 1800, 1074),
        "box_side": (780, 250, 1260, 240), "box_front": (780, 250, 558, 816),
        "box_near": (558, 816, 1602, 764),
        "six_side": (1134, 322, 1308, 318), "six_front": (1134, 322, 1200, 538),
        "six_near": (1200, 538, 1446, 532), "d_arc": (540, 348, 738, 534),
    }),
    "M13_0059.3s": (C / "M13_0059.3s.jpg", {
        "touch": (0, 210, 1482, 206), "goal": (1482, 206, 2016, 1080),
        "box_side": (902, 352, 1566, 348), "box_front": (902, 352, 624, 1080),
        "six_side": (1404, 534, 1678, 532), "six_front": (1404, 534, 1560, 1080),
        "d_arc": (450, 576, 816, 1080),
    }),
}


def seg_dist(pts, seg):
    """Distance of pts to a segment, inf beyond its ends."""
    p1, p2 = np.array(seg[:2], float), np.array(seg[2:], float)
    d = p2 - p1
    L = np.linalg.norm(d)
    d /= L
    rel = pts - p1
    along = rel @ d
    perp = np.abs(rel @ np.array([-d[1], d[0]]))
    return np.where((along > 0) & (along < L), perp, np.inf)


def collect(img, feats, rng):
    """Feature -> (N, 2) white pixels belonging to it."""
    ys, xs = np.nonzero(white_mask(img))
    pts = np.stack([xs, ys], 1).astype(float)
    segs = {k: v for k, v in feats.items() if k in LINES}
    # only circles visible (e.g. just the centre circle): no line pixels to exclude
    near_any = (np.min([seg_dist(pts, s) for s in segs.values()], 0) < TOL) if segs else np.zeros(len(pts), bool)
    out = {}
    for k, v in feats.items():
        if k in LINES:
            sel = pts[seg_dist(pts, v) < TOL]
        else:
            x1, y1, x2, y2 = v
            inbox = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            sel = pts[inbox & ~near_any]
        if len(sel) <= 20:                 # marking not really there: leave it out
            print(f"  {k}: only {len(sel)} white pixels, dropped")
            continue
        if len(sel) > MAX_PTS:
            sel = sel[rng.choice(len(sel), MAX_PTS, replace=False)]
        out[k] = sel / S
    return out


def model_line(k, L, W):
    axis, off = LINES[k]
    o = off(L, W)
    return np.array([1.0, 0.0, -o]) if axis == "x" else np.array([0.0, 1.0, -o])


def model_conic(k, L, W):
    cx, cy = CIRCLES[k](L, W)
    return np.array([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx * cx + cy * cy - R_CIRCLE ** 2]])


def residuals_frame(H, pix, L, W):
    Hi = np.linalg.inv(H)
    res = {}
    for k, p in pix.items():
        ph = np.c_[p, np.ones(len(p))]
        if k in LINES:
            l = Hi.T @ model_line(k, L, W)               # image line
            res[k] = ph @ l / np.hypot(l[0], l[1])
        else:
            Ci = Hi.T @ model_conic(k, L, W) @ Hi        # image conic
            Cp = ph @ Ci
            f = np.sum(Cp * ph, 1)
            res[k] = f / (2 * np.hypot(Cp[:, 0], Cp[:, 1]))  # Sampson distance
    return res


def init_H(pix, L, W):
    """Line DLT: model lines l_m ~ H^T l_img. Image lines from a plain fit."""
    rows = []
    for k, p in pix.items():
        if k not in LINES:
            continue
        vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        li = np.array([vy, -vx, vx * y0 - vy * x0])
        lm = model_line(k, L, W)
        # cross(lm, G li) = 0 with G = H^T
        for a, b in ((1, 2), (2, 0), (0, 1)):
            r = np.zeros(9)
            r[3 * b:3 * b + 3] += lm[a] * li
            r[3 * a:3 * a + 3] -= lm[b] * li
            rows.append(r)
    G = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = G.T
    return H / H[2, 2]


def fit(frames_pix, LW=None, H0=None):
    """Joint fit. LW=(L, W) holds the pitch size fixed; H0 {name: H} replaces the line-DLT init."""
    names = list(frames_pix)
    x0 = [] if LW else [L0, W0]
    for n in names:
        h = H0[n] if H0 else init_H(frames_pix[n], L0, W0)
        x0 += list((h / h[2, 2]).ravel()[:8])
    off = 0 if LW else 2

    def unpack(x):
        Hs = [np.append(x[off + 8 * i: off + 8 + 8 * i], 1.0).reshape(3, 3) for i in range(len(names))]
        return (*(LW or x[:2]), Hs)

    def fun(x):
        L, W, Hs = unpack(x)
        return np.concatenate([r for H, n in zip(Hs, names)
                               for r in residuals_frame(H, frames_pix[n], L, W).values()])

    sol = least_squares(fun, np.array(x0), loss="huber", f_scale=2 / S, x_scale="jac")
    L, W, Hs = unpack(sol.x)
    return L, W, dict(zip(names, Hs))


def band_edges(img, H, L):
    """Brightness steps along x in the grass strip between far touchline and box side."""
    ppm = 10
    Sm = np.array([[ppm / S, 0, 0], [0, ppm / S, 34 * ppm / S], [0, 0, 1]]) * S  # model->top px
    Sm[2] = [0, 0, 1]
    top = cv2.warpPerspective(img, np.diag([S, S, 1]) @ H @ np.linalg.inv(Sm),
                              (int(L / 2 * ppm) + 1, 14 * ppm),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    v = np.where(cv2.inRange(hsv, (30, 60, 40), (90, 255, 255)) > 0, hsv[..., 2].astype(float), np.nan)
    with np.errstate(all="ignore"):
        prof = np.nanmedian(v[2 * ppm:], 0)          # skip 2 m by the touchline
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(len(prof)), np.flatnonzero(ok), prof[ok])
    d = np.gradient(cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 6).ravel())
    edges, last = [], -99
    for i in np.argsort(-np.abs(d)):
        if abs(d[i]) < 0.25:
            break
        if all(abs(i - e) > 2 * ppm for e in edges) and 2 * ppm < i < len(d) - ppm:  # skip goal-line bleed
            edges.append(i)
    edges = np.sort(np.array(edges)) / ppm
    visible = ok.mean()
    return edges, visible


def draw(img, H, L, W):
    ov = img.copy()
    P = np.diag([S, S, 1]) @ H

    def proj(pts):
        q = (P @ np.c_[pts, np.ones(len(pts))].T).T
        good = q[:, 2] > 1e-9
        return q[good, :2] / q[good, 2:], good

    def poly(pts, col, t):
        q, good = proj(pts)
        if good.all() and len(q) > 1:
            cv2.polylines(ov, [q.astype(np.int32)], False, col, t, cv2.LINE_AA)

    t = np.linspace(0, 1, 60)[:, None]
    seg = lambda a, b: np.array(a) + t * (np.array(b) - np.array(a))
    for k in range(10):                                 # band edges k*L/18
        poly(seg((k * L / 18, -W / 2), (k * L / 18, W / 2)), (0, 200, 255), 1)
    for a, b in [((0, -W / 2), (0, W / 2)), ((L / 2, -W / 2), (L / 2, W / 2)),
                 ((0, -W / 2), (L / 2, -W / 2)), ((0, W / 2), (L / 2, W / 2)),
                 ((0, -20.16), (16.5, -20.16)), ((16.5, -20.16), (16.5, 20.16)),
                 ((16.5, 20.16), (0, 20.16)), ((0, -9.16), (5.5, -9.16)),
                 ((5.5, -9.16), (5.5, 9.16)), ((5.5, 9.16), (0, 9.16))]:
        poly(seg(a, b), (255, 0, 255), 2)
    a = np.linspace(0, 2 * np.pi, 200)
    poly(np.c_[L / 2 + R_CIRCLE * np.cos(a), R_CIRCLE * np.sin(a)], (255, 0, 255), 2)
    a = np.linspace(-0.93, 0.93, 60)                    # D: part outside the box
    poly(np.c_[11 + R_CIRCLE * np.cos(a), R_CIRCLE * np.sin(a)], (255, 0, 255), 2)
    return ov


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    imgs = {n: cv2.imread(str(p)) for n, (p, _) in FRAMES.items()}
    pix = {n: collect(imgs[n], f, rng) for n, (_, f) in FRAMES.items()}
    L, W, Hs = fit(pix)
    print(f"pitch L = {L:.2f} m, W = {W:.2f} m, band = L/18 = {L / 18:.2f} m")
    for n, H in Hs.items():
        res = residuals_frame(H, pix[n], L, W)
        rms = {k: float(np.sqrt(np.mean((r * S) ** 2))) for k, r in res.items()}
        edges, vis = band_edges(imgs[n], H, L)
        k = np.round(edges / (L / 18))
        width, phase = np.polyfit(k, edges, 1)          # edge_k = phase + k * width
        print(f"\n{n}: RMS px " + " ".join(f"{k}={v:.1f}" for k, v in rms.items()))
        print(f"  band edges (m): {np.round(edges, 1).tolist()}")
        print(f"  linear fit: band width {width:.2f} m, phase {phase:+.2f} m, "
              f"max dev {np.abs(edges - phase - k * width).max():.2f} m")
        cv2.imwrite(str(OUT / f"{n}_overlay.jpg"), draw(imgs[n], H, L, W))


if __name__ == "__main__":
    main()
