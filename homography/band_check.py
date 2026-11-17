"""Band / pitch-size check: H from hand-picked box lines, then measure the rest.

Per frame, rough endpoints of named lines (eyeballed) are snapped to the white
pixels nearby with a robust line fit. Box-line intersections (standard 40.32 x 16.5
and 18.32 x 5.5 boxes) give H without assuming pitch length or width; the
touchline and halfway line then *measure* W/2 and L/2, and the warped grass gives
the mowing-band profile along the length.

Model coords (m): x from the goal line towards halfway, y from the goal centre
towards the near touchline (far touchline is negative).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "band_check"
PX_PER_M = 10

BOX = {"box_side": -20.16, "six_side": -9.16}           # y of the far side lines
FRAMES = {
    # name: (image, {line: (x1, y1, x2, y2) rough, in that image's pixels})
    "M03_0281.4s": (HERE / "check_frames" / "M03_0281.4s.jpg", {
        "touch": (620, 182, 2400, 180), "goal": (614, 182, 0, 720),
        "box_side": (464, 318, 1114, 318), "box_front": (1114, 318, 1028, 1080),
        "six_side": (270, 490, 540, 490), "six_front": (540, 490, 66, 1080),
    }),
    "screenshot": (HERE / "screenshot_2.webp", {
        "touch": (0, 172, 1630, 172), "goal": (1630, 172, 2000, 414),
        "box_side": (1275, 270, 1780, 270), "box_front": (1275, 270, 1630, 930),
        "six_side": (1765, 392, 1965, 392), "six_front": (1765, 392, 2000, 585),
        "halfway": (310, 172, 0, 350),
    }),
}


def white_mask(img):
    # Far lines are 1-2 px and blend with grass (V~120 vs ~85), so use local contrast.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    # Thin far lines take on grass saturation: bright pitches (grass S~185) give S 120-145.
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    s_max = max(110, np.median(hsv[..., 1][grass]) - 35) if grass.any() else 110
    return (tophat > 15) & (hsv[..., 1] < s_max)   # far touchline peaks at ~22 (M01)


def snap(white, seg, tol=10):
    """Robust line (a, b, c), ax+by+c=0, through white pixels near the rough segment."""
    x1, y1, x2, y2 = seg
    ys, xs = np.nonzero(white)
    d = np.array([x2 - x1, y2 - y1], float)
    L = np.linalg.norm(d)
    d /= L
    rel = np.stack([xs - x1, ys - y1], 1)
    along = rel @ d
    perp = np.abs(rel @ np.array([-d[1], d[0]]))
    keep = (perp < tol) & (along > 0) & (along < L)
    pts = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    assert len(pts) > 30, f"too few white pixels near {seg}"
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, vx * y0 - vy * x0]), len(pts)


def meet(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def model_of_line(H_inv, line, img_shape):
    """Sample a fitted image line, map to model coords."""
    a, b, c = line
    xs = np.linspace(0, img_shape[1] - 1, 200)
    ys = -(a * xs + c) / b if abs(b) > abs(a) else None
    if ys is None:
        ys = np.linspace(0, img_shape[0] - 1, 200)
        xs = -(b * ys + c) / a
    ok = (ys >= 0) & (ys < img_shape[0]) & (xs >= 0) & (xs < img_shape[1])
    pts = np.stack([xs[ok], ys[ok]], 1).reshape(-1, 1, 2).astype(np.float64)
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


def check(name, path, segs):
    img = cv2.imread(str(path))
    assert img is not None, path
    white = white_mask(img)
    lines = {k: snap(white, s)[0] for k, s in segs.items()}

    img_pts = np.array([meet(lines["goal"], lines["box_side"]),
                        meet(lines["box_side"], lines["box_front"]),
                        meet(lines["goal"], lines["six_side"]),
                        meet(lines["six_side"], lines["six_front"])])
    mdl_pts = np.array([[0, BOX["box_side"]], [16.5, BOX["box_side"]],
                        [0, BOX["six_side"]], [5.5, BOX["six_side"]]])
    H, _ = cv2.findHomography(mdl_pts, img_pts)       # model -> image
    H_inv = np.linalg.inv(H)

    res = {}
    t = model_of_line(H_inv, lines["touch"], img.shape)
    res["W/2 from touchline (m)"] = f"{-t[:, 1].mean():.2f} (spread {np.ptp(t[:, 1]):.2f})"
    if "halfway" in lines:
        h = model_of_line(H_inv, lines["halfway"], img.shape)
        res["L/2 from halfway (m)"] = f"{h[:, 0].mean():.2f} (spread {np.ptp(h[:, 0]):.2f})"

    # Top-down warp of the strip between far touchline and box side line.
    x_max = 60
    S = np.array([[PX_PER_M, 0, 0], [0, PX_PER_M, 40 * PX_PER_M], [0, 0, 1]])  # model -> top
    top = cv2.warpPerspective(img, H @ np.linalg.inv(S), (x_max * PX_PER_M, 40 * PX_PER_M),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                              borderValue=(0, 0, 0))
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255)) > 0
    v = np.where(grass, hsv[..., 2].astype(float), np.nan)
    with np.errstate(all="ignore"):
        prof = np.nanmedian(v, axis=0)                 # per model-x column
    cv2.imwrite(str(OUT / f"{name}_top.jpg"), top)

    # Overlay of the model on the image as a sanity check of H.
    ov = img.copy()
    W2 = float(-t[:, 1].mean())
    model_lines = [((0, -W2), (0, W2)), ((0, -20.16), (16.5, -20.16)),
                   ((16.5, -20.16), (16.5, 20.16)), ((0, -9.16), (5.5, -9.16)),
                   ((5.5, -9.16), (5.5, 9.16)), ((0, -W2), (x_max, -W2))]
    for p, q in model_lines:
        pq = cv2.perspectiveTransform(np.array([[p, q]], np.float64), H).reshape(-1, 2)
        cv2.line(ov, tuple(map(int, pq[0])), tuple(map(int, pq[1])), (255, 0, 255), 2)
    for p in img_pts:
        cv2.circle(ov, tuple(map(int, p)), 8, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{name}_overlay.jpg"), ov)
    return res, prof


def main():
    OUT.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(FRAMES), 1, figsize=(12, 3 * len(FRAMES)))
    for ax, (name, (path, segs)) in zip(np.atleast_1d(axes), FRAMES.items()):
        res, prof = check(name, path, segs)
        print(name, res)
        ax.plot(np.arange(len(prof)) / PX_PER_M, prof)
        ax.set_title(name)
        ax.set_xticks(np.arange(0, 61, 5))
        ax.grid(alpha=0.4)
        ax.set_xlabel("metres from goal line")
    fig.tight_layout()
    fig.savefig(OUT / "profiles.png", dpi=90)


if __name__ == "__main__":
    main()
