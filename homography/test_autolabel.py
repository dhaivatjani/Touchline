"""Self-check: autolabel's H must agree with the hand-segment fit on the 4 hand frames."""
import cv2
import numpy as np

import autolabel as A
import fit_lines as F


def main():
    rng = np.random.default_rng(0)
    imgs = {n: cv2.imread(str(p)) for n, (p, _) in F.FRAMES.items()}
    pix = {n: F.collect(imgs[n], f, rng) for n, (_, f) in F.FRAMES.items()}
    _, _, hand = F.fit(pix, LW=(A.L, A.W))
    gx, gy = np.meshgrid(np.arange(0, A.L / 2 + 1, 2.0), np.arange(-A.W / 2, A.W / 2 + 1, 2.0))
    grid = np.c_[gx.ravel(), gy.ravel()]
    worst = 0
    for n, img in imgs.items():
        r = A.label_frame(img)
        assert r["status"] == "ok", (n, r["status"])
        Hh = np.diag([F.S, F.S, 1]) @ hand[n]
        ph = cv2.perspectiveTransform(grid[None].astype(np.float64), Hh)[0]
        pa = cv2.perspectiveTransform(grid[None].astype(np.float64), r["H"])[0]
        h, w = img.shape[:2]
        _, pitch, _ = A.masks(img)
        vis = (ph[:, 0] >= 0) & (ph[:, 0] < w) & (ph[:, 1] >= 0) & (ph[:, 1] < h)
        vis[vis] &= pitch[ph[vis, 1].astype(int), ph[vis, 0].astype(int)]
        err = np.linalg.norm(ph[vis] - pa[vis], axis=1)
        print(f"{n}: {vis.sum()} grid points, median {np.median(err):.1f} px, "
              f"95% {np.percentile(err, 95):.1f} px")
        worst = max(worst, np.median(err))
    assert worst < 5, f"median disagreement {worst:.1f} px"
    print("ok")


if __name__ == "__main__":
    main()
