"""Self-check for eval_gt: clicks made from the hand-fit H must score ~0 against that H,
a shifted H must score worse, and the labeller must score well on the hand frames."""
import cv2
import numpy as np

import autolabel as A
import fit_lines as F
from eval_gt import MODEL, landmark_errors


def main():
    rng = np.random.default_rng(0)
    imgs = {n: cv2.imread(str(p)) for n, (p, _) in F.FRAMES.items()}
    pix = {n: F.collect(imgs[n], f, rng) for n, (_, f) in F.FRAMES.items()}
    _, _, hand = F.fit(pix, LW=(A.L, A.W))
    for n, img in imgs.items():
        H = np.diag([F.S, F.S, 1]) @ hand[n]
        h, w = img.shape[:2]
        clicks = {}
        for name, m in MODEL.items():                  # "perfect clicks" from the hand fit
            q = H @ [*m, 1]
            x, y = q[:2] / q[2]
            if q[2] * np.sign(H[2, 2]) > 0 and 0 <= x < w and 0 <= y < h:
                clicks[name] = (x, y)
        assert len(clicks) >= 3, (n, len(clicks))
        assert max(e for e, _ in landmark_errors(H, clicks).values()) < 1e-6
        shifted = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1.0]]) @ H
        assert min(e for e, _ in landmark_errors(shifted, clicks).values()) > 19
        r = A.label_frame(img)
        med = np.median([e for _, e in landmark_errors(r["H"], clicks).values()])
        print(f"{n}: {len(clicks)} landmarks, labeller median {med:.3f} m")
        assert med < 0.3
    print("ok")


if __name__ == "__main__":
    main()
