"""Self-check: render pitch lines from a known H, L, W and recover them with fit()."""
import cv2
import numpy as np

import fit_lines as F

L_TRUE, W_TRUE = 103.0, 67.0
# A side camera roughly like Camera 4 (model -> image px): far corners high, near edge wide.
H_PX = cv2.getPerspectiveTransform(
    np.float32([[0, -W_TRUE / 2], [L_TRUE / 2, -W_TRUE / 2], [0, 20], [L_TRUE / 2, 20]]),
    np.float32([[500, 150], [2100, 150], [60, 1050], [2340, 1050]])).astype(float)

segs_model = {
    "touch": ((0, -W_TRUE / 2), (L_TRUE / 2, -W_TRUE / 2)),
    "goal": ((0, -W_TRUE / 2), (0, 20)),
    "halfway": ((L_TRUE / 2, -W_TRUE / 2), (L_TRUE / 2, 5)),
    "box_side": ((0, -20.16), (16.5, -20.16)),
    "box_front": ((16.5, -20.16), (16.5, 20.16)),
    "six_side": ((0, -9.16), (5.5, -9.16)),
    "six_front": ((5.5, -9.16), (5.5, 9.16)),
}


def proj(p):
    q = H_PX @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def main():
    img = np.full((1080, 2400, 3), (40, 120, 40), np.uint8)
    feats = {}
    for k, (a, b) in segs_model.items():
        pa, pb = proj(a), proj(b)
        cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), (230, 230, 230), 2)
        # rough segment: shrink 5% at each end, as an eyeballed one would be
        feats[k] = (*(pa + 0.05 * (pb - pa)), *(pb - 0.05 * (pb - pa)))
    assert np.all([0 <= v[i] < (2400, 1080)[i % 2] for v in feats.values() for i in range(4)]), \
        "test camera puts a line off-screen"
    a = np.linspace(0, 2 * np.pi, 300)
    circ = np.array([proj((L_TRUE / 2 + 9.15 * np.cos(t), 9.15 * np.sin(t))) for t in a])
    cv2.polylines(img, [circ.astype(np.int32)], True, (230, 230, 230), 2)
    circ = circ[(circ[:, 0] >= 0) & (circ[:, 0] < 2400) & (circ[:, 1] >= 0) & (circ[:, 1] < 1080)]
    feats["centre"] = (*circ.min(0), *circ.max(0))
    pix = {"synthetic": F.collect(img, feats, np.random.default_rng(0))}
    L, W, Hs = F.fit(pix)
    print(f"L {L:.2f} (true {L_TRUE}), W {W:.2f} (true {W_TRUE})")
    assert abs(L - L_TRUE) < 0.3 and abs(W - W_TRUE) < 0.3
    H = np.diag([F.S, F.S, 1]) @ Hs["synthetic"]
    err = np.linalg.norm(cv2.perspectiveTransform(np.array([[[30.0, 0.0]]]), H)
                         - cv2.perspectiveTransform(np.array([[[30.0, 0.0]]]), H_PX))
    print(f"reprojection error at (30, 0): {err:.2f} px")
    assert err < 2
    print("ok")


if __name__ == "__main__":
    main()
