"""Ball track from per-frame candidates: one ball per frame, false hits gated out, short gaps filled.

Works on the ball's ground point (bottom centre of its box) in image pixels:
- constant-velocity Kalman filter; each frame takes the candidate nearest the prediction inside a
  gate that widens while the ball is missing (occlusion, blur);
- a track starts only from a confident detection (START_CONF) and ends after MAX_GAP s unseen;
- segments with fewer than MIN_DETS detections are dropped (white boots, centre spot, crowd);
- gaps up to MAX_GAP s inside a segment are filled by linear interpolation (kind "fill").
"""
import numpy as np

START_CONF = 0.5
GATE_PX = 60          # gate radius right after a detection
GATE_GROW = 25        # extra px per frame without one
GATE_MAX = 400
MAX_GAP = 0.5         # s
MIN_DETS = 3


def kalman_step(x, P, dt, z=None, q=2000.0, r=9.0):
    """x = [u, v, du, dv] (px, px/s). Predict over dt, then update with z = (u, v) if given."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    x, P = F @ x, F @ P @ F.T + q * G @ G.T
    if z is not None:
        Hm = np.eye(2, 4)
        S = Hm @ P @ Hm.T + r * np.eye(2)
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ (np.asarray(z, float) - Hm @ x)
        P = (np.eye(4) - K @ Hm) @ P
    return x, P


def track(frames):
    """frames: [(t, [(conf, u, v), ...])] in time order -> [(t, u, v, kind)], kind det | fill | None."""
    out = [[t, None, None, None] for t, _ in frames]
    seg, segs = [], []          # current segment: indices of frames with a detection
    x = P = None
    last_t, miss = None, 0
    for i, (t, cands) in enumerate(frames):
        if x is not None and t - last_t > MAX_GAP:
            segs.append(seg)
            seg, x = [], None
        if x is None:
            strong = [c for c in cands if c[0] >= START_CONF]
            if strong:
                c = max(strong)
                x, P = np.array([c[1], c[2], 0.0, 0.0]), np.diag([25.0, 25.0, 1e6, 1e6])
                out[i][1:] = [c[1], c[2], "det"]
                seg, last_t, miss, prev_t = [i], t, 0, t
            continue
        dt = max(t - prev_t, 1e-3)
        xp, _ = kalman_step(x, P, dt)
        gate = min(GATE_PX + GATE_GROW * miss, GATE_MAX)
        near = [(np.hypot(c[1] - xp[0], c[2] - xp[1]), c) for c in cands]
        near = [n for n in near if n[0] <= gate]
        if near:
            c = min(near, key=lambda n: n[0] / (0.5 + n[1][0]))[1]   # close and confident
            x, P = kalman_step(x, P, dt, (c[1], c[2]))
            out[i][1:] = [c[1], c[2], "det"]
            seg.append(i)
            last_t, miss = t, 0
        else:
            x, P = kalman_step(x, P, dt)
            miss += 1
        prev_t = t
    segs.append(seg)

    for s in segs:
        if len(s) < MIN_DETS:
            for i in s:
                out[i][1:] = [None, None, None]
            continue
        for a, b in zip(s, s[1:]):
            ta, tb = out[a][0], out[b][0]
            for k in range(a + 1, b):
                w = (out[k][0] - ta) / (tb - ta)
                out[k][1:] = [out[a][1] + w * (out[b][1] - out[a][1]), out[a][2] + w * (out[b][2] - out[a][2]), "fill"]
    return [tuple(o) for o in out]


def airborne(t, xy, max_speed=35.0, win=3):
    """Pitch positions (n,2) over time -> bool (n,): speed over +-win samples above max_speed m/s.
    shortcut: a lofted ball projects too far up the pitch and so looks too fast; real height needs
    a ballistic fit, which is not worth it for the prototype."""
    t, xy = np.asarray(t, float), np.asarray(xy, float)
    n = len(t)
    air = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - win), min(n - 1, i + win)
        if t[b] > t[a] and np.isfinite(xy[[a, b]]).all():
            air[i] = np.hypot(*(xy[b] - xy[a])) / (t[b] - t[a]) > max_speed
    return air
