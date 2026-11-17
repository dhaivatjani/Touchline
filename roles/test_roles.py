"""Self-check: python roles/test_roles.py"""
import numpy as np

import roles as R

rng = np.random.default_rng(0)
red, green, yellow, grass = map(np.array, ([45, 60, 40], [40, -40, 30], [85, -10, 75], [45, -35, 35]))

# seed: two noisy teams, one stray red player in the green half, one referee
cols = np.vstack([red + rng.normal(0, 2, (10, 3)), green + rng.normal(0, 2, (10, 3)), red, yellow])
sides = np.array([False] * 10 + [True] * 10 + [True, False])
c, thr = R.seed(cols, sides)
assert np.linalg.norm(c[0] - red) < 4 and np.linalg.norm(c[1] - green) < 4, c
assert R.assign(red + 1, c, thr) == "A" and R.assign(green - 1, c, thr) == "B"
assert R.assign(yellow, c, thr) == "O" and R.assign(None, c, thr) is None

# kit_colour: a green shirt on green grass must survive the grass ring removal
img = np.tile(grass, (100, 100, 1)).astype(np.float32)
img[30:60, 40:60] = green + [0, -20, -20]          # darker, bluer green kit
got = R.kit_colour(img, (40, 25, 60, 95))
assert got is not None and np.linalg.norm(got - (green + [0, -20, -20])) < 1, got
print("ok roles")

# to_pitch: same spot whether the model anchored on the left goal or the right one
import pitch as P
H_left = np.array([[20.0, 0, 100], [0, 10, 500], [0, 0, 1]])            # model +x runs right
H_right = np.array([[-20.0, 0, 100 + 20 * P.L], [0, 10, 500], [0, 0, 1]])  # anchored on the right goal
feet = np.array([[100 + 20 * 30, 500 + 10 * 5]])                       # 30 m from the left goal, 5 m to the near side
H_mirror = np.array([[20.0, 0, 100], [0, -10, 500 + 10 * 34], [0, 0, 1]])  # anchored on the mirror image
feet = np.array([[100 + 20 * 30, 500 + 10 * (5 + 34)]])                  # model y measured from the far line
H_left = np.array([[20.0, 0, 100], [0, 10, 500 + 10 * 34], [0, 0, 1]])
H_right = np.array([[-20.0, 0, 100 + 20 * P.L], [0, 10, 500 + 10 * 34], [0, 0, 1]])
for Hm in (H_left, H_right, H_mirror):
    assert np.allclose(P.to_pitch(Hm, feet), [[30, 5]]), P.to_pitch(Hm, feet)
print("ok pitch")

# split_other: touchline -> linesman, inside a box -> keeper, else referee
assert P.split_other(40, -33) == "linesman" and P.split_other(70, 33.5) == "linesman"
assert P.split_other(4, 2) == "keeper" and P.split_other(P.L - 10, -15) == "keeper"
assert P.split_other(52, 5) == "referee" and P.split_other(30, -25) == "referee"
print("ok split")

# ball.track: follow a ball moving 300 px/s past a decoy, fill a short gap, drop a lone blip
import ball as B
fr = []
for i in range(60):                       # 30 fps, 2 s
    t = i / 30
    c = [(0.8, 100 + 300 * t, 500.0)]
    if 20 <= i < 26:
        c = []                            # 0.2 s occlusion -> filled
    if i == 10:
        c.append((0.9, 1500.0, 200.0))    # decoy far from the ball
    if i >= 50:
        c = []                            # ball gone for good
    fr.append((t, c))
fr.append((3.0, [(0.6, 800.0, 800.0)]))   # lone blip after a long gap -> dropped
out = B.track(fr)
assert all(o[3] == "det" and abs(o[1] - (100 + 300 * o[0])) < 1 for o in out[:20]), out[:20]
assert out[10][1] < 300                   # decoy ignored
assert all(o[3] == "fill" and abs(o[1] - (100 + 300 * o[0])) < 1 for o in out[20:26])
assert all(o[3] is None for o in out[50:]), out[50:]
air = B.airborne([0, .1, .2, .3, .4, .5, .6], [[0, 0], [1, 0], [2, 0], [3, 0], [12, 0], [21, 0], [22, 0]], win=1)
assert not air[1] and air[4], air
print("ok ball")

# shape: templates, formation matching, attack direction
import shape as S
assert all(len(S.template(f)) == 10 for f in S.FORMATIONS)
rng = np.random.default_rng(1)
for f in S.FORMATIONS:
    real = S.template(f) * [40, 30] + [20, 0] + rng.normal(0, 0.03, (10, 2)) * [40, 30]   # metres-ish
    assert S.match_formation(real)[0] == f, (f, S.match_formation(real))
hits = sum(S.match_formation(np.delete(S.template(f) * [40, 30] + rng.normal(0, 1, (10, 2)), [0, 9], 0))[0] == f
           for f in S.FORMATIONS)
print(f"  formation from 8 of 10 noisy players: {hits}/{len(S.FORMATIONS)} right")
# team A defends the left goal: its deepest player is deeper in the left half, B's highest is higher on the right
fr = {0: {"A": np.array([[10, 0], [30, 5]]), "B": np.array([[20, 0], [35, 5]])},
      1: {"A": np.array([[60, 0], [80, 5]]), "B": np.array([[70, 0], [95, 5]])}}
assert S.directions(fr)[0] == {"A": 1, "B": -1}
assert np.allclose(S.orient([[100, 10]], -1), [[5, -10]])
print("ok shape")
