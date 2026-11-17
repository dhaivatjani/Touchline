"""One demo video for a stretch: footage, minimap, team stats, live charts.

  python roles/demo.py M04 47.5 60     (needs run_roles live, pitch.py and shape.py outputs)

Layout (1920 x 1080): title bar; footage with role boxes + ball ring (left); per-team panel with
formation (last 10 s majority vs the one set in game), width, depth, defensive line, compactness
(right); minimap with players, ball trail, team hulls and defensive lines (bottom left); defensive
line and width over time with a moving cursor (bottom right). -> roles/out/<stem>_demo.mp4
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

import clock
import pitch as PT
import shape as SH
from run_roles import video_clip

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WIDTH, HEIGHT = 1920, 1080
BG, PANEL, INK, MUTED, GRID = (24, 24, 28), (36, 36, 42), (235, 235, 235), (150, 150, 158), (70, 70, 78)
TOP = 56
FOOT = (0, TOP, 1280, 576)                 # x, y, w, h
SIDE = (1280, TOP, 640, 576)
MAP_Y = TOP + 576 + 8
MAP_HEAD = 26                              # title strip above the minimap
MAP_PX = (HEIGHT - MAP_Y - MAP_HEAD - 8) / (PT.W + 2 * PT.MARGIN)
CHART = (720, MAP_Y, WIDTH - 720 - 16, HEIGHT - MAP_Y - 8)
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"
FORM_WIN = 10.0
OFFICIAL_NAMES = {"referee": "Referee", "linesman": "Linesman", "keeper": "Keeper"}


def vivid(bgr):
    """Kit colour brightened for a dark UI (dark kits vanish on it); greys stay grey."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    hsv[2] = max(hsv[2], 215)
    if hsv[1] >= 40:
        hsv[1] = max(hsv[1], 140)
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def load(stem):
    rd = lambda p: list(csv.DictReader(open(OUT / p, newline="")))
    rows = [r for r in rd(f"{stem}_pitch.csv")]
    ball = {r["frame"]: r for r in rd(f"{stem}_balltrack.csv")}
    shape = rd(f"{stem}_shape.csv")
    colour = {k: tuple(v) for k, v in json.loads((OUT / f"{stem}_teams.json").read_text()).items()}
    dirs = json.loads((OUT / f"{stem}_dirs.json").read_text())
    return rows, ball, shape, colour, dirs


def smooth_series(shape, team, key):
    rs = [r for r in shape if r["team"] == team]
    t = np.array([float(r["t"]) for r in rs])
    v = np.array([float(r[key]) for r in rs])
    return t, np.array([np.median(v[np.abs(t - ti) <= 0.5]) for ti in t])


def at(t_arr, v_arr, t):
    i = np.searchsorted(t_arr, t, side="right") - 1
    return None if i < 0 or t - t_arr[i] > 1.0 else v_arr[i]


def chart_layer(series, colour, names, t0, t1, rect, title, unit):
    """Static chart image (BGR) + a function mapping (t, v) to pixel coords."""
    x, y, w, h = rect
    img = np.full((h, w, 3), PANEL, np.uint8)
    vals = np.concatenate([v for _, v in series.values()])
    lo, hi = np.floor(vals.min() / 10) * 10, np.ceil(vals.max() / 10) * 10
    pad_l, pad_r, pad_t, pad_b = 52, 16, 34, 26
    px = lambda tt: pad_l + (tt - t0) / (t1 - t0) * (w - pad_l - pad_r)
    py = lambda vv: pad_t + (hi - vv) / max(hi - lo, 1) * (h - pad_t - pad_b)
    ticks = np.linspace(lo, hi, 3)
    for v in ticks:
        cv2.line(img, (pad_l, int(py(v))), (w - pad_r, int(py(v))), GRID, 1)
    for team, (tt, vv) in series.items():
        pts = np.c_[px(tt), py(vv)].astype(np.int32)
        dim = tuple(int(c * 0.45 + p * 0.55) for c, p in zip(colour[team], PANEL))
        cv2.polylines(img, [pts], False, dim, 2, cv2.LINE_AA)
    labels = [(title, (12, 6), INK, 17, True)] + [
        ("● " + names[k], (w - 260 + 120 * i, 8), colour[k][::-1], 15, False) for i, k in enumerate(("A", "B"))]
    labels = labels + [(f"{v:.0f} {unit}", (6, int(py(v)) - 10), MUTED, 13, False)
                                                  for v in ticks]
    return img, px, py, labels


def team_names(colour):
    return {k: f"Team {k}" for k in colour if k in "AB"}


def team_side(colour):
    """user | opponent per team (the user's side wears red) - only to look up the set formation."""
    return {k: ("user" if SH.is_red(c) else "opponent") for k, c in colour.items() if k in "AB"}


def main():
    match, start = sys.argv[1], float(sys.argv[2])
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    stem = f"{match}_live{start:g}"
    rows, ball, shape, colour, dirs = load(stem)
    names, side = team_names(colour), team_side(colour)
    colour = {k: vivid(c) for k, c in colour.items()}
    truth = {}
    with open(PT.ROOT / "dataset" / "formations.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["match"] == match:
                truth[r["team"]] = r["formation"]

    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    shp = {}
    for r in shape:
        shp.setdefault(r["frame"], {})[r["team"]] = r
    form_hist = {k: [(float(r["t"]), r["formation"]) for r in shape if r["team"] == k and r["formation"]]
                 for k in "AB"}
    series = {key: {k: smooth_series(shape, k, key) for k in "AB"} for key in ("line", "width", "depth", "hull", "n")}

    order, frames = video_clip(match, start, dur)
    t0, t1 = order[0][1], order[-1][1]
    fps = (len(order) - 1) / (t1 - t0) if t1 > t0 else 25.0
    cx, cy, cw, ch = CHART
    half_h = (ch - 8) // 2
    charts = [chart_layer({k: series["line"][k] for k in "AB"}, colour, names, t0, t1, (cx, cy, cw, half_h),
                          "Defensive line - metres from own goal", "m"),
              chart_layer({k: series["width"][k] for k in "AB"}, colour, names, t0, t1, (cx, cy + half_h + 8, cw, half_h),
                          "Team width - metres", "m")]

    PT.MAP_PX = MAP_PX                       # pitch.minimap / P read it at call time
    base, P = PT.minimap()
    base = cv2.copyMakeBorder(base, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    map_x = (CHART[0] - base.shape[1]) // 2

    vw = cv2.VideoWriter(str(OUT / f"{stem}_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    f_title, f_big, f_med, f_small = font(26, True), font(30, True), font(19), font(15)
    game_clock = "--:--"
    for k, (name, t, img) in enumerate(frames()):
        canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        if k % 10 == 0:
            m = clock.read(img)
            if m is not None:
                game_clock = f"{int(m):02d}:{round((m % 1) * 60) % 60:02d}"

        # footage
        b = ball.get(name)
        for r in by_frame.get(name, []):
            c = colour.get(r["role2"]) or PT.OFFICIAL.get(r["role2"], (128, 128, 128))
            cv2.rectangle(img, (int(float(r["x1"])), int(float(r["y1"]))), (int(float(r["x2"])), int(float(r["y2"]))),
                          c, 3)
        if b and b["kind"]:
            cv2.circle(img, (int(float(b["u"])), int(float(b["v"])) - 8), 24, (0, 255, 255), 4 if b["kind"] == "det" else 2)
        fx, fy, fw, fh = FOOT
        canvas[fy:fy + fh, fx:fx + fw] = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)

        # minimap: hulls, defensive lines, players, ball trail
        mm = base.copy()
        layer = mm.copy()
        teams_xy = {}
        for r in by_frame.get(name, []):
            if r["X"] != "" and r["role2"] in ("A", "B"):
                teams_xy.setdefault(r["role2"], []).append((float(r["X"]), float(r["Y"])))
        for team, xy in teams_xy.items():
            xy = np.array(xy)
            if len(xy) >= 3 and np.ptp(xy[:, 0]) > 0.1 and np.ptp(xy[:, 1]) > 0.1:
                hv = ConvexHull(xy).vertices
                cv2.fillPoly(layer, [np.array([P(*xy[i]) for i in hv], np.int32)], colour[team])
            s = shp.get(name, {}).get(team)
            if s:
                line = float(s["line"])
                X = line if dirs[team] > 0 else PT.L - line
                cv2.line(mm, P(X, -PT.W / 2), P(X, PT.W / 2), colour[team], 3)
        mm = cv2.addWeighted(layer, 0.28, mm, 0.72, 0)
        for r in by_frame.get(name, []):
            if r["X"] == "":
                continue
            c = colour.get(r["role2"]) or PT.OFFICIAL.get(r["role2"], (128, 128, 128))
            q = P(float(r["X"]), float(r["Y"]))
            cv2.circle(mm, q, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(mm, q, 5, c, -1, cv2.LINE_AA)
        trail = [ball[n] for n, tt in order if t - 1.0 <= tt <= t and n in ball and ball[n]["X"] != ""]
        for p, q in zip(trail, trail[1:]):
            cv2.line(mm, P(float(p["X"]), float(p["Y"])), P(float(q["X"]), float(q["Y"])), (0, 220, 255), 2, cv2.LINE_AA)
        if b and b["X"] != "":
            q = P(float(b["X"]), float(b["Y"]))
            cv2.circle(mm, q, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(mm, q, 6, (0, 255, 255), -1 if b["kind"] == "det" else 2, cv2.LINE_AA)
        my = MAP_Y + MAP_HEAD
        canvas[my:my + mm.shape[0], map_x:map_x + mm.shape[1]] = mm

        # charts with cursor and the played part bright
        chart_labels = []
        for (cimg, px, py, labels), key, (rx, ry) in zip(charts, ("line", "width"), ((cx, cy), (cx, cy + half_h + 8))):
            ci = cimg.copy()
            for team in "AB":
                tt, vv = series[key][team]
                sel = tt <= t
                if sel.sum() > 1:
                    cv2.polylines(ci, [np.c_[px(tt[sel]), py(vv[sel])].astype(np.int32)], False, colour[team], 3,
                                  cv2.LINE_AA)
            cv2.line(ci, (int(px(t)), 30), (int(px(t)), ci.shape[0] - 24), INK, 1)
            canvas[ry:ry + ci.shape[0], rx:rx + ci.shape[1]] = ci
            chart_labels += [(txt, (rx + x_, ry + y_), col, size, bold) for txt, (x_, y_), col, size, bold in labels]

        # side panel background + team swatches
        sx, sy, sw, sh = SIDE
        cv2.rectangle(canvas, (sx + 12, sy), (sx + sw - 16, sy + sh), PANEL, -1)
        card_h = 228
        for i, team in enumerate(("A", "B")):
            y0 = sy + 14 + i * (card_h + 10)
            cv2.rectangle(canvas, (sx + 28, y0 + 10), (sx + 44, y0 + 34), colour[team], -1)

        # text (PIL)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        rgb = lambda c: tuple(int(v) for v in c[::-1]) if len(c) == 3 else c
        ink = INK
        d.text((20, 12), f"Match analysis  ·  {match} (test match)", font=f_title, fill=ink)
        d.text((WIDTH - 360, 14), f"game clock {game_clock}   ·   video {t:6.1f} s", font=f_med, fill=MUTED)
        for i, team in enumerate(("A", "B")):
            y0 = sy + 14 + i * (card_h + 10)
            s = shp.get(name, {}).get(team)
            recent = Counter(f for tt, f in form_hist[team] if t - FORM_WIN <= tt <= t).most_common(1)
            form = recent[0][0] if recent else "-"
            d.text((sx + 54, y0 + 4), f"{names[team]}", font=f_big, fill=ink)
            d.text((sx + 28, y0 + 50), "Formation (last 10 s)", font=f_small, fill=MUTED)
            d.text((sx + 28, y0 + 68), form, font=f_big, fill=ink)
            d.text((sx + 330, y0 + 50), "Set in game", font=f_small, fill=MUTED)
            d.text((sx + 330, y0 + 68), truth.get(side[team], "?"), font=f_big, fill=ink)
            vals = [("Width", "width", "m"), ("Depth", "depth", "m"), ("Def. line", "line", "m"),
                    ("Compactness", "hull", "m²")]
            for j, (lab, key, unit) in enumerate(vals):
                v = at(*series[key][team], t)
                txt = "-" if v is None else (f"{v:.0f} {unit}")
                xx = sx + 28 + (j % 2) * 302
                yy = y0 + 120 + (j // 2) * 52
                d.text((xx, yy), lab, font=f_small, fill=MUTED)
                d.text((xx, yy + 16), txt, font=f_med, fill=ink)
            n = at(*series["n"][team], t)
            d.text((sx + 28, y0 + card_h - 12), f"based on {0 if n is None else int(n)} visible outfield players",
                   font=f_small, fill=MUTED)
        # legend under the cards
        ly = sy + sh - 30
        items = [(names["A"], colour["A"]), (names["B"], colour["B"]),
                 *((OFFICIAL_NAMES[k], c) for k, c in PT.OFFICIAL.items()), ("Ball", (0, 255, 255))]
        lx = sx + 28
        for lab, c in items:
            d.ellipse((lx, ly + 5, lx + 12, ly + 17), fill=rgb(c), outline=(0, 0, 0))
            d.text((lx + 18, ly), lab, font=f_small, fill=ink)
            lx += 30 + int(d.textlength(lab, font=f_small))
        for txt, (x_, y_), col, size, bold in chart_labels:
            d.text((x_, y_), txt, font=font(size, bold), fill=col)
        d.text((map_x, MAP_Y + 2), "Pitch view  ·  shaded: team shape  ·  line: defensive line  ·  yellow: ball",
               font=f_small, fill=INK)
        canvas = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        vw.write(canvas)
        if k % 200 == 0:
            print(f"  frame {k}/{len(order)}", flush=True)
    vw.release()
    print(f"-> out/{stem}_demo.mp4 ({len(order)} frames, {fps:.1f} fps)")


if __name__ == "__main__":
    main()
