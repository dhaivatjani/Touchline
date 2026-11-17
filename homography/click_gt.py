"""Click pitch landmarks on the test-match frames -> homography/ground_truth.csv.

Run in your own terminal:  python click_gt.py
For each frame the panel on the right names one landmark and marks it on a small pitch
diagram. Click it in the image if you can see it, or press n to skip it.

  click  record the landmark, go to the next one
  n      not visible, skip this landmark
  h      jump to the halfway-line landmarks
  b      go back one landmark (removes its click if it had one)
  s      save this frame, go to the next frame
  x      frame is not live play / unusable: mark it, go to the next frame
  q      save this frame and quit (rerun later to resume)

"far" = the side of the pitch away from the camera (top of the screen). The goal end
can be on either side of the screen; the diagram always draws it on the left.
Precision: use the magnifier (bottom right). Click the centre of the line.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES = sorted(p for p in (HERE / "check_frames").glob("*.jpg")
                if p.stem[:3] in ("M04", "M11", "M12"))       # test split only
OUT = HERE / "ground_truth.csv"
SCALE = 0.625               # frame -> screen; main() resets it to fit the actual screen
PANEL = 360

R, D = 9.15, (9.15 ** 2 - 5.5 ** 2) ** 0.5
# name -> model (x from the goal line towards halfway, y from goal centre towards near touchline)
LANDMARKS = [
    ("far corner", (0, -34)),
    ("far penalty box: on goal line", (0, -20.16)),
    ("far penalty box: front corner", (16.5, -20.16)),
    ("far 6-yard box: on goal line", (0, -9.16)),
    ("far 6-yard box: front corner", (5.5, -9.16)),
    ("near 6-yard box: front corner", (5.5, 9.16)),
    ("near 6-yard box: on goal line", (0, 9.16)),
    ("near penalty box: front corner", (16.5, 20.16)),
    ("near penalty box: on goal line", (0, 20.16)),
    ("D arc meets box edge: far", (16.5, -D)),
    ("D arc meets box edge: near", (16.5, D)),
    ("near corner", (0, 34)),
    ("halfway meets far touchline", (52.5, -34)),
    ("centre circle meets halfway: far", (52.5, -R)),
    ("centre spot", (52.5, 0)),
    ("centre circle meets halfway: near", (52.5, R)),
    ("halfway meets near touchline", (52.5, 34)),
]
HALFWAY_START = 12


def diagram(idx):
    """Half pitch, goal on the left, current landmark in red."""
    w, h, m = PANEL - 40, 170, 10      # + 2m margin = PANEL - 20, placed at x=10
    img = np.full((h + 2 * m, w + 2 * m, 3), 40, np.uint8)
    sx, sy = w / 60, h / 68

    def p(x, y):
        return int(m + x * sx), int(m + (y + 34) * sy)

    col = (200, 200, 200)
    cv2.rectangle(img, p(0, -34), p(52.5, 34), col, 1)
    cv2.rectangle(img, p(0, -20.16), p(16.5, 20.16), col, 1)
    cv2.rectangle(img, p(0, -9.16), p(5.5, 9.16), col, 1)
    cv2.ellipse(img, p(52.5, 0), (int(R * sx), int(R * sy)), 0, 90, 270, col, 1)
    cv2.ellipse(img, p(11, 0), (int(R * sx), int(R * sy)), 0, -53, 53, col, 1)
    cv2.putText(img, "far (top of screen)", (m + 5, m + 14), 0, 0.4, col, 1)
    cv2.putText(img, "goal", (m + 2, h // 2 + m - 4), 0, 0.35, col, 1)
    cv2.putText(img, "halfway", (int(m + 44 * sx), h // 2 + m - 4), 0, 0.35, col, 1)
    for i, (_, (x, y)) in enumerate(LANDMARKS):
        cv2.circle(img, p(x, y), 6 if i == idx else 3,
                   (0, 0, 255) if i == idx else (120, 120, 120), -1)
    return img


def loupe(frame, mx, my, size=30, zoom=4):
    """Magnified crop around the mouse (full-res coords) with a crosshair."""
    h, w = frame.shape[:2]
    pad = cv2.copyMakeBorder(frame, size, size, size, size, cv2.BORDER_CONSTANT)
    crop = pad[my:my + 2 * size, mx:mx + 2 * size]
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    c = size * zoom
    cv2.line(crop, (c, 0), (c, 2 * c), (0, 0, 255), 1)
    cv2.line(crop, (0, c), (2 * c, c), (0, 0, 255), 1)
    return crop


def done_frames():
    if not OUT.exists():
        return set()
    with open(OUT, newline="") as f:
        return {r["frame"] for r in csv.DictReader(f)}


def save(frame_name, clicks, skipped_frame=False):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["frame", "landmark", "x_px", "y_px"])
        if skipped_frame:
            w.writerow([frame_name, "UNUSABLE", "", ""])
        elif not clicks:
            w.writerow([frame_name, "NONE", "", ""])        # looked at, nothing clickable
        for i, (x, y) in clicks.items():
            w.writerow([frame_name, LANDMARKS[i][0], f"{x:.1f}", f"{y:.1f}"])


def fit_scale():
    """Scale so frame + panel fit the screen, in real pixels (no Windows DPI stretching)."""
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()   # else 125-150% scaling pushes the panel off-screen
        sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return SCALE
    return min((0.9 * sw - PANEL) / 2400, (0.85 * sh - 60) / 1080, 1.0)  # margin: window chrome, taskbar


def main():
    global SCALE
    SCALE = fit_scale()
    print(f"display scale {SCALE:.2f}")
    todo = [p for p in FRAMES if p.stem not in done_frames()]
    print(f"{len(todo)} of {len(FRAMES)} test frames left")
    state = {"mouse": (0, 0), "click": None}

    def on_mouse(ev, x, y, *_):
        if x < int(2400 * SCALE):
            state["mouse"] = (int(x / SCALE), int(y / SCALE))
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x / SCALE, y / SCALE)

    cv2.namedWindow("ground truth", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ground truth", on_mouse)
    for k, path in enumerate(todo):
        frame = cv2.imread(str(path))
        clicks, order, idx = {}, [], 0
        while True:
            if state["click"] is not None and idx < len(LANDMARKS):
                clicks[idx] = state["click"]
                order.append(idx)
                idx += 1
            state["click"] = None
            view = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            for i, (x, y) in clicks.items():
                c = (int(x * SCALE), int(y * SCALE))
                cv2.drawMarker(view, c, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.putText(view, str(i + 1), (c[0] + 6, c[1] - 6), 0, 0.5, (0, 255, 255), 1)
            panel = np.full((view.shape[0], PANEL, 3), 25, np.uint8)
            name = LANDMARKS[idx][0] if idx < len(LANDMARKS) else "all done: press s"
            lines = [f"frame {k + 1}/{len(todo)}: {path.stem}", "",
                     f"{idx + 1}. {name}" if idx < len(LANDMARKS) else name, "",
                     "click = mark   n = not visible", "h = jump to halfway   b = back one",
                     "s = next frame   x = unusable frame", "q = save + quit"]
            for j, t in enumerate(lines):
                cv2.putText(panel, t, (10, 22 + 20 * j), 0, 0.5,
                            (0, 255, 255) if j == 2 else (220, 220, 220), 1)
            dg = diagram(idx)
            panel[180:180 + dg.shape[0], 10:10 + dg.shape[1]] = dg
            lp = loupe(frame, *state["mouse"])
            y0 = view.shape[0] - lp.shape[0] - 10
            if y0 > 180 + dg.shape[0]:
                panel[y0:y0 + lp.shape[0], 10:10 + lp.shape[1]] = lp[:, :PANEL - 20]
            cv2.imshow("ground truth", np.hstack([view, panel]))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n") and idx < len(LANDMARKS):
                idx += 1
            elif key == ord("h"):
                idx = max(idx, HALFWAY_START)
            elif key == ord("b") and idx > 0:             # back one landmark, clicked or skipped
                idx -= 1
                clicks.pop(idx, None)
                if idx in order:
                    order.remove(idx)
            elif key == ord("s"):
                save(path.stem, clicks)
                break
            elif key == ord("x"):
                save(path.stem, {}, skipped_frame=True)
                break
            elif key == ord("q"):
                if clicks:                                 # an untouched frame stays to-do
                    save(path.stem, clicks)
                cv2.destroyAllWindows()
                print(f"saved to {OUT}; rerun to resume")
                return
    cv2.destroyAllWindows()
    print(f"all test frames done; saved to {OUT}")


if __name__ == "__main__":
    main()
