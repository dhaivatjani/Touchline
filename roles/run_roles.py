"""Run the role prototype.

  python roles/run_roles.py seed            team colours for every match with kickoff frames
  python roles/run_roles.py clip M04 M11    seed + track + role video for the val/test clips
  python roles/run_roles.py live M11 60 10  same on 10 s of the original video from 60 s

Outputs in roles/out/: seeds.csv, <match>_clip.mp4, <match>_clip.csv
(frame, track, box, raw role, final role = majority vote per track).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import roles as R
import autolabel as A   # roles puts homography/ on sys.path

ROOT = R.ROOT
OUT = Path(__file__).resolve().parent / "out"
KICK = ROOT / "dataset" / "kickoff_frames"
SAMPLED = ROOT / "kitman" / "data" / "sampled"
WEIGHTS = ROOT / "kitman" / "data" / "runs" / "run5" / "train" / "weights" / "best.pt"
PERSON, BALL, CONF, BALL_CONF, IMGSZ = 1, 0, 0.4, 0.1, 2400
VOTE_WIN = 15        # frames each side (~0.4 s at the clips' ~37 fps)
OTHER_BGR = (255, 0, 255)

_model = None


def model():
    global _model
    _model = _model or YOLO(str(WEIGHTS))
    return _model


def persons(res):
    b = res.boxes
    keep = (b.cls.cpu().numpy() == PERSON) & (b.conf.cpu().numpy() >= CONF)
    ids = b.id.cpu().numpy().astype(int)[keep] if b.id is not None else np.full(keep.sum(), -1)
    return b.xyxy.cpu().numpy()[keep], ids


def new_tracker():
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


def track_people(tracker, res, img):
    """Every person detection (conf >= CONF) with its ByteTrack id, or -1 when the tracker has not
    confirmed it. model.track() returns confirmed tracks only, so after a fast pan (boxes jump, no
    overlap to match) it drops players until they are confirmed again."""
    b = res.boxes.cpu()
    keep = ((b.cls.numpy() == PERSON) & (b.conf.numpy() >= CONF))
    det = b[keep]
    ids = np.full(len(det), -1)
    if len(det):
        for x in tracker.update(det, img):
            ids[int(x[7])] = int(x[4])
    return det.xyxy.numpy(), ids


def balls(res):
    """All ball candidates (conf >= BALL_CONF) -> [(conf, x1, y1, x2, y2)]; the ball tracker picks one."""
    b = res.boxes
    keep = b.cls.cpu().numpy() == BALL
    return [(float(c), *map(float, xy)) for c, xy in zip(b.conf.cpu().numpy()[keep], b.xyxy.cpu().numpy()[keep])]


def seed_match(match):
    """-> (centres (2,3), thr, info) or None. Seeds each kickoff frame on its own (its H may be
    mirrored relative to the other frame), then aligns frames by colour before averaging."""
    per = []
    for f in sorted(KICK.glob(f"{match}_kickoff_*.jpg")):
        img = cv2.imread(str(f))
        r = A.label_frame(img)
        if r["H"] is None or r["status"] != "ok":
            print(f"  {f.name}: homography {r['status']}, skipped")
            continue
        boxes, _ = persons(model().predict(img, imgsz=IMGSZ, conf=CONF, quantize=16, verbose=False)[0])
        L = R.lab(img)
        cols = [R.kit_colour(L, b) for b in boxes]
        ok = [i for i, c in enumerate(cols) if c is not None]
        if len(ok) < 6:
            print(f"  {f.name}: only {len(ok)} people, skipped")
            continue
        feet = np.c_[(boxes[ok, 0] + boxes[ok, 2]) / 2, boxes[ok, 3]]
        sides = R.side_of_halfway(r["H"], feet)
        if sides.all() or not sides.any():
            print(f"  {f.name}: everyone on one side, skipped")
            continue
        c, thr = R.seed(np.array([cols[i] for i in ok]), sides)
        per.append((c, thr, len(ok)))
    if not per:
        return None
    c0 = per[0][0]
    cs = [c if np.linalg.norm(c - c0, axis=1).sum() <= np.linalg.norm(c[::-1] - c0, axis=1).sum() else c[::-1]
          for c, _, _ in per]
    centres = np.mean(cs, axis=0)
    thr = float(np.mean([t for _, t, _ in per]))
    return centres, thr, dict(frames=len(per), people=sum(n for *_, n in per),
                              team_dist=float(np.linalg.norm(centres[0] - centres[1])))


def cmd_seed(only=()):
    OUT.mkdir(exist_ok=True)
    matches = list(only) or sorted({f.name.split("_")[0] for f in KICK.glob("*_kickoff_*.jpg")})
    rows = []
    for m in matches:
        s = seed_match(m)
        if s is None:
            print(f"{m}: no usable kickoff frame")
            rows.append(dict(match=m, frames=0))
            continue
        c, thr, info = s
        print(f"{m}: A Lab {np.round(c[0], 1)}  B Lab {np.round(c[1], 1)}  "
              f"dist {info['team_dist']:.1f}  thr {thr:.1f}  ({info['frames']} frames, {info['people']} people)")
        rows.append(dict(match=m, **info, thr=round(thr, 1),
                         A=" ".join(f"{v:.1f}" for v in c[0]), B=" ".join(f"{v:.1f}" for v in c[1])))
    with open(OUT / "seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["match", "frames", "people", "team_dist", "thr", "A", "B"])
        w.writeheader()
        w.writerows(rows)


def sampled_clip(match):
    """Kitman's val/test tracking clip: [(name, t)] and a generator of (name, t, img)."""
    with open(SAMPLED / match / "frames.csv", newline="") as f:
        clip = sorted(((r["name"], float(r["t"])) for r in csv.DictReader(f) if r["bucket"] == "clip"),
                      key=lambda x: x[1])
    def frames():
        for name, t in clip:
            yield name, t, cv2.imread(str(SAMPLED / match / "images" / name))
    return clip, frames


def video_clip(match, start, dur):
    """Every decoded frame of the original video from start for dur seconds (second pass re-decodes)."""
    with open(ROOT / "dataset" / "splits.csv", newline="") as f:
        file = next(r["file"] for r in csv.DictReader(f) if r["match_id"] == match)
    video = ROOT / "Screen Recordings" / "Final Recordings" / file
    def frames():
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        while True:
            ok, img = cap.read()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or t > start + dur:
                break
            if img.shape[:2] != (1080, 2400):
                img = cv2.resize(img, (2400, 1080))
            yield f"{match}_t{t:08.3f}", t, img
        cap.release()
    return [(n, t) for n, t, _ in frames()], frames


def cmd_clip(match, start=None, dur=10.0):
    s = seed_match(match)
    if s is None:
        print(f"{match}: no seed, skipped")
        return
    centres, thr, info = s
    print(f"{match}: seeded, team dist {info['team_dist']:.1f}, thr {thr:.1f}")
    clip, frames = sampled_clip(match) if start is None else video_clip(match, start, dur)
    stem = f"{match}_clip" if start is None else f"{match}_live{start:g}"
    m = model()
    tracker = new_tracker()   # fresh state per clip
    rows, votes, ball_rows = [], defaultdict(Counter), []
    for name, t, img in frames():
        res = m.predict(img, imgsz=IMGSZ, conf=BALL_CONF, quantize=16, verbose=False)[0]
        boxes, ids = track_people(tracker, res, img)
        ball_rows += [dict(frame=name, t=t, conf=c, x1=x1, y1=y1, x2=x2, y2=y2) for c, x1, y1, x2, y2 in balls(res)]
        L = R.lab(img)
        for b, tid in zip(boxes, ids):
            role = R.assign(R.kit_colour(L, b), centres, thr)
            if role and tid >= 0:
                votes[tid][role] += 1
            rows.append(dict(frame=name, t=t, track=tid, x1=b[0], y1=b[1], x2=b[2], y2=b[3], raw=role))
    # vote in a sliding window, not over the whole track: ByteTrack swaps ids when players cross,
    # and a whole-track vote then carries the old player's team onto the new one
    order = {n: i for i, (n, _) in enumerate(clip)}
    by_track = defaultdict(list)
    for row in rows:
        if row["track"] >= 0:
            by_track[row["track"]].append(row)
    for trs in by_track.values():
        idx = np.array([order[r["frame"]] for r in trs])
        for r, i in zip(trs, idx):
            v = Counter(o["raw"] for o, j in zip(trs, idx) if abs(j - i) <= VOTE_WIN and o["raw"])
            r["role"] = v.most_common(1)[0][0] if v else r["raw"]
    for row in rows:
        row.setdefault("role", row["raw"])
    final = {tid: Counter(r["role"] for r in trs).most_common(1)[0][0] for tid, trs in by_track.items()}
    with open(OUT / f"{stem}_ball.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["frame", "t", "conf", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(ball_rows)
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    span = clip[-1][1] - clip[0][1]
    fps = max(1.0, (len(clip) - 1) / span) if span > 0 else 25.0
    colour = {"A": R.lab_to_bgr(centres[0]), "B": R.lab_to_bgr(centres[1]), "O": OTHER_BGR}
    (OUT / f"{stem}_teams.json").write_text(json.dumps(colour))   # BGR per role, for the minimap
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)
    vw = cv2.VideoWriter(str(OUT / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1200, 540))
    for name, _, img in frames():
        for row in by_frame[name]:
            p1, p2 = (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"]))
            c = colour.get(row["role"], (128, 128, 128))
            cv2.rectangle(img, p1, p2, (0, 0, 0), 6)
            cv2.rectangle(img, p1, p2, c, 3)
            cv2.putText(img, f"{row['role'] or '?'}{row['track']}", (p1[0], p1[1] - 8), 0, 1.0, (0, 0, 0), 5)
            cv2.putText(img, f"{row['role'] or '?'}{row['track']}", (p1[0], p1[1] - 8), 0, 1.0, c, 2)
        for k, (name, c) in enumerate(colour.items()):
            cv2.rectangle(img, (20, 1000 - 50 * k), (60, 1035 - 50 * k), c, -1)
            cv2.putText(img, {"A": "team A", "B": "team B", "O": "other"}[name], (70, 1030 - 50 * k), 0, 1.1,
                        (255, 255, 255), 2)
        vw.write(cv2.resize(img, (1200, 540), interpolation=cv2.INTER_AREA))
    vw.release()
    n = Counter(final.values())
    flips = sum(1 for v in votes.values() if len(v) > 1)
    print(f"{match}: {len(clip)} frames, {len(final)} tracks -> A {n['A']}, B {n['B']}, other {n['O']}; "
          f"{flips} tracks had mixed raw votes -> out/{stem}.mp4")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "seed":
        cmd_seed(sys.argv[2:])
    elif sys.argv[1] == "live":          # live M11 60 [10]
        cmd_clip(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 10.0)
    else:
        for m in sys.argv[2:]:
            cmd_clip(m)
