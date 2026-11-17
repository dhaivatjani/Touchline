"""
Turn eda/out/* (from video_eda.py) into the tidy tables the EDA dashboard reads.

Output (eda/tables/):
  videos.csv      one row per match: metadata, timing, audio, visual and object summaries
  samples.csv     one row per 0.5 s sample: visual stats, state, motion, object counts
  fps_hist.csv    instantaneous frame-rate histogram per match
  per_second.csv  frames and Mbit delivered in every wall-clock second
  box_sizes.csv   detected box height histogram (px at 1080p) per class
  heatmap.csv     where detections sit on screen, 48x22 grid per class
  hue.csv         hue histogram of saturated pixels per split
  img_<match>.js  contact sheet and mean/std images as data: URLs for the page
"""

import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

meta = pd.read_csv(OUT / "metadata.csv")
if "android_version" not in meta:          # runs of video_eda.py from before the tag was collected
    import json
    import subprocess
    VIDEO_DIR = HERE.parent / "Screen Recordings" / "Final Recordings"
    tag = lambda f: json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(VIDEO_DIR / f)],
        capture_output=True, text=True, check=True).stdout)["format"].get("tags", {}).get("com.android.version", "")
    meta["android_version"] = [tag(f) for f in meta.file]
timing = pd.read_csv(OUT / "timing.csv")
audio = pd.read_csv(OUT / "audio.csv")
fr = pd.read_csv(OUT / "frames.csv")
det = pd.read_csv(OUT / "detections.csv")
# Same cuts as dataset/sample_frames.py: on-pitch people at conf >= 0.30, on-pitch balls at >= 0.15.
# Off-pitch people (crowd, bench, photographers) are counted separately.
det["keep"] = det.on_pitch & ((det.cls == "ball") | (det.conf >= 0.30))
off_pitch = det[(det.cls == "person") & ~det.on_pitch & (det.conf >= 0.30)]
det = det[det.keep]
det_t = fr[fr.get("det", False) == True][["match_id", "t", "state"]]  # noqa: E712

# ---- object counts per detection sample (zero-filled)
counts = det.assign(person=det.cls.eq("person"), ball=det.cls.eq("ball")) \
    .groupby(["match_id", "t"])[["person", "ball"]].sum().reset_index()
off = off_pitch.groupby(["match_id", "t"]).size().rename("off_pitch").reset_index()
det_t = det_t.merge(counts, on=["match_id", "t"], how="left").merge(off, on=["match_id", "t"], how="left") \
    .fillna({"person": 0, "ball": 0, "off_pitch": 0})
fr = fr.merge(det_t[["match_id", "t", "person", "ball", "off_pitch"]], on=["match_id", "t"], how="left")
fr = fr.rename(columns={"person": "n_person", "ball": "n_ball", "off_pitch": "n_off_pitch"})

# ---- samples
samples = fr[["match_id", "t", "state", "luma", "contrast", "saturation", "colourfulness",
              "sharpness", "entropy", "green_frac", "panel_frac", "mad", "hist_corr",
              "flow_dx", "flow_dy", "flow_mag", "scene_cut", "near_dup", "n_person", "n_ball", "n_off_pitch"]].copy()
samples = samples.merge(meta[["match_id", "split"]], on="match_id")
for c in ["luma", "contrast", "saturation", "colourfulness", "sharpness", "entropy", "mad",
          "flow_dx", "flow_dy", "flow_mag"]:
    samples[c] = samples[c].round(2)
for c in ["green_frac", "panel_frac", "hist_corr"]:
    samples[c] = samples[c].round(3)
samples["scene_cut"] = samples.scene_cut.astype(int)
samples["near_dup"] = samples.near_dup.astype(int)
samples.to_csv(TAB / "samples.csv", index=False)

# ---- per-video summary
g = fr.groupby("match_id")
live = fr[fr.state == "live"]
gl = live.groupby("match_id")
dl = det.merge(det_t[["match_id", "t", "state"]], on=["match_id", "t"])
dl = dl[dl.state == "live"]
persons = dl[dl.cls == "person"]
balls = dl[dl.cls == "ball"]
live_det = det_t[det_t.state == "live"].groupby("match_id")
vis = pd.DataFrame({
    "n_samples": g.size(),
    "share_live": g.state.apply(lambda s: (s == "live").mean()),
    "share_replay": g.state.apply(lambda s: (s == "replay_or_cutscene").mean()),
    "share_other": g.state.apply(lambda s: (s == "other").mean()),
    "share_black": g.state.apply(lambda s: (s == "black").mean()),
    "cuts_per_min": g.scene_cut.sum() / (g.t.max() / 60),
    "share_near_dup_live": gl.near_dup.mean(),
    "luma_live": gl.luma.median(), "contrast_live": gl.contrast.median(),
    "saturation_live": gl.saturation.median(), "sharpness_live": gl.sharpness.median(),
    "sharpness_live_p10": gl.sharpness.quantile(0.1),
    "green_live": gl.green_frac.median(),
    "pan_speed_live": gl.flow_mag.median(), "pan_speed_live_p95": gl.flow_mag.quantile(0.95),
    "persons_per_frame_live": live_det.person.mean(),
    "off_pitch_people_live": live_det.off_pitch.mean(),
    "ball_visible_live": live_det.ball.apply(lambda b: (b > 0).mean()),
    "person_h_px_median": persons.groupby("match_id").h.median() * 1080,
    "ball_px_median": balls.groupby("match_id").apply(lambda b: np.sqrt(b.w * 2400 * b.h * 1080).median()),
})
videos = meta.merge(timing, on="match_id").merge(audio, on="match_id") \
    .merge(vis.reset_index(), on="match_id")
videos["minutes"] = (videos.duration_s / 60).round(2)
videos["live_minutes"] = (videos.minutes * videos.share_live).round(2)
videos["creation_date"] = videos.creation_time.str.slice(0, 10)
num = videos.select_dtypes("number").columns
videos[num] = videos[num].round(3)
videos.drop(columns=["container", "creation_time"]).to_csv(TAB / "videos.csv", index=False)

# ---- frame timing tables
EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 1000]
fh, ps = [], []
for mid in meta.match_id:
    z = np.load(OUT / "packets" / f"{mid}.npz")
    dt = np.diff(z["pts"]); dt = dt[dt > 0]
    inst = 1 / dt
    h, _ = np.histogram(inst, EDGES)
    time_w, _ = np.histogram(inst, EDGES, weights=dt)   # share of wall time spent at that rate
    for lo, hi, n, w in zip(EDGES[:-1], EDGES[1:], h, time_w):
        fh.append(dict(match_id=mid, fps_lo=lo, fps_bin=f"{lo}-{hi}" if hi < 1000 else f"{lo}+",
                       frames=int(n), share_time=round(w / dt.sum(), 4)))
    for s, (n, mb) in enumerate(zip(z["frames_per_s"], z["mbps_per_s"])):
        ps.append(dict(match_id=mid, sec=s, frames=int(n), mbps=round(float(mb), 2)))
pd.DataFrame(fh).to_csv(TAB / "fps_hist.csv", index=False)
pd.DataFrame(ps).to_csv(TAB / "per_second.csv", index=False)

# ---- detections: sizes and positions (live play only)
dl = dl.merge(meta[["match_id", "split"]], on="match_id")
dl["h_px"] = dl.h * 1080
bins = {"person": np.arange(0, 241, 10), "ball": np.arange(0, 41, 2)}
rows = []
for cls, e in bins.items():
    d = dl[dl.cls == cls]
    size = d.h_px if cls == "person" else np.sqrt(d.w * 2400 * d.h * 1080)
    for split in ["train", "val", "test"]:
        h, _ = np.histogram(size[d.split == split].clip(upper=e[-1] - 1e-6), e)
        for lo, n in zip(e[:-1], h):
            rows.append(dict(cls=cls, split=split, px_lo=int(lo), px_hi=int(lo + e[1]), count=int(n)))
pd.DataFrame(rows).to_csv(TAB / "box_sizes.csv", index=False)

GX, GY = 48, 22
hm = []
for cls in ["person", "ball"]:
    d = dl[dl.cls == cls]
    h, _, _ = np.histogram2d(d.cx.clip(0, 0.9999), d.cy.clip(0, 0.9999), [GX, GY], [[0, 1], [0, 1]])
    for x in range(GX):
        for y in range(GY):
            hm.append(dict(cls=cls, gx=x, gy=y, count=int(h[x, y])))
pd.DataFrame(hm).to_csv(TAB / "heatmap.csv", index=False)

# ---- hue
hue = []
for split in ["train", "val", "test"]:
    ms = meta[meta.split == split].match_id
    if ms.empty:
        continue
    acc = sum(np.load(OUT / "hue" / f"{m}.npy") for m in ms)
    acc = acc.reshape(90, 2).sum(1)
    for i, v in enumerate(acc / acc.sum()):
        hue.append(dict(split=split, hue_deg=i * 4, share=round(float(v), 5)))
pd.DataFrame(hue).to_csv(TAB / "hue.csv", index=False)


# ---- images as data: URLs, a few per script so each stays under 256 KB
def url(path, width, q=70):
    im = cv2.imread(str(path))
    im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


img = OUT / "images"
for old in TAB.glob("images_*.js"):
    old.unlink()
for mid in meta.match_id:
    js = (f'window.EDA_IMG=window.EDA_IMG||{{}};window.EDA_IMG["{mid}"]={{'
          f'sheet:"{url(img / f"{mid}_sheet.jpg", 1000, 65)}",'
          f'mean:"{url(img / f"{mid}_mean.png", 480, 80)}",'
          f'std:"{url(img / f"{mid}_std.png", 480, 80)}"}};\n')
    (TAB / f"img_{mid}.js").write_text(js)
    print(f"img_{mid}.js", len(js) // 1024, "KB")
print("tables ->", TAB)
