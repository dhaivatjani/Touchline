"""Builds the markdown tables for the benchmark report from results/frames.csv.

Writes results/tables.md. Reuses camera_benchmark.summarise so the numbers match
the CSVs exactly.
"""

from pathlib import Path

import numpy as np

import pandas as pd

from camera_benchmark import OUT_DIR, summarise

COLS = {
    "PVR": "PVR",
    "players_mean": "Players/frame",
    "share_ge18": "≥18 visible",
    "players_p10": "Worst 10% (players)",
    "BVR": "BVR",
    "bbox_h_med_pct": "Player height (% frame)",
    "cam_speed_mean": "Pan speed (frame-w/s)",
    "cam_speed_p95": "Pan p95",
    "cam_zoom_mean": "Zoom rate (log-scale/s)",
    "cam_zoom_p95": "Zoom p95",
    "gameplay_s": "Gameplay s",
    "gameplay_share": "Gameplay share",
}
TRACK = {
    "track_len_med_s": "Median track length (s)",
    "dets_in_stable_tracks": "Detections in tracks ≥2 s",
    "ids_per_player_min": "New IDs per player-minute",
}


def to_md(t: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown table (avoids a tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.3g}"
        return str(v)
    names = [" / ".join(map(str, t.index.names))] if any(t.index.names) else [""]
    head = "| " + " | ".join(names + [str(c) for c in t.columns]) + " |"
    sep = "|" + "---|" * (len(t.columns) + 1)
    rows = ["| " + " | ".join([fmt(i if not isinstance(i, tuple) else " / ".join(map(str, i)))]
                              + [fmt(v) for v in r]) + " |" for i, r in zip(t.index, t.itertuples(index=False))]
    return "\n".join([head, sep, *rows])


def md(df, index, cols):
    if df.empty:
        return "_no data_"
    cols = [c for c in cols if c in df.columns]
    t = df.set_index(index)[cols].rename(columns={**COLS, **TRACK})
    return to_md(t)


# Live-play filter, applied on top of the grass filter in camera_benchmark.py.
# Lineup screens, player close-ups, goal-net replays and celebrations are green
# enough to pass the grass test; they show <4 detected players or close-up-sized
# boxes. These cutscenes are the same whichever camera is selected, so they say
# nothing about the camera and only add noise that depends on match events.
LIVE_MIN_PLAYERS = 4
LIVE_MAX_MEDIAN_H = 0.20
BLOCK_S = 30.0      # block length for the bootstrap (frames within a block are correlated)
N_BOOT = 2000


def live_play(df):
    df = df.copy()
    df["gameplay"] = (df.gameplay & (df.n_players.fillna(0) >= LIVE_MIN_PLAYERS)
                      & (df.bbox_h_med.fillna(0) <= LIVE_MAX_MEDIAN_H))
    return df


def block_bootstrap(df, metric_fn, seed=0):
    """95% CI of a per-frame metric for one recording, resampling 30 s blocks."""
    rng = np.random.default_rng(seed)
    d = df[df.gameplay]
    blocks = [b for _, b in d.groupby((d.t // BLOCK_S).astype(int))]
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        stats.append(metric_fn(pd.concat([blocks[i] for i in pick])))
    return np.percentile(stats, [2.5, 97.5])


def ci_table(primary):
    fns = {
        "PVR": lambda d: (d.n_players.clip(upper=22) / 22).mean(),
        "≥18 visible": lambda d: (d.n_players >= 18).mean(),
        "BVR": lambda d: d.ball.mean(),
        "Zoom rate": lambda d: d.cam_zoom_rate.abs().mean(),
    }
    rows = []
    for angle, d in primary.groupby("angle"):
        row = {"angle": angle}
        for name, fn in fns.items():
            lo, hi = block_bootstrap(d, fn)
            row[name] = f"{fn(d[d.gameplay]):.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return to_md(pd.DataFrame(rows).set_index("angle"))


def sensitivity(primary):
    rows = []
    for nmin, hmax in [(0, 1.0), (4, 0.20), (6, 0.20), (6, 0.15)]:
        d = primary.copy()
        d["gameplay"] = d.gameplay & (d.n_players.fillna(0) >= nmin) & (d.bbox_h_med.fillna(0) <= hmax)
        s = summarise(d, ["angle"]).set_index("angle")
        rank = " > ".join(f"C{a}" for a in s.PVR.sort_values(ascending=False).index)
        rows.append({"filter": f"players ≥ {nmin}, median height ≤ {hmax:.2f}",
                     **{f"C{a} PVR": s.PVR[a] for a in s.index}, "PVR ranking": rank})
    return to_md(pd.DataFrame(rows).set_index("filter"))


def main():
    raw = pd.read_csv(OUT_DIR / "frames.csv", low_memory=False)
    raw["gameplay"] = raw.gameplay.astype(bool)
    raw["ball"] = raw.ball.fillna(False).astype(bool)
    df = live_play(raw)
    full = df[df.situation == "FullMatch"]
    clips = df[df.situation != "FullMatch"]
    primary = full[full["clip"].str.fullmatch(r"Camera \d")]
    primary_raw = raw[(raw.situation == "FullMatch") & raw["clip"].str.fullmatch(r"Camera \d")]

    out = ["# Benchmark tables", "",
           f"Live-play filter: grass test, plus ≥ {LIVE_MIN_PLAYERS} players and median player height "
           f"≤ {LIVE_MAX_MEDIAN_H:.0%} of frame. Applied to every table except 1b.", ""]
    out += ["## 1. One full match per angle (equal basis: `Camera N.mp4`, 10 fps)", "",
            md(summarise(primary, ["angle"]), "angle", list(COLS)), ""]
    out += ["## 1a. 95% block-bootstrap intervals (30 s blocks), table 1", "",
            ci_table(primary), ""]
    out += ["## 1b. Table 1 with the grass filter only (cutscenes included)", "",
            md(summarise(primary_raw, ["angle"]), "angle", list(COLS)), ""]
    out += ["## 1c. Sensitivity of full-match PVR to the live-play filter", "",
            sensitivity(primary_raw), ""]
    out += ["## 2. All full-match recordings per angle", "",
            md(summarise(full, ["angle"]), "angle", ["clips"] + list(COLS)), ""]
    ko = clips[clips.situation == "KickOff"]
    out += ["## 3. KickOff clips (the only situation recorded on all five angles)", "",
            md(summarise(ko, ["angle"]), "angle", list(COLS) + list(TRACK)), ""]
    sit = summarise(clips, ["angle", "situation"])
    for metric in ["PVR", "BVR", "bbox_h_med_pct", "cam_speed_mean"]:
        piv = sit.pivot(index="situation", columns="angle", values=metric)
        piv.columns = [f"C{c}" for c in piv.columns]
        out += [f"## 4. {COLS[metric]} by situation (situation clips, full frame rate)", "",
                to_md(piv), ""]
    out += ["## 5. All situation clips per angle, incl. tracking", "",
            md(summarise(clips, ["angle"]), "angle", ["clips"] + list(COLS) + list(TRACK)), ""]
    out += ["## 6. Per clip", "",
            md(summarise(df, ["angle", "clip", "situation"]), ["angle", "clip"],
               ["situation", "PVR", "share_ge18", "BVR", "bbox_h_med_pct", "cam_speed_mean",
                "cam_zoom_mean", "gameplay_share", "dets_in_stable_tracks"]), ""]
    (OUT_DIR / "tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
