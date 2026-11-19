#!/usr/bin/env python
"""
MMST-ViT pipeline dashboard.

Shows the data -> pre-training -> fine-tuning -> evaluation pipeline, what data
is available locally, an interactive forward pass through the model, and the
results of training runs found in the output directory.

Usage:
    streamlit run app.py
"""

import glob
import json
import os
import subprocess
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataset.data_wrapper import DataWrapper
from dataset.hrrr_loader import HRRRDataset
from dataset.sentinel_loader import SentinelDataset
from util.misc import default_device

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_FILES = {"train": "data/soybean_train.json", "val": "data/soybean_val.json"}
FOLDER_ALIASES = {"WRF-HRRR": "WRF-HRRR Computed Dataset", "USDA": "USDA Crop Dataset"}

# Categorical slots from the reference palette (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS = {"ready": "#0ca30c", "partial": "#fab219", "missing": "#898781"}

WEATHER_UNITS = {
    'Avg Temperature (K)': 'K', 'Max Temperature (K)': 'K', 'Min Temperature (K)': 'K',
    'Precipitation (kg m**-2)': 'kg m⁻²', 'Relative Humidity (%)': '%',
    'Wind Gust (m s**-1)': 'm s⁻¹', 'Wind Speed (m s**-1)': 'm s⁻¹',
    'Downward Shortwave Radiation Flux (W m**-2)': 'W m⁻²', 'Vapor Pressure Deficit (kPa)': 'kPa',
}

st.set_page_config(page_title="MMST-ViT Pipeline", page_icon="🌱", layout="wide")


# --------------------------------------------------------------------------- helpers

def resolve(root_dir, rel_path):
    """Path of an index entry on disk, accepting both short and HuggingFace folder names."""
    path = os.path.join(root_dir, rel_path)
    if os.path.exists(path):
        return path
    top, _, rest = rel_path.partition("/")
    return os.path.join(root_dir, FOLDER_ALIASES.get(top, top), rest)


@st.cache_data
def load_index():
    rows = []
    for split, path in INDEX_FILES.items():
        with open(os.path.join(ROOT, path)) as f:
            for i, obj in enumerate(json.load(f)):
                rows.append({**obj, "split": split, "idx": i})
    return rows


def entry_files(entry):
    d = entry["data"]
    return {
        "sentinel": d["sentinel"],
        "hrrr": d["HRRR"]["short_term"] + sum(d["HRRR"]["long_term"], []),
        "usda": [d["USDA"]],
    }


def availability(entries, root_dir):
    """Per-modality (present, total) over the unique files referenced by `entries`."""
    files = {"sentinel": set(), "hrrr": set(), "usda": set()}
    for e in entries:
        for k, v in entry_files(e).items():
            files[k].update(v)
    return {k: (sum(os.path.exists(resolve(root_dir, p)) for p in v), len(v)) for k, v in files.items()}


def status_of(present, total):
    if total and present == total:
        return "ready"
    return "partial" if present else "missing"


def label(e):
    return f"{e['county'].title()}, {e['state']} ({e['FIPS']}) · {e['year']} · {e['split']}"


@st.cache_data
def usda_table(root_dir):
    rows = []
    for e in load_index():
        path = resolve(root_dir, e["data"]["USDA"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"state_ansi": str, "county_ansi": str})
        df = df[(df.state_ansi.str.zfill(2) == e["state_ansi"]) & (df.county_ansi.str.zfill(3) == e["county_ansi"])]
        if len(df):
            rows.append({"FIPS": e["FIPS"], "County": f"{e['county'].title()}, {e['state']}", "Year": e["year"],
                         "Split": e["split"],
                         "Yield (bu/acre)": float(df["YIELD, MEASURED IN BU / ACRE"].iloc[0]),
                         "Production (bu)": float(df["PRODUCTION, MEASURED IN BU"].iloc[0])})
    return pd.DataFrame(rows)


@st.cache_data
def hrrr_frame(paths, fips, kind):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[(df["FIPS Code"].astype(str) == fips) & (df["Daily/Monthly"] == kind)]
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def hrrr_tensors(root_dir, split, idx):
    xs, xl, _, _ = HRRRDataset(root_dir, os.path.join(ROOT, INDEX_FILES[split]), num_workers=1)[idx]
    return xs.numpy(), xl.numpy()


@st.cache_data
def sentinel_tensor(root_dir, split, idx):
    x, _, _ = SentinelDataset(root_dir, os.path.join(ROOT, INDEX_FILES[split]))[idx]
    return x.numpy()


@st.cache_resource
def build_model(num_year, checkpoint=None):
    from models_mmst_vit import MMSTViT
    from models_pvt_simclr import PVTSimCLR
    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    # same configuration as main_finetune_mmst_vit.py
    model = MMSTViT(out_dim=2, pvt_backbone=pvt, context_dim=9, dim=512, batch_size=64, num_year=num_year)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return model.eval()


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def find_runs(output_dir):
    runs = {}
    for d in sorted(glob.glob(os.path.join(output_dir, "*"))):
        if os.path.isdir(d):
            runs[os.path.basename(d)] = {
                "dir": d,
                "checkpoints": sorted(glob.glob(os.path.join(d, "checkpoint-*.pth")),
                                      key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0])
                                      if p.rsplit("-", 1)[1].split(".")[0].isdigit() else -1),
                "log": os.path.join(d, "log.txt"),
                "metrics": os.path.join(d, "metrics.txt"),
                "predictions": os.path.join(d, "predictions.csv"),
                "stdout": os.path.join(d, "stdout.log"),
            }
    return runs


def read_jsonl(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


@st.cache_data
def tensorboard_scalars(run_dir, _mtime):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return pd.DataFrame()
    rows = []
    for ev in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
        acc = EventAccumulator(ev)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            rows += [{"tag": tag, "step": s.step, "value": s.value} for s in acc.Scalars(tag)]
    return pd.DataFrame(rows)


def line_chart(df, x, y, color=None, x_title=None, y_title=None, height=260, log_y=False):
    y_scale = alt.Scale(type="log") if log_y else alt.Scale(zero=False)
    enc = dict(x=alt.X(x, title=x_title or x), y=alt.Y(y, title=y_title or y, scale=y_scale),
               tooltip=[x, alt.Tooltip(y, format=".4g")] + ([color] if color else []))
    if color:
        enc["color"] = alt.Color(color, scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="top", title=None))
    else:
        enc["color"] = alt.value(SERIES[0])
    return alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)) \
        .encode(**enc).properties(height=height).interactive(bind_y=False)


def process_running(pid_file):
    if not os.path.exists(pid_file):
        return False
    try:
        os.kill(int(open(pid_file).read()), 0)
        return True
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    root_dir = st.text_input("Dataset root (--root_dir)", os.path.join(ROOT, "data", "TinyCropNet"))
    output_dir = st.text_input("Output directory", os.path.join(ROOT, "output_dir"))
    st.caption("Fetch missing data with `python download_data.py` "
               "(add `--modalities sentinel` for the ~1 GB of imagery).")
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

index = load_index()
num_year = len(index[0]["data"]["HRRR"]["long_term"])
avail = availability(index, root_dir)
runs = find_runs(output_dir)
pre_ckpts = runs.get("pvt_simclr", {}).get("checkpoints", [])
ft_ckpts = runs.get("mmst_vit", {}).get("checkpoints", [])
has_metrics = os.path.exists(runs.get("mmst_vit", {}).get("metrics", "/nonexistent"))

st.title("MMST-ViT · Soybean Yield Forecasting")
st.caption(f"{len({e['FIPS'] for e in index})} counties · train {min(e['year'] for e in index if e['split']=='train')} "
           f"· validate {max(e['year'] for e in index if e['split']=='val')} · "
           "Sentinel-2 imagery + HRRR weather → USDA yield")

tab_pipe, tab_data, tab_model, tab_results = st.tabs(["Pipeline", "Data", "Model", "Results"])

# --------------------------------------------------------------------------- pipeline

with tab_pipe:
    stages = {
        "sentinel": status_of(*avail["sentinel"]),
        "hrrr": status_of(*avail["hrrr"]),
        "usda": status_of(*avail["usda"]),
        "pretrain": "ready" if pre_ckpts else "missing",
        "finetune": "ready" if ft_ckpts else "missing",
        "eval": "ready" if has_metrics else "missing",
    }

    cols = st.columns(6)
    tiles = [
        ("Sentinel-2 files", f"{avail['sentinel'][0]} / {avail['sentinel'][1]}", stages["sentinel"]),
        ("HRRR files", f"{avail['hrrr'][0]} / {avail['hrrr'][1]}", stages["hrrr"]),
        ("USDA files", f"{avail['usda'][0]} / {avail['usda'][1]}", stages["usda"]),
        ("Pre-train ckpts", str(len(pre_ckpts)), stages["pretrain"]),
        ("Fine-tune ckpts", str(len(ft_ckpts)), stages["finetune"]),
        ("Eval metrics", "logged" if has_metrics else "none", stages["eval"]),
    ]
    icons = {"ready": "✅", "partial": "⚠️", "missing": "⭕"}
    for col, (name, value, s) in zip(cols, tiles):
        col.metric(name, value)
        col.caption(f"{icons[s]} {s}")

    def node(key, title, sub):
        s = stages.get(key)
        fill = {"ready": "#d9f2d9", "partial": "#fdecc4", "missing": "#ecebe7"}.get(s, "#ffffff")
        tag = f"<BR/><FONT POINT-SIZE='9'><I>{s}</I></FONT>" if s else ""
        return (f'{key} [label=<<B>{title}</B><BR/><FONT POINT-SIZE="10">{sub}</FONT>{tag}> '
                f'fillcolor="{fill}" color="{STATUS.get(s, "#c3c2b7")}"];')

    dot = "\n".join([
        "digraph G { rankdir=LR; bgcolor=transparent; nodesep=0.25; ranksep=0.45;",
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontcolor="#0b0b0b" penwidth=1.5 margin="0.15,0.08"];',
        'edge [color="#898781" arrowsize=0.7];',
        node("sentinel", "Sentinel-2 imagery", "HDF5 · 6×G×384×384×3"),
        node("hrrr", "HRRR weather", f"short-term 6×G×28×9<BR/>long-term {num_year}×12×9"),
        node("usda", "USDA statistics", "log production, log yield"),
        node("pretrain", "PVT-SimCLR pre-training", "main_pretrain_mmst_vit.py<BR/>NT-Xent contrastive loss"),
        node("finetune", "MMST-ViT fine-tuning", "main_finetune_mmst_vit.py<BR/>spatial + temporal transformer, MSE"),
        node("eval", "Evaluation", "RMSE · R² · Pearson r<BR/>on validation year"),
        "sentinel -> pretrain; hrrr -> pretrain; pretrain -> finetune [label=<<FONT POINT-SIZE='9'>backbone</FONT>> fontcolor=\"#898781\"];",
        "sentinel -> finetune; hrrr -> finetune; usda -> finetune [label=<<FONT POINT-SIZE='9'>target</FONT>> fontcolor=\"#898781\"]; finetune -> eval;",
        "}",
    ])
    st.graphviz_chart(dot, width="stretch")

    st.subheader("Run")
    if stages["sentinel"] != "ready":
        st.info(f"Training needs the Sentinel-2 imagery: {avail['sentinel'][0]} of {avail['sentinel'][1]} "
                "files are present. Run `python download_data.py --modalities sentinel` (~59 GB) "
                "or point the sidebar at an existing Tiny CropNet copy.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Model tests** · random tensors, no data needed")
        if st.button("Run tests/test_models.py", width="stretch"):
            with st.spinner("Running forward passes…"):
                proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_models.py"], cwd=ROOT,
                                      capture_output=True, text=True)
            (st.success if proc.returncode == 0 else st.error)(
                "Passed" if proc.returncode == 0 else f"Failed (exit {proc.returncode})")
            st.code((proc.stdout + proc.stderr)[-4000:], language="text")

    device_options = [default_device()] + [d for d in ("cpu", "mps", "cuda") if d != default_device()
                                           and (d == "cpu" or (d == "mps" and torch.backends.mps.is_available())
                                                or (d == "cuda" and torch.cuda.is_available()))]

    def launch(stage, script, extra):
        run_dir = os.path.join(output_dir, stage)
        os.makedirs(run_dir, exist_ok=True)
        log = open(os.path.join(run_dir, "stdout.log"), "w")
        cmd = [sys.executable, script, "--root_dir", root_dir, "--output_dir", run_dir, "--log_dir", run_dir,
               "--num_workers", "0", "--no_pin_mem"] + extra
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})  # live output on the Results tab
        with open(os.path.join(run_dir, "pid"), "w") as f:
            f.write(str(proc.pid))
        st.success(f"Started {script} (pid {proc.pid}); follow it on the Results tab.")

    for col, stage, script, title in [(c2, "pvt_simclr", "main_pretrain_mmst_vit.py", "Pre-train backbone"),
                                      (c3, "mmst_vit", "main_finetune_mmst_vit.py", "Fine-tune MMST-ViT")]:
        with col:
            st.markdown(f"**{title}** · `{script}`")
            running = process_running(os.path.join(output_dir, stage, "pid"))
            dev = st.selectbox("Device", device_options, key=f"dev_{stage}")
            epochs = st.number_input("Epochs", 1, 500, 10, key=f"ep_{stage}")
            extra = ["--device", dev, "--epochs", str(epochs), "--warmup_epochs", str(max(1, epochs // 5))]
            if stage == "mmst_vit":
                # checkpointed 16-tile chunks keep a 186-tile county at ~4.5 GB; same gradients.
                # A step is one county either way, so the LR is set explicitly rather than scaled by chunk size.
                extra += ["--grad_checkpointing", "--batch_size", "16", "--lr", "2.5e-4"]
                if pre_ckpts:
                    extra += ["--pvt_simclr", pre_ckpts[-1]]
            if running:
                st.warning("Running…")
            elif st.button(f"Start {title.lower()}", key=f"go_{stage}", width="stretch",
                           disabled=stages["sentinel"] != "ready"):
                launch(stage, script, extra)

# --------------------------------------------------------------------------- data

with tab_data:
    st.subheader("Data inventory")
    st.caption(f"Index files `{INDEX_FILES['train']}` and `{INDEX_FILES['val']}` · root `{root_dir}`")
    inv = []
    for e in index:
        a = availability([e], root_dir)
        inv.append({"Split": e["split"], "FIPS": e["FIPS"], "County": e["county"].title(), "State": e["state"],
                    "Year": e["year"], **{k.upper() if k != "sentinel" else "Sentinel-2": f"{p}/{t}"
                                          for k, (p, t) in a.items()}})
    st.dataframe(pd.DataFrame(inv), hide_index=True, width="stretch", height=35 * len(inv) + 38)

    st.subheader("USDA soybean yield (target)")
    usda = usda_table(root_dir)
    if usda.empty:
        st.info("USDA files not found. Run `python download_data.py`.")
    else:
        usda["Year"] = usda["Year"].astype(str)
        bars = alt.Chart(usda).mark_bar(cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("County:N", title=None, sort=alt.SortField("County")),
            yOffset=alt.YOffset("Year:N"),
            x=alt.X("Yield (bu/acre):Q"),
            color=alt.Color("Year:N", scale=alt.Scale(range=SERIES[:2]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["County", "Year", "Split", alt.Tooltip("Yield (bu/acre):Q", format=".1f"),
                     alt.Tooltip("Production (bu):Q", format=",.0f")],
        ).properties(height=420)
        st.altair_chart(bars, width="stretch")
        with st.expander("Table"):
            st.dataframe(usda, hide_index=True, width="stretch")

    st.subheader("HRRR weather explorer")
    choice = st.selectbox("County-year", range(len(index)), format_func=lambda i: label(index[i]))
    e = index[choice]
    short_paths = [resolve(root_dir, p) for p in e["data"]["HRRR"]["short_term"]]
    if not all(os.path.exists(p) for p in short_paths):
        st.info("HRRR files for this county-year are not downloaded. Run `python download_data.py`.")
    else:
        var = st.selectbox("Variable", list(WEATHER_UNITS))
        daily = hrrr_frame(tuple(short_paths), e["FIPS"], "Daily")
        daily = daily[daily.Day.isin(range(1, 29))]
        daily["Date"] = pd.to_datetime(dict(year=daily.Year, month=daily.Month, day=daily.Day.astype(int)))
        agg = daily.groupby("Date")[var].agg(["mean", "min", "max"]).reset_index()
        n_grids = daily["Grid Index"].nunique()

        band = alt.Chart(agg).mark_area(opacity=0.18, color=SERIES[0]).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("min:Q", title=f"{var.split(' (')[0]} ({WEATHER_UNITS[var]})",
                                                   scale=alt.Scale(zero=False)), y2="max:Q")
        line = alt.Chart(agg).mark_line(strokeWidth=2, color=SERIES[0]).encode(
            x="Date:T", y="mean:Q",
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("mean:Q", format=".2f", title="grid mean"),
                     alt.Tooltip("min:Q", format=".2f"), alt.Tooltip("max:Q", format=".2f")])
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Growing season (Apr–Sep {e['year']}), days 1–28** · line = mean over {n_grids} grids, "
                        "band = grid min–max")
            st.altair_chart((band + line).properties(height=300).interactive(bind_y=False), width="stretch")
        with right:
            grids = daily.groupby("Grid Index")[["Lat (llcrnr)", "Lat (urcrnr)", "Lon (llcrnr)", "Lon (urcrnr)"]].first()
            pts = pd.DataFrame({"lat": grids[["Lat (llcrnr)", "Lat (urcrnr)"]].mean(axis=1),
                                "lon": grids[["Lon (llcrnr)", "Lon (urcrnr)"]].mean(axis=1)})
            st.markdown(f"**{n_grids} HRRR grid cells in {e['county'].title()} County**")
            st.map(pts, size=1200, color=SERIES[1], height=300)

        long_paths = [resolve(root_dir, p) for y in e["data"]["HRRR"]["long_term"] for p in y]
        if all(os.path.exists(p) for p in long_paths):
            monthly = hrrr_frame(tuple(long_paths), e["FIPS"], "Monthly")
            monthly = monthly.groupby(["Year", "Month"])[var].mean().reset_index()
            monthly["Year"] = monthly["Year"].astype(str)
            st.markdown(f"**Long-term context: monthly {var.split(' (')[0].lower()}, "
                        f"{monthly.Year.min()}–{monthly.Year.max()}** ({num_year} years before {e['year']} feed the model's `yl` input)")
            heat = alt.Chart(monthly).mark_rect(cornerRadius=2, stroke="white", strokeWidth=2).encode(
                x=alt.X("Month:O"), y=alt.Y("Year:O", title=None),
                color=alt.Color(f"{var}:Q", title=WEATHER_UNITS[var],
                                scale=alt.Scale(range=["#cde2fb", "#2a78d6", "#0d366b"])),
                tooltip=["Year", "Month", alt.Tooltip(f"{var}:Q", format=".2f")]).properties(height=180)
            st.altair_chart(heat, width="stretch")

        xs, xl = hrrr_tensors(root_dir, e["split"], e["idx"])
        st.caption(f"`HRRRDataset` output for this sample → short-term `ys` {tuple(xs.shape)} "
                   f"(months, grids, days, variables), long-term `yl` {tuple(xl.shape)} (years, months, variables)")

    st.subheader("Sentinel-2 imagery")
    sent_paths = [resolve(root_dir, p) for p in e["data"]["sentinel"]]
    if not all(os.path.exists(p) for p in sent_paths):
        st.info(f"Not downloaded for {label(e)}. Each state-quarter file is ~3.5–4 GB; "
                "`python download_data.py --modalities sentinel` fetches the 16 files the index needs (~59 GB).")
    else:
        import h5py
        with h5py.File(sent_paths[0], "r") as hf:
            dates = list(hf[e["FIPS"]].keys())
            date = st.select_slider("Date", dates)
            imgs = np.asarray(hf[e["FIPS"]][date]["data"])
        st.caption(f"{imgs.shape[0]} grid cells · each {imgs.shape[1]}×{imgs.shape[2]} px (9×9 km) · showing up to 12")
        st.image([im.astype(np.uint8) for im in imgs[:12]], width=150)

# --------------------------------------------------------------------------- model

with tab_model:
    ckpt = st.selectbox("Weights", ["random init (untrained)"] + ft_ckpts,
                        format_func=lambda p: os.path.relpath(p, output_dir) if p.endswith(".pth") else p)
    model = build_model(num_year, None if ckpt.startswith("random") else ckpt)

    parts = [("PVT-SimCLR backbone", model.pvt_backbone), ("Spatial transformer", model.space_transformer),
             ("Temporal transformer", model.temporal_transformer), ("Long-term context projection", model.proj_context),
             ("Prediction head", model.mlp_head)]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Architecture")
        st.metric("Total parameters", f"{n_params(model) / 1e6:.1f} M")
        st.dataframe(pd.DataFrame([{"Component": n, "Parameters": f"{n_params(m):,}"} for n, m in parts]),
                     hide_index=True, width="stretch")
        st.caption("Configured as in `main_finetune_mmst_vit.py`: pvt_tiny backbone, dim 512, "
                   f"4 layers × 3 heads per transformer, {num_year} years of climate context, "
                   "2 outputs (log production, log yield).")

    with c2:
        st.subheader("Forward pass")
        e = index[st.selectbox("Sample", range(len(index)), format_func=lambda i: label(index[i]), key="fwd")]
        files = entry_files(e)
        hrrr_ok = all(os.path.exists(resolve(root_dir, p)) for p in files["hrrr"])
        sent_ok = all(os.path.exists(resolve(root_dir, p)) for p in files["sentinel"])
        st.caption(("Real Sentinel-2 imagery" if sent_ok else "Random imagery (Sentinel-2 not downloaded)") + " · " +
                   ("real HRRR weather" if hrrr_ok else "random weather (HRRR not downloaded)"))
        if st.button("Run forward pass", type="primary"):
            if hrrr_ok:
                xs, xl = hrrr_tensors(root_dir, e["split"], e["idx"])
                ys, yl = torch.from_numpy(xs)[None], torch.from_numpy(xl)[None]
            else:
                ys, yl = torch.randn(1, 6, 10, 28, 9), torch.randn(1, 5, 12, 9)
            if sent_ok:
                # same preprocessing as evaluation: center crop to 224 and normalize
                tiles = torch.from_numpy(sentinel_tensor(root_dir, e["split"], e["idx"]))
                t, g = tiles.shape[:2]
                x, _ = DataWrapper(train=False)(tiles.flatten(0, 1).permute(0, 3, 1, 2))
                x = x.reshape(1, t, g, 3, 224, 224)
            else:
                x = torch.randn(1, ys.shape[1], ys.shape[2], 3, 224, 224)
            g = x.shape[2]

            shapes = []
            hooks = [m.register_forward_hook(lambda mod, i, o, n=n: shapes.append((n, tuple(o.shape))))
                     for n, m in parts]
            with st.spinner(f"Encoding {x.shape[1] * g} image tiles…"), torch.no_grad():
                out = model(x, ys=ys, yl=yl)
            for h in hooks:
                h.remove()

            seen, trace = set(), [("Input imagery x", tuple(x.shape)), ("Input weather ys", tuple(ys.shape)),
                                  ("Input climate yl", tuple(yl.shape))]
            for n, s in shapes:
                if n not in seen:  # the backbone runs once per mini-batch; show its first call
                    seen.add(n)
                    trace.append((f"{n} (per {s[0]}-tile batch)" if n == parts[0][0] else n, s))
            trace = [(n, " × ".join(map(str, s))) for n, s in trace]
            st.dataframe(pd.DataFrame(trace, columns=["Stage", "Output shape"]), hide_index=True, width="stretch")

            pred = out.exp()[0].tolist()
            truth = usda_table(root_dir)
            truth = truth[(truth.FIPS == e["FIPS"]) & (truth.Year == e["year"])]
            m1, m2 = st.columns(2)
            m1.metric("Predicted yield", f"{pred[1]:.1f} bu/acre",
                      help="exp of the model's 2nd output (log yield)")
            if len(truth):
                m2.metric("Actual USDA yield", f"{truth['Yield (bu/acre)'].iloc[0]:.1f} bu/acre")
            if ckpt.startswith("random"):
                st.warning("Weights are untrained, so this prediction is meaningless. "
                           "It only checks the data flow and shapes.")

# --------------------------------------------------------------------------- results

with tab_results:
    if not runs:
        st.info(f"No runs in `{output_dir}` yet.")
    else:
        run_name = st.radio("Run", list(runs), horizontal=True,
                            format_func=lambda r: {"pvt_simclr": "PVT-SimCLR pre-training",
                                                   "mmst_vit": "MMST-ViT fine-tuning"}.get(r, r))
        run = runs[run_name]
        log = read_jsonl(run["log"])
        mets = read_jsonl(run["metrics"])
        mtime = max([os.path.getmtime(p) for p in glob.glob(os.path.join(run["dir"], "*"))] or [0])
        tb = tensorboard_scalars(run["dir"], mtime)
        running = process_running(os.path.join(run["dir"], "pid"))

        k = st.columns(4)
        k[0].metric("Status", "running" if running else ("finished" if len(log) else "no training logged"))
        k[1].metric("Epochs logged", len(log))
        k[2].metric("Checkpoints", len(run["checkpoints"]))
        k[3].metric("Last train loss", f"{log['train_loss'].iloc[-1]:.4f}" if "train_loss" in log else "—")

        if not len(mets) and not len(log) and (tb.empty or not len(tb)):
            st.info("This run has no logged metrics yet. The event files here are empty, so the earlier "
                    "runs stopped before their first training step, which fits the missing dataset.")

        if len(mets):
            st.subheader("Validation metrics")
            last = mets.iloc[-1]
            v = st.columns(3)
            v[0].metric("RMSE (bu/acre)", f"{last.rmse:.2f}", help="Lower is better")
            v[1].metric("R²", f"{last.r2:.3f}")
            v[2].metric("Pearson r", f"{last['corr']:.3f}")
            if mets["epoch"].notna().sum() > 1:
                long = mets.rename(columns={"corr": "Pearson r", "r2": "R²"}) \
                    .melt(id_vars="epoch", value_vars=["Pearson r", "R²"], var_name="metric")
                st.altair_chart(line_chart(long, "epoch:Q", "value:Q", "metric:N", "Epoch", "Score"), width="stretch")

        if os.path.exists(run["predictions"]):
            pred = pd.read_csv(run["predictions"], dtype={"fips": str})
            names = {e["FIPS"]: f"{e['county'].title()}, {e['state']}" for e in index}
            pred["County"] = pred.fips.map(names)
            st.subheader("Predicted vs actual yield (validation)")
            st.caption("Dashed line = perfect prediction. Points on a flat horizontal band mean the model "
                       "mostly predicts the average yield.")
            lo, hi = float(pred[["true_yield", "pred_yield"]].min().min()) * 0.95, \
                float(pred[["true_yield", "pred_yield"]].max().max()) * 1.05
            ref = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color="#898781", strokeDash=[4, 4]) \
                .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
                        y=alt.Y("v:Q", scale=alt.Scale(domain=[lo, hi], zero=False)))
            dots = alt.Chart(pred).mark_circle(size=90, color=SERIES[0], stroke="white", strokeWidth=2).encode(
                x=alt.X("true_yield:Q", title="Actual yield (bu/acre)", scale=alt.Scale(domain=[lo, hi], zero=False)),
                y=alt.Y("pred_yield:Q", title="Predicted yield (bu/acre)", scale=alt.Scale(domain=[lo, hi], zero=False)),
                tooltip=["County", alt.Tooltip("true_yield:Q", format=".1f"), alt.Tooltip("pred_yield:Q", format=".1f")])
            st.altair_chart((ref + dots).properties(height=360), width="stretch")
            st.dataframe(pred, hide_index=True, width="stretch")

        if "train_loss" in log:
            st.subheader("Training loss per epoch")
            st.altair_chart(line_chart(log, "epoch:Q", "train_loss:Q", x_title="Epoch", y_title="Train loss (log)",
                                       log_y=True),
                            width="stretch")

        if not tb.empty:
            st.subheader("TensorBoard scalars")
            for tag in sorted(tb.tag.unique()):
                d = tb[tb.tag == tag]
                st.altair_chart(line_chart(d, "step:Q", "value:Q", x_title="Step (epoch × 1000)", y_title=tag,
                                           height=200), width="stretch")

        if os.path.exists(run["stdout"]):
            with st.expander("Process output (stdout.log)", expanded=running):
                with open(run["stdout"]) as f:
                    st.code("".join(f.readlines()[-60:]) or "(empty)", language="text")
