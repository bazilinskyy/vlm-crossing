"""How well the generated videos keep the look of their city, measured with CLIP, and a map of the cities.

Every generated video of videos/generated_videos.csv is sampled at 0, 1, ..., 7 s and at its last frame, and each
frame is embedded with CLIP ViT-B/32 (LAION-2B). The first frame is the CROWD reference frame as rendered by the
model, so it stands for the real scene, and the later frames show how the model continued it. For each frame:

- similarity to the first frame (cosine of the CLIP embeddings): how far the scene drifts from the reference;
- rank of the true city among the cities of CROWD's speed table, for zero-shot prompts naming each city
  ("a photo taken from a car driving through a street in {city}, {country}"): whether the city is still recognisable;
- rank of the true country among the countries of those cities;
- mean absolute pixel difference to the previous sampled frame (0-255): motion, so that frozen clips stand out.

Embeddings are cached per video in <output>/appearance/, so a rerun only embeds new videos. Writes
<output>/appearance_frames.csv (one row per frame), <output>/appearance_summary.csv (one row per model) and
<output>/appearance_cities.csv (one row per model and city), and figures to <output>/figures/. The per-frame table is
also kept in results/appearance/frames.csv, so that without the video files (or with --from-results) the summary and
the figures are rebuilt from it.

    python appearance_analysis.py --readme

Embedding needs torch, open_clip_torch and opencv besides the packages of this repository.
"""

import argparse
import math
import os

import cv2
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import common
import readme_figures
from utils.plotting.crossings import Crossings, io

MODELS = {"veo3": ("Veo 3.1 Fast", "#eb6834"), "minimax_h3": ("MiniMax H3", "#1baf7a")}
SECONDS = list(range(8))  # sampled seconds; the last frame is added as second 8
CLIP_MODEL, CLIP_WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"
# Names CLIP knows better than CROWD's local spelling.
ENGLISH = {"København": "Copenhagen", "München": "Munich", "Den Haag": "The Hague"}
PROMPT = "a photo taken from a car driving through a street in {place}"


def sample_frames(path: str) -> list:
    """RGB frames at each of SECONDS and the last frame (None when the video is shorter)."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = [min(round(s * fps), n - 1) for s in SECONDS] + [n - 1]
    frames, index = {}, 0
    while index <= wanted[-1]:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    last = max(frames) if frames else None
    return [frames.get(w, frames.get(last) if w == wanted[-1] else None) for w in wanted]


def embed_videos(videos: pl.DataFrame, cache_dir: str) -> dict:
    """CLIP embeddings (frames x dims) and frame differences of every video, cached per video."""
    import open_clip
    import torch
    from PIL import Image

    model, preprocess = None, None
    out = {}
    for row in videos.iter_rows(named=True):
        cache = os.path.join(cache_dir, os.path.splitext(row["file"])[0].replace(os.sep, "__") + ".npz")
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(row["path"]):
            data = np.load(cache)
            out[row["file"]] = (data["embedding"], data["difference"])
            continue
        if model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
            model.eval()
        frames = sample_frames(row["path"])
        present = [f for f in frames if f is not None]
        with torch.no_grad():
            batch = torch.stack([preprocess(Image.fromarray(f)) for f in present])
            features = model.encode_image(batch)
            features = (features / features.norm(dim=-1, keepdim=True)).numpy()
        embedding = np.full((len(frames), features.shape[1]), np.nan, dtype=np.float32)
        embedding[[i for i, f in enumerate(frames) if f is not None]] = features
        small = [cv2.resize(f, (160, 90)).astype(np.float32) if f is not None else None for f in frames]
        difference = np.array([np.nan] + [np.abs(a - b).mean() if a is not None and b is not None else np.nan
                                          for a, b in zip(small[1:], small[:-1])], dtype=np.float32)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, embedding=embedding, difference=difference)
        out[row["file"]] = (embedding, difference)
    return out


def text_embeddings(places: list) -> np.ndarray:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    with torch.no_grad():
        features = model.encode_text(tokenizer([PROMPT.format(place=p) for p in places]))
    return (features / features.norm(dim=-1, keepdim=True)).numpy()


def wilson(k: int, n: int) -> tuple:
    """95% Wilson interval of a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def analyse(videos: pl.DataFrame, cities: pl.DataFrame, embeddings: dict) -> pl.DataFrame:
    """One row per video and sampled frame with similarity, city and country rank and motion."""
    names = cities["city"].to_list()
    city_text = text_embeddings([f"{ENGLISH.get(c, c)}, {k}" for c, k in zip(names, cities["country"])])
    countries = sorted(cities["country"].unique().to_list())
    country_text = text_embeddings(countries)
    country_of = dict(zip(names, cities["country"]))
    rows = []
    for row in videos.iter_rows(named=True):
        embedding, difference = embeddings[row["file"]]
        city_scores, country_scores = embedding @ city_text.T, embedding @ country_text.T
        truth_city, truth_country = names.index(row["city"]), countries.index(country_of[row["city"]])
        for i, second in enumerate(SECONDS + [8]):
            if np.isnan(embedding[i, 0]):
                continue
            rows.append({
                "model": row["model"], "city": row["city"], "file": row["file"], "second": second,
                "similarity_to_first": float(embedding[i] @ embedding[0]),
                "city_rank": int((city_scores[i] > city_scores[i, truth_city]).sum()) + 1,
                "country_rank": int((country_scores[i] > country_scores[i, truth_country]).sum()) + 1,
                "difference": float(difference[i]),
            })
    return pl.DataFrame(rows)


def per_second(frames: pl.DataFrame) -> pl.DataFrame:
    """Mean and 95% interval per model and second of each measure."""
    stats = []
    for (model, second), g in frames.group_by(["model", "second"]):
        n = g.height
        row = {"model": model, "second": second, "videos": n}
        for name, hit in (("city_top1", g["city_rank"] == 1), ("city_top5", g["city_rank"] <= 5),
                          ("country_top1", g["country_rank"] == 1)):
            k = int(hit.sum())
            row[name], (row[name + "_lo"], row[name + "_hi"]) = k / n, wilson(k, n)
        for name in ("similarity_to_first", "difference"):
            values = g[name].drop_nans().drop_nulls()
            if values.len() < 2:
                row[name] = row[name + "_lo"] = row[name + "_hi"] = None
                continue
            mean, half = values.mean(), 1.96 * values.std() / math.sqrt(values.len())
            row[name], row[name + "_lo"], row[name + "_hi"] = mean, mean - half, mean + half
        stats.append(row)
    return pl.DataFrame(stats).sort(["model", "second"])


def line_figure(stats: pl.DataFrame, panels: list, name: str) -> go.Figure:
    """One panel per measure, one line per model with its 95% interval as a band."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels], horizontal_spacing=0.08)
    for col, (measure, _, axis) in enumerate(panels, start=1):
        for model, (legend, colour) in MODELS.items():
            data = stats.filter(pl.col("model") == model)
            if data.height == 0:
                continue
            x = data["second"].to_list()
            fig.add_trace(go.Scatter(x=x + x[::-1], y=data[measure + "_hi"].to_list() + data[measure + "_lo"].to_list()[::-1],
                                     fill="toself", fillcolor=colour, opacity=0.15, line=dict(width=0), mode="lines",
                                     hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=x, y=data[measure].to_list(), name=legend, legendgroup=model,
                                     showlegend=col == 1, mode="lines+markers", line=dict(color=colour, width=2),
                                     marker=dict(size=8),
                                     hovertemplate=f"{legend}<br>%{{x}} s: %{{y:.3f}}<extra></extra>"),
                          row=1, col=col)
        fig.update_xaxes(title_text="Time in the generated clip (s; 0 = reference frame)", row=1, col=col)
        fig.update_yaxes(title_text=axis, row=1, col=col)
    Crossings._layout(fig).update_layout(legend=dict(orientation="h", y=1.15, x=0))
    io.save_plotly_figure(fig, name, width=700 * len(panels), height=550)
    return fig


def city_map(cities: pl.DataFrame, counts: pl.DataFrame, name: str) -> go.Figure:
    """The CROWD cities used as reference, coloured by which generated videos they have."""
    data = cities.join(counts, on="city", how="left").with_columns(pl.col("veo3", "minimax_h3").fill_null(0))
    groups = {
        "CROWD only": (data.filter((pl.col("veo3") == 0) & (pl.col("minimax_h3") == 0)), "#9b9a96"),
        "CROWD + MiniMax H3": (data.filter((pl.col("veo3") == 0) & (pl.col("minimax_h3") > 0)), "#1baf7a"),
        "CROWD + MiniMax H3 + Veo 3.1 Fast": (data.filter(pl.col("veo3") > 0), "#eb6834"),
    }
    fig = go.Figure()
    for legend, (group, colour) in groups.items():
        fig.add_trace(go.Scattergeo(
            lon=group["lon"].to_list(), lat=group["lat"].to_list(), name=f"{legend} ({group.height})",
            mode="markers", marker=dict(size=9, color=colour, line=dict(width=1, color="white")),
            text=[f"{c}, {k}<br>Veo 3.1: {v}, MiniMax H3: {m}" for c, k, v, m in
                  zip(group["city"], group["country"], group["veo3"], group["minimax_h3"])],
            hoverinfo="text"))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#d0d0d0",
                    showland=True, landcolor="#f2f2f0", showocean=False, showframe=False,
                    lataxis_range=[-50, 75])
    Crossings._layout(fig).update_layout(legend=dict(orientation="h", y=-0.05, x=0), margin=dict(l=0, r=0, t=10, b=0))
    io.save_plotly_figure(fig, name, width=1400, height=700)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", default=os.path.join(common.root_dir, "videos", "generated_videos.csv"))
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "crowd", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--mapping", default=os.path.join(common.root_dir, "crowd", "cities.csv"),
                        help="the cities of CROWD's mapping.csv (crowd/cities.csv holds the 100), for their coordinates")
    parser.add_argument("--pdf-dir", help="also write the paper figures as PDF to this folder")
    parser.add_argument("--readme", action="store_true", help="also export the figures to readme/")
    parser.add_argument("--from-results", action="store_true",
                        help="use results/appearance/frames.csv instead of embedding the videos (the default when the "
                             "video files are not there)")
    args = parser.parse_args()
    output = common.get_output_dir()
    kept = os.path.join(common.root_dir, "results", "appearance", "frames.csv")

    videos = (pl.read_csv(args.videos, infer_schema_length=None).filter(pl.col("status") == "used")
              .with_columns(pl.col("file").map_elements(lambda f: os.path.join(common.root_dir, f),
                                                        return_dtype=pl.String).alias("path")))
    # crowd-city names a city by its locality or one of its other names (locality_aka, e.g. New Delhi for Delhi),
    # and its state field is not always the mapping's, so coordinates are looked up by name and country.
    mapping = (pl.read_csv(args.mapping, infer_schema_length=0)
               .with_columns(pl.concat_list(pl.col("locality"), pl.col("locality_aka").fill_null("")
                                            .str.strip_chars("[]").str.split(",")).alias("city"))
               .explode("city").filter(pl.col("city") != "")
               .select("city", "country", pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64))
               .unique(["city", "country"], keep="first"))
    crowd = pl.read_csv(args.crowd, infer_schema_length=None).select(pl.col("locality").alias("city"), "country")
    # Cities with generated videos but no CROWD speed (Seattle) are added from the mapping.
    extra = (mapping.filter(pl.col("city").is_in(videos["city"].unique().to_list())
                            & ~pl.col("city").is_in(crowd["city"].to_list()))
             .unique("city", keep="first").select("city", "country"))
    cities = pl.concat([crowd, extra]).join(mapping, on=["city", "country"], how="left")
    missing = cities.filter(pl.col("lat").is_null())["city"].to_list()
    if missing:
        common.logger.warning(f"No coordinates in the mapping for: {', '.join(missing)}")

    counts = videos.group_by("city").agg(*[(pl.col("model") == m).sum().alias(m) for m in MODELS])
    fig_map = city_map(cities, counts, "map_cities")

    have_videos = all(os.path.isfile(path) for path in videos["path"])
    if args.from_results or not have_videos:
        if not os.path.isfile(kept):
            raise SystemExit(f"No video files and no {kept}: fetch the videos (see README) to embed them.")
        common.logger.info(f"Reading the CLIP results of every frame from {kept} instead of embedding the videos.")
        frames = pl.read_csv(kept)
    else:
        embeddings = embed_videos(videos, os.path.join(output, "appearance"))
        frames = analyse(videos, cities.select("city", "country"), embeddings)
        os.makedirs(os.path.dirname(kept), exist_ok=True)
        frames.write_csv(kept)
    frames.write_csv(os.path.join(output, "appearance_frames.csv"))
    stats = per_second(frames)
    fig_recognition = line_figure(stats, [("city_top5", "City in CLIP's top 5 (of 100)", "Share of videos"),
                                          ("country_top1", "Country ranked first by CLIP", "Share of videos")],
                                  "line_city_recognition")
    fig_drift = line_figure(stats, [("similarity_to_first", "Similarity to the reference frame", "CLIP cosine similarity"),
                        ("difference", "Change since the previous second", "Mean absolute pixel difference")],
                "line_appearance_drift")

    end, start = pl.col("second") == 8, pl.col("second") == 0
    summary = frames.group_by("model").agg(
        pl.col("file").n_unique().alias("videos"), pl.col("city").n_unique().alias("cities"),
        (pl.col("city_rank").filter(start) <= 5).mean().alias("city_top5_first"),
        (pl.col("city_rank").filter(end) <= 5).mean().alias("city_top5_last"),
        (pl.col("country_rank").filter(start) == 1).mean().alias("country_top1_first"),
        (pl.col("country_rank").filter(end) == 1).mean().alias("country_top1_last"),
        pl.col("similarity_to_first").filter(end).mean().alias("similarity_last"),
        (pl.col("difference").filter(pl.col("second") > 0) < 1).mean().alias("still_seconds"),
    ).sort("model", descending=True)
    summary.write_csv(os.path.join(output, "appearance_summary.csv"))
    frames.group_by("model", "city").agg(
        pl.col("file").n_unique().alias("videos"),
        (pl.col("city_rank").filter(start) <= 5).mean().alias("city_top5_first"),
        (pl.col("city_rank").filter(end) <= 5).mean().alias("city_top5_last"),
        pl.col("similarity_to_first").filter(end).mean().alias("similarity_last"),
    ).sort("model", "city").write_csv(os.path.join(output, "appearance_cities.csv"))
    print(summary)

    if args.readme:
        readme_figures.save(fig_map, "map_cities", width=1400, height=700)
        readme_figures.save(fig_recognition, "line_city_recognition", width=1400, height=550)
        readme_figures.save(fig_drift, "line_appearance_drift", width=1400, height=550)
    if args.pdf_dir:
        fig_map.write_image(os.path.join(args.pdf_dir, "map_cities.pdf"), width=1400, height=700)
        fig_recognition.write_image(os.path.join(args.pdf_dir, "line_city_recognition.pdf"), width=1400, height=550)


if __name__ == "__main__":
    main()
