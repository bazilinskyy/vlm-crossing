"""Crossing speed per city and model from the tracked clips, and how it compares with CROWD.

Reads <run>/crossings.csv and <run>/videos.csv of each tracked set (analysis.py with speed_model set, so speeds are
crowd-city's Waymo-calibrated m/s): Veo 3.1 Fast, MiniMax H3, and the real 8 seconds that follow each MiniMax
reference frame in its CROWD video (paired_real_clips.py). As in crowd-city's table of CROWD, a crossing's speed is the
bounding-box speed of a track that passes the speed model's reliability gates. Writes to <output>:

- generated_speed_by_city.csv   per set and city: clips, crossings, median and quartiles of the speed (the --generated
                                input of speed_generated_vs_crowd.py)
- generated_usability.csv       per set: share of clips with a counted crossing and with an m/s speed (RQ1)
- generated_correlations.csv    per set: Spearman rank correlation of the city medians with CROWD's and with the real
                                continuations', and the spread of the city medians relative to the reference (RQ2)
- generated_paired_clips.csv    per MiniMax clip: crossings and median speed in the generated and the real continuation

and the figures scatter_speed_city_vs_crowd and scatter_speed_paired (with --readme also to readme/, with --pdf-dir
also as PDF for the paper).

    uv run python generated_speeds.py --runs veo3=_output/veo3 minimax_h3=_output/minimax_h3 \\
        real_minimax=_output/real_minimax --readme
"""

import argparse
import os

import plotly.graph_objects as go
import polars as pl

import common
import readme_figures
from appearance_analysis import wilson
from utils.plotting.crossings import Crossings, io

SETS = {
    "veo3": ("Veo 3.1 Fast", "#eb6834"),
    "minimax_h3": ("MiniMax H3", "#1baf7a"),
    "real_minimax": ("Real continuation", "#eda100"),
}
CROWD = ("CROWD", "#2a78d6")
SPEED = "speed_bbox_index"  # m/s when analysis.py ran with speed_model
MIN_TRACKS = 3  # cities with fewer m/s speeds are left out of the correlations and scatter plots


def folder_cities(generated: str) -> dict:
    """Folder name under data/<model>/ -> CROWD city name, from fetch_generated_videos.py's table."""
    table = pl.read_csv(generated, infer_schema_length=None).filter(pl.col("file").is_not_null())
    return {os.path.basename(os.path.dirname(f)): c for f, c in zip(table["file"], table["city"])}


def read_run(name: str, folder: str, cities: dict) -> tuple:
    """Videos and crossings of one tracked set, with CROWD city names."""
    videos = pl.read_csv(os.path.join(folder, "videos.csv"), schema_overrides={"city": pl.Utf8, "video": pl.Utf8})
    crossings = pl.read_csv(os.path.join(folder, "crossings.csv"), infer_schema_length=None,
                            schema_overrides={"city": pl.Utf8, "video": pl.Utf8, "track_id": pl.Utf8})
    units = crossings["speed_unit"].unique().to_list() if "speed_unit" in crossings.columns else ["relative"]
    if crossings.height and units != ["m/s"]:
        raise SystemExit(f"{folder}: speeds are {units}, not m/s; run analysis.py with speed_model set.")
    rename = pl.col("city").replace(cities)
    return (videos.with_columns(rename, pl.lit(name).alias("set")).select("set", "city", "video"),
            crossings.with_columns(rename, pl.lit(name).alias("set")))


def by_city(videos: pl.DataFrame, crossings: pl.DataFrame) -> pl.DataFrame:
    clips = videos.group_by("set", "city").agg(pl.len().alias("clips"))
    speed = pl.col(SPEED).drop_nulls()
    stats = crossings.group_by("set", "city").agg(
        pl.len().alias("crossings"), pl.col("video").n_unique().alias("clips_with_crossing"),
        speed.len().alias("reliable_metric_speed_tracks"), speed.median().alias("median_crossing_speed_mps"),
        speed.quantile(0.25, "linear").alias("crossing_speed_mps_q25"),
        speed.quantile(0.75, "linear").alias("crossing_speed_mps_q75"))
    return (clips.join(stats, on=["set", "city"], how="left")
            .with_columns(pl.col("crossings", "clips_with_crossing", "reliable_metric_speed_tracks").fill_null(0))
            .rename({"set": "model", "city": "locality"}).sort("model", "locality"))


def usability(videos: pl.DataFrame, crossings: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name in SETS:
        clips = videos.filter(pl.col("set") == name)
        if clips.height == 0:
            continue
        mine = crossings.filter(pl.col("set") == name)
        with_crossing = mine["video"].n_unique()
        with_speed = mine.filter(pl.col(SPEED).is_not_null())["video"].n_unique()
        rows.append({"model": name, "clips": clips.height, "cities": clips["city"].n_unique(),
                     "clips_with_crossing": with_crossing, "share_with_crossing": with_crossing / clips.height,
                     "share_with_crossing_ci": "%.2f-%.2f" % wilson(with_crossing, clips.height),
                     "crossings": mine.height, "crossings_per_clip": mine.height / clips.height,
                     "clips_with_speed": with_speed, "share_with_speed": with_speed / clips.height,
                     "speeds": mine[SPEED].drop_nulls().len(), "median_speed_mps": mine[SPEED].median()})
    return pl.DataFrame(rows)


def correlations(cities: pl.DataFrame, crowd: pl.DataFrame) -> pl.DataFrame:
    """Rank correlation and relative spread of the city medians of each set against each reference."""
    usable = cities.filter(pl.col("reliable_metric_speed_tracks") >= MIN_TRACKS)
    references = {"crowd": crowd.filter(pl.col("n") >= MIN_TRACKS).select("locality", pl.col("median").alias("ref")),
                  "real_minimax": usable.filter(pl.col("model") == "real_minimax")
                  .select("locality", pl.col("median_crossing_speed_mps").alias("ref"))}
    rows = []
    for name in SETS:
        for reference, ref in references.items():
            if name == reference:
                continue
            pair = (usable.filter(pl.col("model") == name).select("locality", "median_crossing_speed_mps")
                    .join(ref, on="locality"))
            if pair.height < 3:
                continue
            x, y = pair["ref"], pair["median_crossing_speed_mps"]
            rows.append({"model": name, "reference": reference, "cities": pair.height,
                         "spearman": pair.select(pl.corr("median_crossing_speed_mps", "ref", method="spearman"))
                         .item(),
                         "median_difference_mps": (y - x).median(),
                         "spread_ratio": (y.quantile(0.75) - y.quantile(0.25)) / (x.quantile(0.75) - x.quantile(0.25))
                         if x.quantile(0.75) > x.quantile(0.25) else None})
    return pl.DataFrame(rows)


def paired(crossings: pl.DataFrame, videos: pl.DataFrame) -> pl.DataFrame:
    """Per MiniMax clip: crossings and median m/s speed in the generated clip and in the real continuation."""
    per_clip = (videos.join(crossings.group_by("set", "video").agg(
        pl.len().alias("crossings"), pl.col(SPEED).drop_nulls().len().alias("speeds"),
        pl.col(SPEED).median().alias("median_speed_mps")), on=["set", "video"], how="left")
        .with_columns(pl.col("crossings", "speeds").fill_null(0)))
    generated = per_clip.filter(pl.col("set") == "minimax_h3").drop("set")
    real = per_clip.filter(pl.col("set") == "real_minimax").drop("set", "city")
    return generated.join(real, on="video", how="inner", suffix="_real").sort("city", "video")


def scatter(points: dict, x_title: str, y_title: str, name: str) -> go.Figure:
    """One point per city and set: reference median (x) against the set's median (y), with the diagonal."""
    fig = go.Figure()
    values = [v for table in points.values() for v in table["x"].to_list() + table["y"].to_list()]
    lo, hi = (min(values) - 0.05, max(values) + 0.05) if values else (0.5, 2.0)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#9b9a96", width=1, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    for name_, table in points.items():
        legend, colour = SETS[name_]
        fig.add_trace(go.Scatter(
            x=table["x"], y=table["y"], mode="markers", name=f"{legend} ({table.height} cities)",
            marker=dict(color=colour, size=9, line=dict(color="white", width=1)),
            customdata=table.select("locality", "n_x", "n_y").rows(),
            hovertemplate=(f"{legend}<br>%{{customdata[0]}}<br>{x_title}: %{{x:.2f}} m/s (%{{customdata[1]}})<br>"
                           f"{legend}: %{{y:.2f}} m/s (%{{customdata[2]}})<extra></extra>")))
    fig.update_layout(xaxis_title=x_title + " (m/s)", yaxis_title=y_title + " (m/s)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
    Crossings._layout(fig).update_layout(margin=dict(l=20, r=20, t=40, b=60))
    io.save_plotly_figure(fig, name, width=700, height=700)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=[f"{s}=_output/{s}" for s in SETS],
                        help="SET=OUTPUT_FOLDER of analysis.py, SET one of " + ", ".join(SETS))
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "data", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--generated", default=os.path.join(common.root_dir, "data", "generated_videos.csv"))
    parser.add_argument("--pdf-dir", help="also write the figures as PDF to this folder (e.g. the paper's figures)")
    parser.add_argument("--readme", action="store_true", help="also export the figures to readme/")
    args = parser.parse_args()

    cities = folder_cities(args.generated)
    runs = [read_run(*run.split("=", 1), cities) for run in args.runs]
    videos = pl.concat([v for v, _ in runs])
    crossings = pl.concat([c for _, c in runs], how="diagonal_relaxed")
    crowd = (pl.read_csv(args.crowd, infer_schema_length=None)
             .select("locality", pl.col("median_crossing_speed_mps").cast(pl.Float64).alias("median"),
                     pl.col("reliable_metric_speed_tracks").cast(pl.Int64).alias("n")))

    output = common.get_output_dir()
    city_table = by_city(videos, crossings)
    city_table.write_csv(os.path.join(output, "generated_speed_by_city.csv"))
    use = usability(videos, crossings)
    use.write_csv(os.path.join(output, "generated_usability.csv"))
    corr = correlations(city_table, crowd)
    corr.write_csv(os.path.join(output, "generated_correlations.csv"))
    pairs = paired(crossings, videos)
    pairs.write_csv(os.path.join(output, "generated_paired_clips.csv"))
    with pl.Config(tbl_rows=20, tbl_cols=20, fmt_str_lengths=20, tbl_width_chars=200):
        common.logger.info(f"Usability:\n{use}")
        common.logger.info(f"Correlations of the city medians:\n{corr}")

    usable = city_table.filter(pl.col("reliable_metric_speed_tracks") >= MIN_TRACKS)
    reference = crowd.filter(pl.col("n") >= MIN_TRACKS)
    against_crowd = {s: usable.filter(pl.col("model") == s)
                     .select("locality", pl.col("median_crossing_speed_mps").alias("y"),
                             pl.col("reliable_metric_speed_tracks").alias("n_y"))
                     .join(reference.select("locality", pl.col("median").alias("x"), pl.col("n").alias("n_x")),
                           on="locality") for s in SETS}
    real = usable.filter(pl.col("model") == "real_minimax").select(
        "locality", pl.col("median_crossing_speed_mps").alias("x"), pl.col("reliable_metric_speed_tracks").alias("n_x"))
    against_real = {"minimax_h3": usable.filter(pl.col("model") == "minimax_h3").select(
        "locality", pl.col("median_crossing_speed_mps").alias("y"), pl.col("reliable_metric_speed_tracks").alias("n_y"))
        .join(real, on="locality")}
    figures = {
        "scatter_speed_city_vs_crowd": scatter({k: v for k, v in against_crowd.items() if v.height}, "CROWD",
                                               "Clips", "scatter_speed_city_vs_crowd"),
        "scatter_speed_paired": scatter(against_real, "Real continuation", "MiniMax H3", "scatter_speed_paired"),
    }
    for name, fig in figures.items():
        if args.pdf_dir:
            fig.write_image(os.path.join(args.pdf_dir, name + ".pdf"), width=700, height=700)
        if args.readme:
            readme_figures.save(fig, name, width=700, height=700)


if __name__ == "__main__":
    main()
