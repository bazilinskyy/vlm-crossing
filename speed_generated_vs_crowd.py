"""Figure of the median crossing speed per city: CROWD next to each set of generated videos.

Reads
    --crowd      crowd-city's crowd_crossing_speed_by_city.csv (Waymo-calibrated m/s from YOLO tracks of CROWD)
    --generated  the same table per video model, with a column model (veo3, minimax_h3); cities without
                 a value get no bar, so the figure can be drawn before the videos are analysed
    --videos     data/generated_videos.csv from fetch_generated_videos.py (videos per model and city)
and writes <output>/figures/speed_generated_vs_crowd.html (and .png/.eps when save_images is true).

    uv run python speed_generated_vs_crowd.py --crowd <crowd-city>/_output/crossing_speed/crowd_crossing_speed_by_city.csv
"""

import argparse
import os

import polars as pl

import common
import readme_figures
from utils.plotting.crossings import Crossings

# Bars per city, in this order, with the reference palette's first three categorical colours.
SOURCES = {
    "crowd": ("CROWD", "#2a78d6"),
    "veo3": ("Veo 3", "#eb6834"),
    "minimax_h3": ("MiniMax H3", "#1baf7a"),
}
SPEED_COLUMNS = {"locality": "city", "median_crossing_speed_mps": "median", "crossing_speed_mps_q25": "q25",
                 "crossing_speed_mps_q75": "q75", "reliable_metric_speed_tracks": "n"}


def read_speeds(path: str, source: str = None) -> pl.DataFrame:
    """Per-city speeds of crowd-city's by-city table, renamed to source, city, median, q25, q75, n."""
    table = pl.read_csv(path, infer_schema_length=None)
    if source is not None:
        table = table.with_columns(pl.lit(source).alias("model"))
    return (table.rename({k: v for k, v in SPEED_COLUMNS.items() if k in table.columns})
            .select(pl.col("model").alias("source"), "city", *[pl.col(c).cast(pl.Float64) for c in
                                                               ("median", "q25", "q75", "n")]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "data", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--generated", default=os.path.join(common.get_output_dir(), "generated_speed_by_city.csv"))
    parser.add_argument("--videos", default=os.path.join(common.root_dir, "data", "generated_videos.csv"))
    parser.add_argument("--pdf", help="also write the figure as PDF to this path (e.g. the paper's figures folder)")
    parser.add_argument("--readme", action="store_true", help="also export the figure to readme/")
    args = parser.parse_args()

    speeds = [read_speeds(args.crowd, "crowd")]
    if os.path.exists(args.generated):
        speeds.append(read_speeds(args.generated))
    videos = (pl.read_csv(args.videos, infer_schema_length=None).filter(pl.col("status") == "used")
              .group_by(pl.col("model").alias("source"), "city").agg(pl.len().alias("videos")))
    missing = videos.join(speeds[0], on="city", how="anti")["city"].unique().sort().to_list()
    if missing:
        common.logger.warning(f"Generated videos of cities without CROWD speed: {', '.join(missing)}")
    fig = Crossings().speed_sources_per_city(pl.concat(speeds), videos, SOURCES, "speed_generated_vs_crowd")
    if args.pdf:
        fig.write_image(args.pdf, width=1000, height=fig.layout.height)
    if args.readme:
        readme_figures.save(fig, "speed_generated_vs_crowd", width=1000, height=fig.layout.height)


if __name__ == "__main__":
    main()
