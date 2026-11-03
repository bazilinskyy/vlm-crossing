"""Statistical tests and figures on the crossing speed of the generated clips, for the paper.

Reads the same tracked sets as generated_speeds.py (Veo 3.1 Fast, MiniMax H3 and the real continuations of the MiniMax
reference frames) and CROWD's per-city table, and writes to <output>:

- generated_statistics.csv  one row per test: what was compared, the estimate with its 95% interval, the test
                            statistic and p-value (with --scaled also the real continuations scaled to MiniMax H3's
                            864x480 against the original ones and against MiniMax H3)
- generated_reliability.csv per set: split-half reliability of the city medians, and the clips per city that a
                            reliability of 0.8 would need

and the figures ecdf_speed_sets (speed of every crossing per set) and hist_speed_paired_difference (MiniMax H3 minus
its real continuation, per clip); with --readme also to readme/, with --pdf-dir also as PDF.

Crossings within a clip, and clips within a city, are not independent, so intervals come from bootstraps that resample
whole clips (set medians), clip pairs (paired differences) or cities (city-level statistics), and tests are run on
clip or city medians rather than on single crossings.

    uv run python speed_statistics.py --readme
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats

import common
import readme_figures
from generated_speeds import CROWD, MIN_TRACKS, SETS, SPEED, folder_cities, read_run
from utils.plotting.crossings import Crossings, io

REPS = 5000
SEED = 2026
TARGET_RELIABILITY = 0.8
DECISION = "hesitation_s"  # crossing decision time, only for pedestrians seen standing before they step onto the road


def bootstrap(groups: list, statistic, reps: int = REPS) -> tuple:
    """95% percentile interval of statistic(list of groups) over resamples of whole groups."""
    rng = np.random.default_rng(SEED)
    values = [statistic([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(reps)]
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def pooled_median(groups: list) -> float:
    return float(np.median(np.concatenate(groups)))


def row(analysis: str, set_: str, reference: str, n: int, estimate: float, ci: tuple = (None, None),
        statistic: str = "", value: float = None, p: float = None) -> dict:
    return {"analysis": analysis, "set": set_, "reference": reference, "n": n, "estimate": estimate,
            "ci_low": ci[0], "ci_high": ci[1], "statistic": statistic, "value": value, "p": p}


def clip_table(videos: pl.DataFrame, crossings: pl.DataFrame) -> pl.DataFrame:
    """Per set and clip: crossings, m/s speeds and their median, and decision times."""
    return (videos.join(crossings.group_by("set", "video").agg(
        pl.len().alias("crossings"), pl.col(SPEED).drop_nulls().alias("speeds"),
        pl.col(DECISION).drop_nulls().alias("decisions")), on=["set", "video"], how="left")
        .with_columns(pl.col("crossings").fill_null(0), pl.col("speeds", "decisions").fill_null([]))
        .with_columns(pl.col("speeds").list.median().alias("median_speed")))


def usability_tests(clips: pl.DataFrame) -> list:
    rows = []
    pairs = (clips.filter(pl.col("set") == "minimax_h3").select("video", "crossings")
             .join(clips.filter(pl.col("set") == "real_minimax").select("video", pl.col("crossings").alias("real")),
                   on="video"))
    generated_only = pairs.filter((pl.col("crossings") > 0) & (pl.col("real") == 0)).height
    real_only = pairs.filter((pl.col("crossings") == 0) & (pl.col("real") > 0)).height
    test = stats.binomtest(generated_only, generated_only + real_only)  # exact McNemar test
    rows.append(row("clips with a crossing: MiniMax H3 minus real continuation (exact McNemar)", "minimax_h3",
                    "real_minimax", pairs.height, (generated_only - real_only) / pairs.height,
                    statistic="discordant pairs generated-only/real-only", value=generated_only / max(real_only, 1),
                    p=test.pvalue))
    rho = stats.spearmanr(pairs["crossings"], pairs["real"])
    rows.append(row("crossings per clip, MiniMax H3 against real continuation (Spearman)", "minimax_h3",
                    "real_minimax", pairs.height, rho.statistic, statistic="rho", value=rho.statistic,
                    p=rho.pvalue))
    veo_cities = clips.filter(pl.col("set") == "veo3")["city"].unique()
    shared = clips.filter(pl.col("city").is_in(veo_cities.implode()))
    counts = {s: (shared.filter((pl.col("set") == s) & (pl.col("crossings") > 0)).height,
                  shared.filter(pl.col("set") == s).height) for s in ("veo3", "minimax_h3")}
    (a, n), (b, m) = counts["veo3"], counts["minimax_h3"]
    odds, p = stats.fisher_exact([[a, n - a], [b, m - b]])
    rows.append(row("clips with a crossing: Veo 3.1 Fast minus MiniMax H3 on the Veo cities (Fisher)", "veo3",
                    "minimax_h3", n + m, a / n - b / m, statistic="odds ratio", value=odds, p=p))
    return rows


def paired_speed(with_speed: pl.DataFrame, a: str, b: str, label: str) -> list:
    """Clip median speed of set a against set b on the same clips: Wilcoxon signed-rank test and Spearman."""
    pairs = (with_speed.filter(pl.col("set") == a).select("video", "median_speed")
             .join(with_speed.filter(pl.col("set") == b).select("video", pl.col("median_speed").alias("other")),
                   on="video"))
    if pairs.height < 5:
        return []
    difference = (pairs["median_speed"] - pairs["other"]).to_numpy()
    test = stats.wilcoxon(difference)
    rho = stats.spearmanr(pairs["median_speed"], pairs["other"])
    return [row(f"clip median speed: {label} (Wilcoxon signed-rank)", a, b, len(difference),
                float(np.median(difference)), bootstrap([np.array([d]) for d in difference], pooled_median), "W",
                test.statistic, test.pvalue),
            row(f"clip median speed: {label} (Spearman)", a, b, pairs.height, rho.statistic, statistic="rho",
                value=rho.statistic, p=rho.pvalue)]


def speed_tests(clips: pl.DataFrame, crowd: pl.DataFrame) -> list:
    rows = []
    with_speed = clips.filter(pl.col("speeds").list.len() > 0)
    for s in SETS:
        groups = [np.array(v) for v in with_speed.filter(pl.col("set") == s)["speeds"].to_list()]
        if groups:
            rows.append(row("median crossing speed (m/s), clip bootstrap", s, "", sum(map(len, groups)),
                            pooled_median(groups), bootstrap(groups, pooled_median)))

    rows += paired_speed(with_speed, "minimax_h3", "real_minimax", "MiniMax H3 minus real continuation")

    veo_cities = with_speed.filter(pl.col("set") == "veo3")["city"].unique().implode()
    shared = with_speed.filter(pl.col("city").is_in(veo_cities))
    veo, minimax = (shared.filter(pl.col("set") == s)["median_speed"].to_numpy() for s in ("veo3", "minimax_h3"))
    test = stats.mannwhitneyu(veo, minimax)
    rows.append(row("clip median speed: Veo 3.1 Fast minus MiniMax H3 on the Veo cities (Mann-Whitney)", "veo3",
                    "minimax_h3", len(veo) + len(minimax), float(np.median(veo) - np.median(minimax)),
                    statistic="U", value=test.statistic, p=test.pvalue))

    # City level: each set's city medians against CROWD's and against the real continuations'.
    cities = (with_speed.group_by("set", "city").agg(pl.col("speeds").explode())
              .filter(pl.col("speeds").list.len() >= MIN_TRACKS)
              .with_columns(pl.col("speeds").list.median().alias("median")))
    references = {"crowd": crowd.filter(pl.col("n") >= MIN_TRACKS).select(pl.col("locality").alias("city"),
                                                                          pl.col("median").alias("ref")),
                  "real_minimax": cities.filter(pl.col("set") == "real_minimax")
                  .select("city", pl.col("median").alias("ref"))}
    for s in SETS:
        for reference, ref in references.items():
            if s == reference:
                continue
            pair = cities.filter(pl.col("set") == s).select("city", "median").join(ref, on="city")
            if pair.height < 5:
                continue
            x, y = pair["ref"].to_numpy(), pair["median"].to_numpy()
            points = [np.array([[a, b]]) for a, b in zip(x, y)]
            rho = stats.spearmanr(x, y)
            rows.append(row("city medians, rank correlation (Spearman, city bootstrap)", s, reference, pair.height,
                            rho.statistic, bootstrap(points, lambda g: stats.spearmanr(*np.concatenate(g).T).statistic),
                            "rho", rho.statistic, rho.pvalue))
            test = stats.wilcoxon(y - x)
            rows.append(row("city medians, difference in m/s (Wilcoxon signed-rank, city bootstrap)", s, reference,
                            pair.height, float(np.median(y - x)),
                            bootstrap(points, lambda g: float(np.median(np.diff(np.concatenate(g), axis=1)))),
                            "W", test.statistic, test.pvalue))

            def spread(g):
                both = np.concatenate(g)
                return stats.iqr(both[:, 1]) / stats.iqr(both[:, 0])
            rows.append(row("city medians, interquartile range relative to the reference (city bootstrap)", s,
                            reference, pair.height, spread(points), bootstrap(points, spread)))

    # Does the city explain speed within a set at all? Kruskal-Wallis across cities on clip medians.
    for s in SETS:
        mine = with_speed.filter(pl.col("set") == s)
        keep = mine.group_by("city").len().filter(pl.col("len") >= 3)["city"].implode()
        groups = [g["median_speed"].to_numpy() for _, g in mine.filter(pl.col("city").is_in(keep)).group_by("city")]
        if len(groups) < 3:
            continue
        test = stats.kruskal(*groups)
        n, k = sum(map(len, groups)), len(groups)
        rows.append(row("clip median speed across cities (Kruskal-Wallis, eta squared H)", s, "", n,
                        (test.statistic - k + 1) / (n - k), statistic=f"H ({k} cities)", value=test.statistic,
                        p=test.pvalue))
    return rows


def decision_tests(clips: pl.DataFrame) -> list:
    """Crossing decision time: how often it can be measured, and its median, per set and per MiniMax/real pair."""
    rows = []
    for s in SETS:
        mine = clips.filter(pl.col("set") == s)
        groups = [np.array(v) for v in mine["decisions"].to_list() if len(v)]
        if groups:
            rows.append(row("crossing decision time (s), clip bootstrap", s, "", sum(map(len, groups)),
                            pooled_median(groups), bootstrap(groups, pooled_median),
                            "clips with a decision time", len(groups)))
    pairs = (clips.filter(pl.col("set") == "minimax_h3").select("video", pl.col("decisions").list.len().alias("g"))
             .join(clips.filter(pl.col("set") == "real_minimax")
                   .select("video", pl.col("decisions").list.len().alias("r")), on="video"))
    generated_only = pairs.filter((pl.col("g") > 0) & (pl.col("r") == 0)).height
    real_only = pairs.filter((pl.col("g") == 0) & (pl.col("r") > 0)).height
    if generated_only + real_only:
        rows.append(row("clips with a decision time: MiniMax H3 minus real continuation (exact McNemar)",
                        "minimax_h3", "real_minimax", pairs.height, (generated_only - real_only) / pairs.height,
                        statistic="discordant pairs generated-only/real-only",
                        value=generated_only / max(real_only, 1),
                        p=stats.binomtest(generated_only, generated_only + real_only).pvalue))
    return rows


def reliability(clips: pl.DataFrame, reps: int = 2000) -> pl.DataFrame:
    """Split-half reliability of the city medians: split each city's clips at random into halves, take the median
    speed of each half, correlate the halves across cities and step the correlation up to the full number of clips
    (Spearman-Brown)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SETS:
        cities = [[np.array(v) for v in g["speeds"].to_list()]
                  for _, g in clips.filter((pl.col("set") == s) & (pl.col("speeds").list.len() > 0)).group_by("city")]
        cities = [c for c in cities if len(c) >= 4]
        if len(cities) < 5:
            continue
        values = []
        for _ in range(reps):
            halves = []
            for c in cities:
                order = rng.permutation(len(c))
                halves.append((np.median(np.concatenate([c[i] for i in order[:len(c) // 2]])),
                               np.median(np.concatenate([c[i] for i in order[len(c) // 2:]]))))
            r = stats.spearmanr(*np.array(halves).T).statistic
            values.append(2 * r / (1 + r) if r > -1 else np.nan)
        full = float(np.nanmean(values))
        clips_per_city = float(np.mean([len(c) for c in cities]))
        needed = (clips_per_city * TARGET_RELIABILITY * (1 - full) / (full * (1 - TARGET_RELIABILITY))
                  if full > 0 else None)
        rows.append({"set": s, "cities": len(cities), "clips_per_city": clips_per_city, "reliability": full,
                     "reliability_low": float(np.nanpercentile(values, 2.5)),
                     "reliability_high": float(np.nanpercentile(values, 97.5)),
                     "ceiling_of_correlation": float(np.sqrt(full)) if full > 0 else 0.0,
                     f"clips_per_city_for_{TARGET_RELIABILITY}": needed})
    return pl.DataFrame(rows)


def ecdf_figure(clips: pl.DataFrame, crowd: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    for s, (legend, colour) in SETS.items():
        speeds = np.sort(np.concatenate([np.array(v) for v in clips.filter(pl.col("set") == s)["speeds"].to_list()]))
        if len(speeds) == 0:
            continue
        fig.add_trace(go.Scatter(x=speeds, y=np.arange(1, len(speeds) + 1) / len(speeds), mode="lines",
                                 name=f"{legend} ({len(speeds):,} crossings)", line=dict(color=colour, width=2),
                                 hovertemplate=f"{legend}<br>%{{x:.2f}} m/s: %{{y:.0%}}<extra></extra>"))
    reference = crowd.filter(pl.col("n") >= MIN_TRACKS)["median"].median()
    fig.add_vline(x=reference, line=dict(color=CROWD[1], width=2, dash="dash"))
    fig.add_annotation(x=reference, y=0.02, text=f"CROWD: median of city medians ({reference:.2f})", showarrow=False,
                       xanchor="left", xshift=4, font=dict(color=CROWD[1]))
    fig.update_layout(xaxis_title="Crossing speed (m/s)", yaxis_title="Share of crossings at or below",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    Crossings._layout(fig).update_layout(margin=dict(l=20, r=20, t=40, b=60))
    io.save_plotly_figure(fig, "ecdf_speed_sets", width=700, height=500)
    return fig


def paired_figure(clips: pl.DataFrame) -> go.Figure:
    with_speed = clips.filter(pl.col("speeds").list.len() > 0)
    pairs = (with_speed.filter(pl.col("set") == "minimax_h3").select("video", "median_speed")
             .join(with_speed.filter(pl.col("set") == "real_minimax")
                   .select("video", pl.col("median_speed").alias("real")), on="video"))
    difference = (pairs["median_speed"] - pairs["real"]).to_numpy()
    fig = go.Figure(go.Histogram(x=difference, xbins=dict(start=-1.0, end=1.0, size=0.1),
                                 marker=dict(color=SETS["minimax_h3"][1], line=dict(color="white", width=2)),
                                 name="Clip pairs", hovertemplate="%{x} m/s: %{y} clips<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#9b9a96", width=1, dash="dot"))
    fig.add_vline(x=float(np.median(difference)), line=dict(color="#0b0b0b", width=2, dash="dash"))
    fig.add_annotation(x=float(np.median(difference)), yref="paper", y=1.0, showarrow=False, xanchor="right",
                       xshift=-4, text=f"median {np.median(difference):+.2f} m/s ({len(difference)} clips)")
    fig.update_layout(xaxis_title="Clip median speed, MiniMax H3 minus real (m/s)",
                      yaxis_title="Clips", showlegend=False, bargap=0)
    Crossings._layout(fig).update_layout(margin=dict(l=20, r=20, t=40, b=60))
    io.save_plotly_figure(fig, "hist_speed_paired_difference", width=700, height=500)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=[f"{s}=results/{s}" for s in SETS],
                        help="SET=OUTPUT_FOLDER of analysis.py, SET one of " + ", ".join(SETS))
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "crowd", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--generated", default=os.path.join(common.root_dir, "videos", "generated_videos.csv"))
    parser.add_argument("--scaled", help="output folder of the real continuations scaled to 864x480 (robustness check: "
                                         "is the speed difference a matter of resolution?)")
    parser.add_argument("--pdf-dir", help="also write the figures as PDF to this folder (e.g. the paper's figures)")
    parser.add_argument("--readme", action="store_true", help="also export the figures to readme/")
    args = parser.parse_args()

    cities = folder_cities(args.generated)
    runs = [read_run(*run.split("=", 1), cities) for run in args.runs]
    clips = clip_table(pl.concat([v for v, _ in runs]), pl.concat([c for _, c in runs], how="diagonal_relaxed"))
    crowd = (pl.read_csv(args.crowd, infer_schema_length=None)
             .select("locality", pl.col("median_crossing_speed_mps").cast(pl.Float64).alias("median"),
                     pl.col("reliable_metric_speed_tracks").cast(pl.Int64).alias("n")))

    output = common.get_output_dir()
    rows = usability_tests(clips) + speed_tests(clips, crowd) + decision_tests(clips)
    if args.scaled:
        scaled = read_run("real_minimax_480", args.scaled, cities)
        both = pl.concat([clip_table(*scaled), clips], how="diagonal_relaxed").filter(pl.col("speeds").list.len() > 0)
        rows += paired_speed(both, "real_minimax_480", "real_minimax", "real continuation at 864x480 minus at 1280x720")
        rows += paired_speed(both, "minimax_h3", "real_minimax_480", "MiniMax H3 minus real continuation at 864x480")
    tests = pl.DataFrame(rows, strict=False)
    tests.write_csv(os.path.join(output, "generated_statistics.csv"))
    split = reliability(clips)
    split.write_csv(os.path.join(output, "generated_reliability.csv"))
    with pl.Config(tbl_rows=60, tbl_cols=20, fmt_str_lengths=90, tbl_width_chars=250):
        common.logger.info(f"Tests:\n{tests}")
        common.logger.info(f"Split-half reliability of the city medians:\n{split}")

    figures = {"ecdf_speed_sets": ecdf_figure(clips, crowd), "hist_speed_paired_difference": paired_figure(clips)}
    for name, fig in figures.items():
        if args.pdf_dir:
            fig.write_image(os.path.join(args.pdf_dir, name + ".pdf"), width=700, height=500)
        if args.readme:
            readme_figures.save(fig, name, width=700, height=500)


if __name__ == "__main__":
    main()
