"""Plotly figures of crossing speed and hesitation time per city."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

import common
from utils.plotting.io import IO

io = IO()

METRIC_LABELS = {
    "speed_index": "Crossing speed (relative index, 1 = video median)",
    "speed_bbox_index": "Speed, bounding boxes (relative index)",
    "speed_seg_index": "Speed, on-road frames (relative index)",
    "hesitation_s": "Hesitation time (s)",
    "hesitation_bbox_s": "Hesitation, bounding boxes (s)",
    "hesitation_seg_s": "Hesitation before road entry (s)",
    "crossing_time_s": "Crossing time (s)",
}


class Crossings:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _layout(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            font=dict(family=common.get_configs("font_family"), size=common.get_configs("font_size")),
            margin=dict(l=70, r=20, t=50, b=80),
        )
        return fig

    def bar_mean_per_city(self, summary: pl.DataFrame, mean_col: str, std_col: str, n_col: str, label: str,
                          name: str, reference: float = None) -> None:
        """Bar chart of a per-city mean with one standard deviation as error bar, sorted by the mean."""
        data = summary.filter(pl.col(mean_col).is_not_null()).sort(mean_col, descending=True)
        if data.height == 0:
            return
        fig = go.Figure(go.Bar(
            x=data["city"].to_list(),
            y=data[mean_col].to_list(),
            error_y=dict(type="data", array=data[std_col].fill_null(0).to_list()),
            text=[f"n={n}" for n in data[n_col].fill_null(0).to_list()],
            textposition="outside",
        ))
        if reference is not None:
            fig.add_hline(y=reference, line_dash="dot", line_color="grey")
        fig.update_layout(xaxis_title="City", yaxis_title=label)
        io.save_plotly_figure(self._layout(fig), name)

    def crossings_per_city(self, summary: pl.DataFrame, name: str) -> None:
        """Counted crossings per city, next to how many of them got each measurement."""
        data = summary.sort("crossings", descending=True)
        if data.height == 0:
            return
        fig = go.Figure([
            go.Bar(name="Crossings", x=data["city"].to_list(), y=data["crossings"].to_list()),
            go.Bar(name="with speed", x=data["city"].to_list(), y=data["crossings_with_speed"].to_list()),
            go.Bar(name="with hesitation", x=data["city"].to_list(), y=data["crossings_with_hesitation"].to_list()),
        ])
        fig.update_layout(barmode="group", xaxis_title="City", yaxis_title="Crossings")
        io.save_plotly_figure(self._layout(fig), name)

    def box_per_city(self, crossings: pl.DataFrame, metric: str, name: str) -> None:
        """Box plot of a per-crossing metric by city, every crossing shown as a point."""
        data = crossings.filter(pl.col(metric).is_not_null())
        if data.height == 0:
            return
        fig = px.box(data.to_dict(as_series=False), x="city", y=metric, points="all",
                     hover_data=["video", "track_id"],
                     labels={"city": "City", metric: METRIC_LABELS.get(metric, metric)})
        io.save_plotly_figure(self._layout(fig), name)

    def scatter_speed_hesitation(self, summary: pl.DataFrame, name: str) -> None:
        """Mean hesitation time against mean crossing speed, one point per city."""
        data = summary.filter(pl.col("speed_index_mean").is_not_null() & pl.col("hesitation_mean_s").is_not_null())
        if data.height == 0:
            return
        fig = px.scatter(data.to_dict(as_series=False), x="speed_index_mean", y="hesitation_mean_s", text="city",
                         size="crossings",
                         labels={"speed_index_mean": METRIC_LABELS["speed_index"],
                                 "hesitation_mean_s": METRIC_LABELS["hesitation_s"]})
        fig.update_traces(textposition="top center")
        io.save_plotly_figure(self._layout(fig), name)

    def bbox_vs_segmentation_speed(self, crossings: pl.DataFrame, name: str) -> None:
        """Per-crossing speed from the whole track against the on-road frames only."""
        data = crossings.filter(pl.col("speed_bbox_index").is_not_null() & pl.col("speed_seg_index").is_not_null())
        if data.height == 0:
            return
        fig = px.scatter(data.to_dict(as_series=False), x="speed_bbox_index", y="speed_seg_index", color="city",
                         hover_data=["video", "track_id"],
                         labels={"speed_bbox_index": METRIC_LABELS["speed_bbox_index"],
                                 "speed_seg_index": METRIC_LABELS["speed_seg_index"], "city": "City"})
        top = max(data["speed_bbox_index"].max(), data["speed_seg_index"].max()) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dot", color="grey"))
        io.save_plotly_figure(self._layout(fig), name)

    def speed_sources_per_city(self, speeds: pl.DataFrame, videos: pl.DataFrame, sources: dict,
                               name: str) -> go.Figure:
        """Tall bar chart of the median crossing speed per city, one bar per source (CROWD and each video model).

        Args:
            speeds: one row per source and city with columns source, city, median, q25, q75 and n (tracks).
            videos: generated videos per model and city (columns source, city, videos), shown in the city labels.
            sources: source -> (legend name, colour), in the order of the bars.
            name: file name of the figure.
        """
        crowd = speeds.filter(pl.col("source") == "crowd").sort("median", descending=False, nulls_last=False)
        cities = crowd["city"].to_list()
        others = pl.concat([speeds["city"], videos["city"]]).unique().sort().to_list()
        cities = [c for c in others if c not in cities] + cities
        counts = {(r["source"], r["city"]): r["videos"] for r in videos.iter_rows(named=True)}
        generated = [s for s in sources if s != "crowd"]
        labels = [f"{c} ({'/'.join(str(counts.get((s, c), 0)) for s in generated)})" for c in cities]
        fig = go.Figure()
        # Horizontal groups stack their bars bottom-up: add the sources in reverse so CROWD is on top.
        for source, (legend, colour) in reversed(list(sources.items())):
            rows = {r["city"]: r for r in speeds.filter(pl.col("source") == source).iter_rows(named=True)}
            median = [rows.get(c, {}).get("median") for c in cities]
            fig.add_trace(go.Bar(
                name=legend, y=labels, x=median, orientation="h", marker_color=colour,
                error_x=dict(type="data", symmetric=False, thickness=1, width=0,
                             array=[(rows[c]["q75"] - rows[c]["median"]) if rows.get(c, {}).get("q75") is not None
                                    else None for c in cities],
                             arrayminus=[(rows[c]["median"] - rows[c]["q25"]) if rows.get(c, {}).get("q25")
                                         is not None else None for c in cities]),
                customdata=[[rows.get(c, {}).get("n"), counts.get((source, c), "")] for c in cities],
                hovertemplate=(f"{legend}<br>%{{y}}<br>median %{{x:.2f}} m/s<br>%{{customdata[0]}} tracks"
                               + ("" if source == "crowd" else ", %{customdata[1]} videos") + "<extra></extra>"),
            ))
        fig.update_layout(
            barmode="group", bargap=0.25, bargroupgap=0.05, height=150 + 24 * len(cities),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, traceorder="reversed"),
            xaxis_title="Median crossing speed (m/s; whiskers: 25th-75th percentile)",
            yaxis_title="City (videos: " + " / ".join(sources[s][0] for s in generated) + ")",
        )
        fig.update_yaxes(tickfont=dict(size=11), automargin=True)
        fig.update_xaxes(rangemode="tozero")
        self._layout(fig).update_layout(margin=dict(l=20, r=20, t=40, b=60))
        io.save_plotly_figure(fig, name, width=1000, height=150 + 24 * len(cities))
        return fig

    def plot_all(self, crossings: pl.DataFrame, summary: pl.DataFrame) -> None:
        """Write every figure to <output>/figures."""
        self.crossings_per_city(summary, "crossings_city")
        self.bar_mean_per_city(summary, "speed_index_mean", "speed_index_std", "crossings_with_speed",
                               METRIC_LABELS["speed_index"], "speed_crossing_city", reference=1.0)
        self.bar_mean_per_city(summary, "hesitation_mean_s", "hesitation_std_s", "crossings_with_hesitation",
                               METRIC_LABELS["hesitation_s"], "hesitation_time_city")
        for metric in ("speed_index", "hesitation_s", "hesitation_bbox_s", "speed_bbox_index", "crossing_time_s"):
            self.box_per_city(crossings, metric, f"box_{metric}")
        self.scatter_speed_hesitation(summary, "scatter_speed-hesitation")
        self.bbox_vs_segmentation_speed(crossings, "scatter_speed_bbox-segmentation")
