"""Plotly figures of crossing speed and hesitation time per city."""

import base64
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

import common
from utils.plotting.io import IO

io = IO()

# Font of city labels drawn next to flag images, and files with its metrics (Liberation Sans matches Arial's widths), so
# that a flag can be placed just before a right-aligned name; DejaVu Sans is close enough when neither is installed.
LABEL_FONT = "Arial"
LABEL_FONT_FILES = ("/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def text_width(text: str, size: int) -> float:
    """Width in pixels of text in LABEL_FONT at size (px); about 0.55 em per character without a font file."""
    path = next((f for f in LABEL_FONT_FILES if os.path.isfile(f)), None)
    if path is None:
        return 0.55 * size * len(text)
    from PIL import ImageFont
    return ImageFont.truetype(path, size).getlength(text)


def data_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

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
                               name: str, labels: dict = None) -> go.Figure:
        """Tall bar chart of the median crossing speed per city, one bar per source (CROWD and each video model).

        Args:
            speeds: one row per source and city with columns source, city, median, q25, q75 and n (tracks).
            videos: generated videos per model and city (columns source, city, videos), shown in the city labels.
            sources: source -> (legend name, colour), in the order of the bars.
            name: file name of the figure.
            labels: city -> label to show (e.g. with flag and state; places.city_labels), the city name otherwise.
        """
        labels = labels or {}
        crowd = speeds.filter(pl.col("source") == "crowd").sort("median", descending=False, nulls_last=False)
        cities = crowd["city"].to_list()
        others = pl.concat([speeds["city"], videos["city"]]).unique().sort().to_list()
        cities = [c for c in others if c not in cities] + cities
        counts = {(r["source"], r["city"]): r["videos"] for r in videos.iter_rows(named=True)}
        generated = [s for s in sources if s != "crowd"]
        names = [f"{labels.get(c, c)} ({'/'.join(str(counts.get((s, c), 0)) for s in generated)})" for c in cities]
        fig = go.Figure()
        # Horizontal groups stack their bars bottom-up: add the sources in reverse so CROWD is on top.
        for source, (legend, colour) in reversed(list(sources.items())):
            rows = {r["city"]: r for r in speeds.filter(pl.col("source") == source).iter_rows(named=True)}
            median = [rows.get(c, {}).get("median") for c in cities]
            fig.add_trace(go.Bar(
                name=legend, y=names, x=median, orientation="h", marker_color=colour,
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

    def speed_sources_page(self, speeds: pl.DataFrame, sources: dict, name: str, width: int = 780,
                           height: int = 1010, columns: int = 2, labels: dict = None,
                           flags: dict = None) -> go.Figure:
        """One-page dot plot of the median crossing speed per city: one dot (with 25th-75th percentile whisker) per
        source, cities sorted by the CROWD value and split over side-by-side panels so that 100 cities fit a page.

        Args:
            speeds: one row per source and city with columns source, city, median, q25, q75.
            sources: source -> (legend name, colour), in the order of the dots within a city.
            name: file name of the figure.
            width, height: size in pixels, chosen for the page's text area.
            columns: number of panels.
            labels: city -> label to show (e.g. with flag and state; places.city_labels), the city name otherwise.
            flags: city -> flag image (PNG; places.city_flag_images), drawn before its label, which is then drawn as
                an annotation in LABEL_FONT instead of a tick label.
        """
        labels, flags = labels or {}, flags or {}
        size, flag_px, pad = 10, 13, 4  # label font size, flag side and gaps in px
        crowd = speeds.filter(pl.col("source") == "crowd").sort("median", descending=True, nulls_last=True)
        cities = crowd["city"].to_list()
        cities += sorted(c for c in speeds["city"].unique().to_list() if c not in cities)
        per_column = -(-len(cities) // columns)
        # The gap between panels holds the next panel's city labels: about 6 px per character at size 10.
        if flags:
            label_px = max((text_width(labels.get(c, c), size) for c in cities), default=60) + flag_px + 3 * pad + 4
        else:
            label_px = 6.2 * max((len(labels.get(c, c)) for c in cities), default=10) + 24
        margin = dict(l=label_px if flags else 10, r=10, t=30, b=40)
        fig = make_subplots(rows=1, cols=columns, horizontal_spacing=label_px / width, shared_xaxes=True)
        plot_px = width - margin["l"] - margin["r"]
        images = {}
        offsets = {s: 0.27 * (i - (len(sources) - 1) / 2) for i, s in enumerate(sources)}
        values = speeds.drop_nulls("median")
        for column in range(columns):
            chunk = cities[column * per_column:(column + 1) * per_column]
            position = {c: i for i, c in enumerate(chunk)}
            for source, (legend, colour) in sources.items():
                rows = values.filter((pl.col("source") == source) & pl.col("city").is_in(chunk))
                if rows.height == 0:
                    continue
                y = [position[c] + offsets[source] for c in rows["city"]]
                fig.add_trace(go.Scatter(
                    x=rows["median"], y=y, mode="markers", name=legend, legendgroup=source, showlegend=column == 0,
                    marker=dict(color=colour, size=5, line=dict(color="white", width=0.5)),
                    error_x=dict(type="data", symmetric=False, thickness=0.8, width=0, color=colour,
                                 array=(rows["q75"] - rows["median"]).to_list(),
                                 arrayminus=(rows["median"] - rows["q25"]).to_list()),
                    customdata=rows["city"],
                    hovertemplate=f"{legend}<br>%{{customdata}}: %{{x:.2f}} m/s<extra></extra>"),
                    row=1, col=column + 1)
            fig.update_yaxes(tickvals=list(range(len(chunk))), ticktext=[labels.get(c, c) for c in chunk],
                             range=[len(chunk) - 0.5, -0.5], showticklabels=not flags,
                             tickfont=dict(size=size), showgrid=False, row=1, col=column + 1)
            if flags:  # name right-aligned at the axis, its flag just before it (as in the frame grid)
                axis = "" if column == 0 else str(column + 1)
                domain = fig.layout[f"xaxis{axis}"].domain
                panel_px = (domain[1] - domain[0]) * plot_px
                for i, city in enumerate(chunk):
                    label = labels.get(city, city)
                    fig.add_annotation(text=label, x=0, y=i, xref=f"x{axis} domain", yref=f"y{axis}",
                                       xanchor="right", xshift=-pad, showarrow=False, borderpad=0,
                                       font=dict(family=LABEL_FONT, size=size, color="#222"))
                    if flags.get(city):
                        images.setdefault(flags[city], data_uri(flags[city]))
                        left = pad + text_width(label, size) + pad + flag_px
                        fig.add_layout_image(source=images[flags[city]], x=-left / panel_px, y=i,
                                             xref=f"x{axis} domain", yref=f"y{axis}", sizex=flag_px / panel_px,
                                             sizey=0.8, xanchor="left", yanchor="middle", sizing="contain",
                                             layer="above")
            for i in range(0, len(chunk), 2):  # light bands on every other city, to follow a row across the panel
                fig.add_hrect(y0=i - 0.5, y1=i + 0.5, fillcolor="#f2f1ed", line_width=0, layer="below",
                              row=1, col=column + 1)
        low = values["q25"].drop_nulls().min()
        high = values["q75"].drop_nulls().max()
        fig.update_xaxes(range=[max(0, low - 0.05), high + 0.05], tickfont=dict(size=10), title_font=dict(size=11),
                         title_text="Median crossing speed (m/s)", gridcolor="#e4e3df")
        self._layout(fig).update_layout(
            width=width, height=height, margin=margin, font=dict(size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)))
        io.save_plotly_figure(fig, name, width=width, height=height)
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
