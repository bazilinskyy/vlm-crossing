# Pedestrian crossing speed and hesitation time from dashcam videos

Detects and tracks road users in dashcam videos with YOLOv11x and BoT-SORT, then
measures, for every pedestrian who crosses the road in front of the car:

- **crossing speed**: crowd-city's relative motion index (see below)
- **hesitation time** (s): how long the pedestrian stood still before stepping onto the road
- **crossing time** (s): how long the crossing was observed

Tracking follows [crowd-dataset/crowd](https://github.com/crowd-dataset/crowd). The analysis is
the crowd-city code itself ([crowd-dataset/crowd-city](https://github.com/crowd-dataset/crowd-city)):
`crowd_city/` holds an unchanged copy of its crossing detection (including track joining), metrics and road-surface
segmentation modules, at the commit of the crowd-city run that produces CROWD's current per-city values (Shadab's
incremental run on lin5080, `crowd_city/SOURCE_COMMIT`; refresh with `sync_crowd_city.py`), and `analysis.py` calls
them in the same order as crowd-city's own `analysis.py`. With the Waymo speed model, a crossing's m/s speed is
computed by the same function as CROWD's per-city table (`analyse_crowd_sources`: whole track, kept when the model
rates it valid), as in the city paper (Alam, Ebel, Martens and Bazilinskyy, *Beyond National Averages*, under review).

## Reproducing the paper

Every number and figure of the paper can be rebuilt from this repository alone, without crowd-city, a GPU or the
video files:

```bash
uv sync
./reproduce.sh          # figures to paper_figures/, numbers to _output/
```

| In the paper | Script | Reads |
|---|---|---|
| Table 1: videos per city | (counts of `videos/generated_videos.csv`) | `videos/generated_videos.csv` |
| Fig. 1: one frame per city | `city_frames.py` | the videos (4TU.ResearchData), `results/minimax_h3/city_frames.csv` |
| Appearance (CLIP) results | `appearance_analysis.py --from-results` | `results/appearance/frames.csv` |
| Table 2, usable crossings, crossing speed and initiation time | `generated_speeds.py`, `speed_statistics.py` | `results/<set>/` |
| Fig. 2: speed distributions and paired differences | `speed_statistics.py` | `results/` |
| Fig. 3: speed per city | `speed_generated_vs_crowd.py` | `crowd/`, `_output/generated_speed_by_city.csv` |
| Differences between cities, reliability, resolution check | `speed_statistics.py` (`--scaled`) | `results/`, `crowd/` |

`results/<set>/` holds the crossings that `analysis.py` counted in each set of clips, with `crowd_city/` (crowd-city's
crossing detection and speed code at the commit of CROWD's current run, `crowd_city/SOURCE_COMMIT`) and the Waymo
speed model in `models/`. `crowd/` is refreshed nightly; the commit tagged for the paper holds the CROWD values its
numbers come from. Redoing the tracking itself needs the videos and a GPU: fetch the clips (or download them from
4TU.ResearchData into `videos/`), cut the real continuations from the CROWD videos with `paired_real_clips.py`, and run
`run_sets.sh` (see [Generated videos and the comparison with CROWD](#generated-videos-and-the-comparison-with-crowd)).

## Data layout

```
data/
├── Amsterdam/
│   ├── drive_01.mp4
│   └── drive_02.mp4
└── Delhi/
    └── morning.mov
```

Every sub-folder of `data/` is a city, and every video in it (sub-folders included) is analysed.

## Setup

```bash
uv sync
cp default.config config
```

`config` must contain every key of `default.config` (the run stops otherwise). `default.secret` → `secret` is only needed for email notifications. Segmentation needs `ffmpeg`/`ffprobe` on the PATH; the SegFormer model is downloaded from Hugging Face on first use.

## Running

1. **Tracking** writes one CSV per video to `_output/bbox/<city>/<video>_<fps>.csv`, a `<video>_<fps>.json` next to it with the tracking settings used, and `_output/videos.csv` (city, fps, size and duration of every video). As in crowd, YOLO runs at confidence 0.0 so the tracker sees every box; `min_confidence` is applied when the analysis reads the CSV. Videos that already have a CSV are skipped unless `always_analyse` is true, and an interrupted video is redone on the next run. If the tracking settings differ from those stored with a CSV, the run warns but keeps the CSV: delete it (or set `always_analyse`) to track it again.

   ```bash
   uv run python main.py
   ```

   The first run downloads `yolo11x.pt` through ultralytics. The device is picked automatically (CUDA, Apple MPS, then CPU).

2. **Analysis** writes `_output/crossings.parquet` and `_output/crossings.csv` (one row per crossing), `_output/city_summary.csv` (one row per city) and plotly figures in `_output/figures/`.

   ```bash
   uv run python analysis.py
   ```

   Add `--force` to ignore the cache and analyse every video again.

### What is rerun

| Change | Rerun |
|---|---|
| New video in `data/` | Tracking and analysis of that video only |
| Nothing | Nothing (cached results are combined again) |
| A tracking CSV or video is rewritten | Analysis of that video |
| Code in `crowd_city/` or `analysis.py`, or a setting that changes results (`min_confidence`, road strip, `check_per_sec_time`, `crossing_rule`, segmentation model and sampling) | Analysis of every video; SegFormer labels are reused from the store when the crossing tracks and segmentation settings are unchanged |
| Comments, docstrings or formatting in that code | Nothing |
| Speed or waiting-time limits, `segmentation_is_primary`, plots, summary | Nothing (always rebuilt from the cached values) |

Per-video results are cached in `_output/analysis/<city>/<video>_<fps>.json` with a key built from their inputs (see `utils/cache.py`), and crowd-city's surface-label store is kept in `_output/segmentation/`.

### Generated videos and the comparison with CROWD

Everything the comparison needs, apart from the video files, is in the repository, so a fresh checkout runs it:

```
videos/
├── generated_videos.csv          every video: model, city, status, batch, place in Drive, CROWD source frame, size, fps
├── veo3/<city>/*.mp4             Veo 3.1 Fast clips analysed (109)              ┐ not in git: on 4TU.ResearchData,
├── minimax_h3/<city>/*.mp4       MiniMax H3 clips analysed (797)                │ or fetched from Drive with
└── not_analysed/                 left-out clips, Veo prompt tests, LTX benchmark ┘ fetch_generated_videos.py
crowd/
├── crowd_crossing_speed_by_city.csv   CROWD's per-city values from crowd-city (refreshed nightly, see below)
├── cities.csv                    the 100 cities of CROWD's mapping.csv, with country and coordinates
└── city_footage.csv              YouTube videos and hours CROWD found per city, population, economic output
                                  (city_footage.py, from CROWD's mapping.csv)
assets/flags/                     Twemoji flag images (CC-BY 4.0, github.com/jdecked/twemoji) for the city labels of
                                  the figures; places.py fetches a missing one
models/waymo_calibration_v32/     crowd-city's frozen Waymo speed model (the default speed_model: speeds in m/s)
results/<set>/                    crossings.csv and videos.csv of each tracked set; real_minimax/clips.csv lists the
                                  real continuations with their CROWD video and times
```

The videos come from the project's Google Drive folders (Veo 3 by Tanvir, MiniMax H3 by Zi). `fetch_generated_videos.py` mirrors the folders to `data/_drive/` (not committed), links every video into `videos/` and rewrites `videos/generated_videos.csv` (the clips themselves stay out of git and will be published on 4TU.ResearchData); rerun it for new videos. It needs `ffprobe` on the PATH. Drive refuses anonymous downloads after a few hundred files; then download the folder in a signed-in browser and pass the zip(s) with `--zip <folder>=<path pattern>` (see the script's docstring).

```bash
uvx --with gdown python fetch_generated_videos.py
```

**Tracking** (GPU): `run_sets.sh` writes `config` for each set (`videos/veo3`, `videos/minimax_h3`; `data/real_minimax`, the real 8 seconds after each MiniMax H3 reference frame, which `paired_real_clips.py --videos '<CROWD video folders>'` cuts from the CROWD videos, as YouTube footage is not redistributed here), runs `main.py` and `analysis.py` into `_output/<set>/`, and keeps the results in `results/<set>/` (`export_results.py`). The run of 3 November 2026 was on lin4080 (RTX 4080).

```bash
SETS="veo3 minimax_h3" ./run_sets.sh
```

**Comparison** (CPU, from `results/` and `crowd/`):

```bash
uv run python generated_speeds.py --readme          # per-city speeds, usability, correlations, paired clips
uv run python speed_statistics.py --readme          # tests, bootstrap intervals, reliability of the city medians
uv run python speed_generated_vs_crowd.py --readme  # speed per city: CROWD, Veo 3.1 Fast, MiniMax H3
uv run python city_frames.py --readme               # one frame per city and the table of cities below
uv run python appearance_analysis.py --readme       # CLIP: does a clip keep the look of its city? and the map
```

`generated_speeds.py` writes the per-city speeds (`_output/generated_speed_by_city.csv`, the `--generated` input of `speed_generated_vs_crowd.py`), the share of clips with a counted crossing, the rank correlations of the city medians with CROWD and with the real continuations, a per-clip table of the MiniMax/real pairs, and two scatter plots. `speed_statistics.py` runs the paper's tests (bootstrap intervals, McNemar, Wilcoxon, Kruskal-Wallis, split-half reliability of the city medians; whether cities with more YouTube footage, people or economic output, or in English-speaking countries, were generated closer to CROWD's speed, a proxy for their share of the training data; `--scaled results/real_minimax_480` adds the real continuations scaled to 864×480, a check that the speed difference is not a matter of resolution). `city_frames.py` crops each city's frame around the crossing pedestrian when main.py's tracks are in `_output/minimax_h3/bbox/`, and takes the middle frame of a clip otherwise. `appearance_analysis.py` needs `open_clip_torch` besides this project's packages; its embeddings are cached in `_output/appearance/`. Each script takes `--pdf-dir` (or `--pdf`) to write the paper's figures.

**Nightly** (mini, 05:30): `nightly.sh` copies crowd-city's current `crowd_crossing_speed_by_city.csv` (the newest on lin5080: each finished step of Shadab's crowd-city run leaves one in `~/crowd_city_runs/<step>/_output/crossing_speed/`) to `crowd/` when it changed, reruns the comparison and pushes `crowd/`, `results/`, `readme/`, `assets/` and README.md. The README's prose numbers are not rewritten; the figures and the table of cities are.

## Method

All of this is crowd-city's code; see its README for the details.

1. **Detections** below `min_confidence` (0.7) and boxes without a track id are dropped.
2. **Crossing candidates.** With `crossing_rule: "road_crossing"` (rule D, crowd-city's default), a pedestrian counts as crossing when SegFormer-B2 (Cityscapes) puts their feet on the road, the track moves across at least 14% of the image width while there, and the box size changes slowly (rejecting people walking along the road). With `"detector"`, CROWD's original detector is used instead: the track passes through the central strip `[boundary_left, boundary_right]` and survives its geometric, rider and camera-motion filters.
3. **Road entry and exit.** The road surface under each crossing pedestrian's feet is read at 1 Hz, then at 4 Hz around the changes, giving the frames where they step onto and off the road.
4. **Hesitation time** is the stationary time ending at road entry: checks `check_per_sec_time` times per second, still when the box centre moved at most 10% of the box height, at least three still checks in a row. It is missing when the pedestrian did not stand still, or was already on the road when first seen (`hesitation_unobservable`).
5. **Speed.** crowd-city converts box motion to m/s with a model calibrated on Waymo; that model is not used here, and crowd-city then reports its **relative motion index** instead: a pedestrian's camera-motion-compensated speed divided by the median of the comparable pedestrians in the same video (1.0 = typical for that video). It needs at least three comparable tracks in the video (`too_few_comparable_tracks` otherwise). The road-surface speed uses only the frames on the road.

Both derivations are kept in `crossings.csv`: `*_bbox_*` from the whole bounding-box track and `*_seg_*` from the road surface. `speed_index` and `hesitation_s` hold the reported one, chosen by `segmentation_is_primary` (true, as in crowd-city's configuration).

### Limitations

- The speed index is relative within a video, so it compares pedestrians in the same video, not cities: with one video per city each city is centred near 1.0 by construction.
- Short clips rarely show the wait: a pedestrian already on the road in the first frame has no observable hesitation time.
- `crowd_city/core/` replaces crowd-city's YouTube `mapping.csv` lookups, which these videos are not in. The video fps comes from the detection file name, and stature correction stays off (crowd-city's default).

## Results

Run on the 20 clips in `data/` (one 8 s, 1920×1080, 24 fps dashcam clip per city) with crowd-city `26785f6`, the road-crossing rule and road-surface values reported (the settings in `default.config`). SegFormer selected **47 crossings** in 17 cities out of 72 candidates; 35 have a speed index and 2 a hesitation time before road entry. The bounding-box hesitation time is shown alongside because the road-surface one is rarely observable in such short clips.

| City | Crossings | Speed index (n) | Hesitation before road entry, s (n) | Hesitation, boxes, s (n) |
|---|---|---|---|---|
| Moscow | 7 | 1.04 (7) | – (0) | 1.0 (1) |
| Bangkok | 6 | 1.10 (5) | – (0) | 4.1 (4) |
| Cotabato | 5 | 1.09 (5) | – (0) | 3.0 (4) |
| Montreal | 5 | 0.78 (4) | – (0) | 4.0 (4) |
| Toronto | 4 | 0.36 (1) | – (0) | 4.9 (3) |
| Berlin | 3 | 1.71 (3) | 1.3 (1) | 3.3 (3) |
| Philadelphia | 3 | 1.20 (3) | – (0) | 1.0 (1) |
| Warsaw | 3 | 0.55 (3) | 2.0 (1) | 4.6 (3) |
| BuenosAires | 2 | 1.02 (2) | – (0) | – (0) |
| Kyiv | 2 | – (0) | – (0) | 1.2 (2) |
| Detroit | 1 | – (0) | – (0) | 1.6 (1) |
| Dubai | 1 | – (0) | – (0) | 4.7 (1) |
| Guangzhou | 1 | – (0) | – (0) | 2.7 (1) |
| Helsinki | 1 | 0.40 (1) | – (0) | 2.3 (1) |
| Paris | 1 | 0.32 (1) | – (0) | 3.3 (1) |
| Townsville | 1 | – (0) | – (0) | 3.0 (1) |
| Yangon | 1 | – (0) | – (0) | 3.7 (1) |
| Copenhagen | 0 | – (0) | – (0) | – (0) |
| Melbourne | 0 | – (0) | – (0) | – (0) |
| SaoPaulo | 0 | – (0) | – (0) | – (0) |

How to read these numbers:

- **Speed index** compares a pedestrian with the others in the same clip, so with one clip per city each city is centred near 1.0 by construction; it does not rank cities. A speed index needs at least three comparable pedestrians in the clip, which is why 12 crossings have none.
- **Hesitation before road entry** is missing for 45 of the 47 crossings because the pedestrian was already on the road when first seen: 38 crossings start in the first half-second of the clip.
- **Hesitation from the boxes** is crowd-city's bounding-box initiation time: the first still period anywhere in the track, which can include standing in the road.

### Figures

Click a figure to open its interactive version. The interactive figures are displayed using `htmlpreview.github.io`, which may be blocked on some networks; the `.html` files are also in `readme/` and can be opened locally. `analysis.py` writes every figure to `_output/figures/`.

**Crossings per city**, and how many of them got a speed index and a hesitation time:

[![Crossings per city](readme/crossings_city.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/crossings_city.html)

**Mean speed index per city** (error bars: one standard deviation; dotted line: the video median, 1.0):

[![Speed index per city](readme/speed_crossing_city.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/speed_crossing_city.html)

**Speed index of every crossing:**

[![Speed index per crossing](readme/box_speed_index.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/box_speed_index.html)

**Bounding-box hesitation time of every crossing:**

[![Bounding-box hesitation time](readme/box_hesitation_bbox_s.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/box_hesitation_bbox_s.html)

**Speed from the whole track against the on-road frames only**, one point per crossing (dotted line: equal values). Most crossings agree; the road-restricted speed is lower for some slow crossings.

[![Bounding-box against road-surface speed](readme/scatter_speed_bbox-segmentation.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/scatter_speed_bbox-segmentation.html)

**Observed crossing time:** capped at the 8 s clip length, since many pedestrians are visible for the whole clip.

[![Crossing time](readme/box_crossing_time_s.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/vlm_crossing/blob/main/readme/box_crossing_time_s.html)

The figures in `readme/` are refreshed from the latest results with `uv run python readme_figures.py`.

## Generated videos and CROWD

The paper asks whether video generators reproduce the crossing behaviour of a city or only its look. Frames of real crossings from CROWD were continued as 8-second clips by two image-to-video models with one prompt that does not name the place, and the same pipeline measures crossing speed in the generated clips and in CROWD. Numbers below are as of 3 November 2026; generation with MiniMax H3 continues towards 100 cities.

| Model | Videos | Cities | Resolution | Frame rate | Not used |
|---|---|---|---|---|---|
| Veo 3.1 Fast (Tanvir) | 109 | 30 (10 with 5-11 videos, 20 with one) | 1920×1080 | 24 fps | 16 prompt tests |
| MiniMax H3 (Zi, 6 steps on an RTX 4090) | 797 | 77 in 42 countries (9-15 each) | 864×480 | 24 fps | 56 clips Zi left out; 100 settings benchmark clips (MiniMax H3 and LTX 2.3) |

All 30 Veo cities also have MiniMax videos. CROWD's reference values come from crowd-city's run over the 100 cities with the most footage (cars only, footage in a fixed random order until 100 crossings or 100 hours per city); Seattle has no crossing in it.

**One frame per city** (`city_frames.py`: in one MiniMax H3 clip, the frame halfway through the crossing with the largest pedestrian, cropped around the pedestrian; grey tiles are cities without generated clips yet):

[![One frame per city](readme/frames_cities.jpg)](readme/frames_cities.jpg)

<details>
<summary><b>The cities one by one</b>: clips and crossings, median crossing speeds (CROWD and the generated clips, cities with at least one m/s speed), and the frame above at full size</summary>

<!-- city table -->
| City | Country | Veo 3.1 Fast clips | MiniMax H3 clips | MiniMax H3 crossings | CROWD speed (m/s) | Veo 3.1 Fast speed | MiniMax H3 speed | Frame |
|---|---|--:|--:|--:|--:|--:|--:|---|
| 🇸🇾 Aleppo | Syria |  |  |  | 1.25 |  |  |  |
| 🇯🇴 Amman | Jordan |  |  |  | 1.40 |  |  |  |
| 🇳🇱 Amsterdam | Netherlands |  | 10 | 11 | 1.30 |  | 1.30 | <img src="readme/frames/Amsterdam.jpg" width="240" alt="Amsterdam"> |
| 🇺🇸 Anchorage, AK | United States |  |  |  | 1.38 |  |  |  |
| 🇺🇸 Atlanta, GA | United States |  | 10 | 11 | 1.25 |  | 1.29 | <img src="readme/frames/Atlanta.jpg" width="240" alt="Atlanta"> |
| 🇳🇿 Auckland | New Zealand |  |  |  | 1.29 |  |  |  |
| 🇺🇸 Baltimore, MD | United States |  | 10 | 15 | 1.35 |  | 1.24 | <img src="readme/frames/Baltimore.jpg" width="240" alt="Baltimore"> |
| 🇹🇭 Bangkok | Thailand | 1 | 10 | 19 | 1.29 |  | 1.19 | <img src="readme/frames/Bangkok.jpg" width="240" alt="Bangkok"> |
| 🇨🇳 Beijing | China |  |  |  | 1.44 |  |  |  |
| 🇩🇪 Berlin | Germany | 1 | 10 | 16 | 1.32 |  | 1.35 | <img src="readme/frames/Berlin.jpg" width="240" alt="Berlin"> |
| 🇺🇸 Boston, MA | United States |  |  |  | 1.36 |  |  |  |
| 🇦🇺 Brisbane, QLD | Australia |  | 10 | 42 | 1.31 |  | 1.26 | <img src="readme/frames/Brisbane.jpg" width="240" alt="Brisbane"> |
| 🇷🇴 Bucharest | Romania |  | 10 | 5 | 1.24 |  | 1.09 | <img src="readme/frames/Bucharest.jpg" width="240" alt="Bucharest"> |
| 🇭🇺 Budapest | Hungary |  | 10 | 13 | 1.31 |  | 1.24 | <img src="readme/frames/Budapest.jpg" width="240" alt="Budapest"> |
| 🇦🇷 Buenos Aires | Argentina | 1 | 10 | 9 | 1.38 | 1.28 | 1.48 | <img src="readme/frames/Buenos_Aires.jpg" width="240" alt="Buenos Aires"> |
| 🇰🇷 Busan | South Korea |  | 10 | 13 | 1.39 |  | 1.19 | <img src="readme/frames/Busan.jpg" width="240" alt="Busan"> |
| 🇨🇦 Calgary, AB | Canada |  |  |  | 1.35 |  |  |  |
| 🇿🇦 Cape Town | South Africa |  | 10 | 10 | 1.44 |  | 1.27 | <img src="readme/frames/Cape_Town.jpg" width="240" alt="Cape Town"> |
| 🇮🇹 Catania | Italy |  |  |  | 1.25 |  |  |  |
| 🇺🇸 Chicago, IL | United States | 11 | 10 | 25 | 1.34 | 1.32 | 1.24 | <img src="readme/frames/Chicago.jpg" width="240" alt="Chicago"> |
| 🇵🇭 Cotabato | Philippines | 1 | 10 | 8 | 1.25 | 1.06 | 1.28 | <img src="readme/frames/Cotabato.jpg" width="240" alt="Cotabato"> |
| 🇺🇸 Dallas, TX | United States |  | 10 | 10 | 1.38 |  | 1.24 | <img src="readme/frames/Dallas.jpg" width="240" alt="Dallas"> |
| 🇸🇾 Damascus | Syria |  | 10 | 4 | 1.29 |  | 1.24 | <img src="readme/frames/Damascus.jpg" width="240" alt="Damascus"> |
| 🇦🇺 Darwin, NT | Australia |  | 10 | 4 | 1.36 |  | 1.16 | <img src="readme/frames/Darwin.jpg" width="240" alt="Darwin"> |
| 🇳🇱 Den Haag | Netherlands |  | 10 | 7 | 1.35 |  | 1.21 | <img src="readme/frames/Den_Haag.jpg" width="240" alt="Den Haag"> |
| 🇮🇩 Denpasar | Indonesia |  |  |  | 1.25 |  |  |  |
| 🇺🇸 Denver, CO | United States |  |  |  | 1.32 |  |  |  |
| 🇺🇸 Detroit, MI | United States | 1 | 12 | 20 | 1.35 |  | 1.33 | <img src="readme/frames/Detroit.jpg" width="240" alt="Detroit"> |
| 🇦🇪 Dubai | United Arab Emirates | 1 | 11 | 5 | 1.29 | 1.02 | 1.12 | <img src="readme/frames/Dubai.jpg" width="240" alt="Dubai"> |
| 🇮🇪 Dublin | Ireland |  | 10 | 11 | 1.46 |  | 1.37 | <img src="readme/frames/Dublin.jpg" width="240" alt="Dublin"> |
| 🇳🇱 Eindhoven | Netherlands |  | 11 | 8 | 1.35 |  | 1.14 | <img src="readme/frames/Eindhoven.jpg" width="240" alt="Eindhoven"> |
| 🇯🇵 Gamagori | Japan |  | 10 | 7 | 1.29 |  | 1.14 | <img src="readme/frames/Gamagori.jpg" width="240" alt="Gamagori"> |
| 🇨🇳 Guangzhou | China | 1 | 12 | 12 | 1.33 | 0.98 | 1.21 | <img src="readme/frames/Guangzhou.jpg" width="240" alt="Guangzhou"> |
| 🇻🇳 Hanoi | Vietnam | 5 | 10 | 8 | 1.16 | 1.09 | 1.16 | <img src="readme/frames/Hanoi.jpg" width="240" alt="Hanoi"> |
| 🇫🇮 Helsinki | Finland | 1 | 10 | 7 | 1.30 |  | 1.34 | <img src="readme/frames/Helsinki.jpg" width="240" alt="Helsinki"> |
| 🇻🇳 Ho Chi Minh City | Vietnam |  |  |  | 1.19 |  |  |  |
| 🇭🇰 Hong Kong | Hong Kong |  |  |  | 1.25 |  |  |  |
| 🇺🇸 Houston, TX | United States |  | 15 | 23 | 1.37 |  | 1.14 | <img src="readme/frames/Houston.jpg" width="240" alt="Houston"> |
| 🇹🇷 Istanbul | Türkiye |  | 10 | 11 | 1.26 |  | 1.45 | <img src="readme/frames/Istanbul.jpg" width="240" alt="Istanbul"> |
| 🇮🇩 Jakarta | Indonesia | 5 | 10 | 10 | 1.29 | 1.00 | 1.13 | <img src="readme/frames/Jakarta.jpg" width="240" alt="Jakarta"> |
| 🇵🇰 Karachi | Pakistan |  |  |  | 1.24 |  |  |  |
| 🇮🇷 Kermanshah | Iran |  |  |  | 1.36 |  |  |  |
| 🇲🇾 Kuala Lumpur | Malaysia |  | 11 | 11 | 1.38 |  | 1.37 | <img src="readme/frames/Kuala_Lumpur.jpg" width="240" alt="Kuala Lumpur"> |
| 🇺🇦 Kyiv | Ukraine | 1 | 10 | 20 | 1.45 | 1.16 | 1.11 | <img src="readme/frames/Kyiv.jpg" width="240" alt="Kyiv"> |
| 🇩🇰 København | Denmark | 1 | 10 | 13 | 1.36 |  | 1.29 | <img src="readme/frames/Copenhagen.jpg" width="240" alt="København"> |
| 🇺🇸 Las Vegas, NV | United States |  | 10 | 12 | 1.35 |  | 1.16 | <img src="readme/frames/Las_Vegas.jpg" width="240" alt="Las Vegas"> |
| 🇬🇧 London | United Kingdom | 11 | 10 | 19 | 1.44 | 1.26 | 1.26 | <img src="readme/frames/London.jpg" width="240" alt="London"> |
| 🇺🇸 Los Angeles, CA | United States | 11 | 10 | 24 | 1.34 | 1.26 | 1.30 | <img src="readme/frames/Los_Angeles.jpg" width="240" alt="Los Angeles"> |
| 🇵🇭 Manila | Philippines | 5 | 10 | 13 | 1.45 | 1.13 | 1.38 | <img src="readme/frames/Manila.jpg" width="240" alt="Manila"> |
| 🇻🇪 Maracaibo | Venezuela |  |  |  | 1.23 |  |  |  |
| 🇦🇺 Melbourne, VIC | Australia | 1 | 10 | 27 | 1.37 | 1.02 | 1.24 | <img src="readme/frames/Melbourne.jpg" width="240" alt="Melbourne"> |
| 🇺🇸 Miami, FL | United States |  | 10 | 16 | 1.35 |  | 1.24 | <img src="readme/frames/Miami.jpg" width="240" alt="Miami"> |
| 🇮🇹 Milan | Italy |  |  |  | 1.36 |  |  |  |
| 🇧🇾 Minsk | Belarus |  | 10 | 12 | 1.39 |  | 1.19 | <img src="readme/frames/Minsk.jpg" width="240" alt="Minsk"> |
| 🇨🇦 Montreal, QC | Canada | 1 | 10 | 13 | 1.34 |  | 1.40 | <img src="readme/frames/Montreal.jpg" width="240" alt="Montreal"> |
| 🇷🇺 Moscow | Russia | 1 | 12 | 3 | 1.33 | 1.25 | 1.21 | <img src="readme/frames/Moscow.jpg" width="240" alt="Moscow"> |
| 🇮🇳 Mumbai | India |  | 10 | 3 | 1.37 |  | 1.63 | <img src="readme/frames/Mumbai.jpg" width="240" alt="Mumbai"> |
| 🇩🇪 München | Germany |  | 10 | 27 | 1.36 |  | 1.30 | <img src="readme/frames/München.jpg" width="240" alt="München"> |
| 🇮🇳 New Delhi | India |  |  |  | 1.37 |  |  |  |
| 🇺🇸 New York, NY | United States | 11 | 10 | 29 | 1.29 | 1.25 | 1.34 | <img src="readme/frames/New_York.jpg" width="240" alt="New York"> |
| 🇻🇳 Ninh Binh | Vietnam |  |  |  | 1.30 |  |  |  |
| 🇩🇪 Oldenburg | Germany |  | 9 | 7 | 1.33 |  | 1.15 | <img src="readme/frames/Oldenburg.jpg" width="240" alt="Oldenburg"> |
| 🇳🇴 Oslo | Norway |  | 10 | 16 | 1.33 |  | 1.28 | <img src="readme/frames/Oslo.jpg" width="240" alt="Oslo"> |
| 🇫🇷 Paris | France | 1 | 10 | 26 | 1.32 |  | 1.30 | <img src="readme/frames/Paris.jpg" width="240" alt="Paris"> |
| 🇦🇺 Perth, WA | Australia |  | 10 | 14 | 1.34 |  | 1.43 | <img src="readme/frames/Perth.jpg" width="240" alt="Perth"> |
| 🇺🇸 Philadelphia, PA | United States | 1 | 10 | 24 | 1.31 | 1.34 | 1.36 | <img src="readme/frames/Philadelphia.jpg" width="240" alt="Philadelphia"> |
| 🇰🇭 Phnom Penh | Cambodia |  | 10 | 8 | 1.30 |  | 1.19 | <img src="readme/frames/Phnom_Penh.jpg" width="240" alt="Phnom Penh"> |
| 🇺🇸 Phoenix, AZ | United States |  | 10 | 7 | 1.31 |  | 1.30 | <img src="readme/frames/Phoenix.jpg" width="240" alt="Phoenix"> |
| 🇺🇸 Portland, OR | United States |  | 10 | 10 | 1.37 |  | 1.32 | <img src="readme/frames/Portland.jpg" width="240" alt="Portland"> |
| 🇨🇿 Prague | Czechia |  | 10 | 22 | 1.27 |  | 1.51 | <img src="readme/frames/Prague.jpg" width="240" alt="Prague"> |
| 🇧🇷 Rio de Janeiro | Brazil |  | 11 | 19 | 1.29 |  | 1.29 | <img src="readme/frames/Rio_de_Janeiro.jpg" width="240" alt="Rio de Janeiro"> |
| 🇺🇸 Sacramento, CA | United States |  | 11 | 14 | 1.44 |  | 1.16 | <img src="readme/frames/Sacramento.jpg" width="240" alt="Sacramento"> |
| 🇷🇺 Saint Petersburg | Russia | 10 | 10 | 32 | 1.34 | 1.10 | 1.40 | <img src="readme/frames/Saint_Petersburg.jpg" width="240" alt="Saint Petersburg"> |
| 🇺🇸 San Diego, CA | United States |  |  |  | 1.40 |  |  |  |
| 🇺🇸 San Francisco, CA | United States |  | 10 | 14 | 1.28 |  | 1.35 | <img src="readme/frames/San_Francisco.jpg" width="240" alt="San Francisco"> |
| 🇺🇸 San Jose, CA | United States |  | 10 | 8 | 1.36 |  | 1.33 | <img src="readme/frames/San_Jose.jpg" width="240" alt="San Jose"> |
| 🇧🇷 Sao Paulo | Brazil | 1 | 10 | 16 | 1.42 |  | 1.41 | <img src="readme/frames/Sao_Paulo.jpg" width="240" alt="Sao Paulo"> |
| 🇺🇸 Seattle, WA | United States |  | 10 | 24 |  |  | 1.27 | <img src="readme/frames/Seattle.jpg" width="240" alt="Seattle"> |
| 🇮🇩 Semarang | Indonesia |  |  |  | 1.30 |  |  |  |
| 🇰🇷 Seoul | South Korea | 10 | 10 | 24 | 1.39 | 1.16 | 1.35 | <img src="readme/frames/Seoul.jpg" width="240" alt="Seoul"> |
| 🇬🇲 Serrekunda | Gambia |  | 10 | 4 | 1.36 |  | 1.16 | <img src="readme/frames/Serrekunda.jpg" width="240" alt="Serrekunda"> |
| 🇸🇬 Singapore | Singapore |  | 13 | 27 | 1.41 |  | 1.26 | <img src="readme/frames/Singapore.jpg" width="240" alt="Singapore"> |
| 🇧🇬 Sofia | Bulgaria |  | 12 | 14 | 1.46 |  | 1.11 | <img src="readme/frames/Sofia.jpg" width="240" alt="Sofia"> |
| 🇸🇪 Stockholm | Sweden |  | 10 | 27 | 1.38 |  | 1.33 | <img src="readme/frames/Stockholm.jpg" width="240" alt="Stockholm"> |
| 🇦🇺 Sydney, NSW | Australia | 10 | 10 | 19 | 1.40 | 1.40 | 1.39 | <img src="readme/frames/Sydney.jpg" width="240" alt="Sydney"> |
| 🇧🇩 Sylhet | Bangladesh |  | 10 | 10 | 1.16 |  | 1.15 | <img src="readme/frames/Sylhet.jpg" width="240" alt="Sylhet"> |
| 🇪🇪 Tallinn | Estonia |  |  |  | 1.43 |  |  |  |
| 🇮🇷 Tehran | Iran |  |  |  | 1.34 |  |  |  |
| 🇯🇵 Tokyo | Japan |  | 10 | 25 | 1.30 |  | 1.38 | <img src="readme/frames/Tokyo.jpg" width="240" alt="Tokyo"> |
| 🇨🇦 Toronto, ON | Canada | 1 | 10 | 16 | 1.37 |  | 1.20 | <img src="readme/frames/Toronto.jpg" width="240" alt="Toronto"> |
| 🇦🇺 Townsville, QLD | Australia | 1 | 14 | 8 | 1.38 |  | 1.39 | <img src="readme/frames/Townsville.jpg" width="240" alt="Townsville"> |
| 🇷🇺 Tyumen | Russia |  | 10 | 11 | 1.31 |  | 1.18 | <img src="readme/frames/Tyumen.jpg" width="240" alt="Tyumen"> |
| 🇲🇳 Ulaanbaatar | Mongolia |  | 10 | 16 | 1.42 |  | 1.25 | <img src="readme/frames/Ulaanbaatar.jpg" width="240" alt="Ulaanbaatar"> |
| 🇨🇦 Vancouver, BC | Canada |  | 10 | 17 | 1.31 |  | 1.36 | <img src="readme/frames/Vancouver.jpg" width="240" alt="Vancouver"> |
| 🇦🇹 Vienna | Austria |  | 10 | 32 | 1.37 |  | 1.20 | <img src="readme/frames/Vienna.jpg" width="240" alt="Vienna"> |
| 🇵🇱 Warsaw | Poland | 1 | 10 | 35 | 1.34 | 1.13 | 1.24 | <img src="readme/frames/Warsaw.jpg" width="240" alt="Warsaw"> |
| 🇺🇸 Washington, DC | United States |  | 10 | 19 | 1.23 |  | 1.30 | <img src="readme/frames/Washington.jpg" width="240" alt="Washington"> |
| 🇨🇦 Winnipeg, MB | Canada |  |  |  | 1.31 |  |  |  |
| 🇲🇲 Yangon | Myanmar | 1 | 11 | 8 | 1.38 |  | 1.30 | <img src="readme/frames/Yangon.jpg" width="240" alt="Yangon"> |
| 🇮🇩 Yogyakarta | Indonesia |  | 12 | 8 | 1.29 |  | 1.17 | <img src="readme/frames/Yogyakarta.jpg" width="240" alt="Yogyakarta"> |
<!-- /city table -->

</details>

**The cities**: grey ones have only CROWD values so far.

[![Map of the cities](readme/map_cities.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/map_cities.html)

### Crossing speed in the generated clips

Each set was tracked with YOLO11x and BoT-SORT and run through the same crossing rule as CROWD (`analysis.py`), with crowd-city's Waymo-calibrated speed model, so a crossing's speed is in m/s only where the model qualifies the track, exactly as in crowd-city's table of CROWD. "Real continuation" is the real 8 seconds that follow each MiniMax H3 reference frame in its CROWD video (`paired_real_clips.py`), tracked the same way: the same scenes as MiniMax H3, but real (793 pairs; the CROWD video of 4 Oldenburg frames is not on lin4080). Numbers are as of 3 November 2026; `speed_statistics.py` writes the tests (`_output/generated_statistics.csv`) and the reliability of the city medians (`generated_reliability.csv`). Tests run on clip or city medians, with 95% intervals from bootstraps over clips, clip pairs or cities.

| Set | Clips | Clips with a counted crossing (95% CI) | Crossings | Crossings with an m/s speed | Median speed (95% CI) | Initiation times |
|---|---|---|---|---|---|---|
| Veo 3.1 Fast | 109 | 72% (62-79%) | 192 | 142 | 1.23 m/s (1.18-1.28) | 2 |
| MiniMax H3 | 797 | 58% (55-62%) | 1,167 | 830 | 1.27 m/s (1.25-1.29) | 29 |
| Real continuation | 793 | 69% (66-72%) | 1,514 | 956 | 1.39 m/s (1.37-1.40) | 13 |

City medians (cities with at least 3 m/s speeds) against CROWD and against the real continuations:

| Set | Reference | Cities | Spearman ρ (95% CI) | Median difference | Spread of city medians / reference's (IQR ratio) |
|---|---|---|---|---|---|
| Veo 3.1 Fast | CROWD | 11 | 0.52 (−0.16 to 0.89) | −0.09 m/s (p = .002) | 1.7 |
| MiniMax H3 | CROWD | 72 | 0.08 (−0.18 to 0.34) | −0.07 m/s (p < .001) | 1.8 (1.3-2.5) |
| Real continuation | CROWD | 71 | −0.13 (−0.37 to 0.12) | +0.04 m/s (p = .001) | 1.4 (1.1-2.1) |
| MiniMax H3 | Real continuation | 68 | −0.13 (−0.39 to 0.14) | −0.12 m/s (p < .001) | 1.4 (0.9-1.8) |

Crossings are counted with the crowd-city code of CROWD's current run (commit 9a35c28: track joining, corrected segmentation clock), and speeds are computed by the same function as CROWD's per-city table. CROWD's values are still those of 8 October 2026, from an earlier version of the code; the nightly job replaces them when the current run ends.

- **Generated pedestrians cross about 0.1 m/s slower than real ones.** On the 289 clip pairs with an m/s speed in both, MiniMax H3 is 0.11 m/s (0.08-0.14) slower than the real continuation of the same scene (Wilcoxon signed-rank, p < .001); per pair the speeds agree only weakly (Spearman 0.26). **Not a matter of resolution:** the real continuations scaled to MiniMax's 864×480 give the same speeds as at 1280×720 (median difference 0.000 m/s, −0.003 to 0.004, Spearman 0.93, 382 clips; `results/real_minimax_480`), and MiniMax H3 is 0.11 m/s slower than those too.
- **MiniMax H3 loses crossings and adds waits.** Of the 549 real continuations with a crossing, 150 (27%) have none in the generated clip, against 64 generated clips with a crossing the real one lacks (exact McNemar, p < .001). MiniMax shows a pedestrian waiting before stepping onto the road in 26 clips against 13 real continuations (p = .035; median wait 1.33 s in both). On the 30 shared cities, Veo 3.1 Fast keeps a crossing more often than MiniMax H3 (72% against 61%, Fisher p = .049).
- **Eight-second clips cannot rank cities, even real ones.** In the real continuations the city explains none of the variance in clip speed (Kruskal-Wallis η²_H = −0.02, 69 cities, p = .67), and the split-half reliability of a city median from about six clips with a speed is −0.24 (−0.67 to 0.12): no reference could correlate with such city medians. So the weak correlations of the models with CROWD say little about the models; the per-clip pairing is the fair test. MiniMax H3 is similar (η²_H = 0.03, p = .21; reliability 0.26, −0.07 to 0.51). Veo's clips of one city are more alike (η²_H = 0.19, p = .03 in nine cities), perhaps because their hand-picked frames resemble each other.

**Speed of every crossing per set** (left; the dashed line is the median of CROWD's city medians) and **MiniMax H3 minus its real continuation, per clip** (right):

[![Speed of every crossing per set](readme/ecdf_speed_sets.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/ecdf_speed_sets.html) [![MiniMax H3 minus its real continuation](readme/hist_speed_paired_difference.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/hist_speed_paired_difference.html)

**Crossing speed per city**: CROWD (crowd-city's run of 8 October 2026, made before the repair of the tracking files: to be replaced by its current values), Veo 3.1 Fast and MiniMax H3. The numbers after each city are the clips per model (Veo 3.1 / MiniMax H3); cities with one Veo clip have a bar from at most a few crossings. `--pdf` writes the paper's version, a dot plot of the same values in two panels that fills one page.

[![Crossing speed per city](readme/speed_generated_vs_crowd.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/speed_generated_vs_crowd.html)

**City medians against CROWD** (cities with at least 3 m/s speeds; the line is equality):

[![City medians against CROWD](readme/scatter_speed_city_vs_crowd.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/scatter_speed_city_vs_crowd.html)

**MiniMax H3 against the real continuation of the same scenes**, per city:

[![MiniMax H3 against the real continuations](readme/scatter_speed_paired.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/scatter_speed_paired.html)

### Do the generated videos keep the look of their city?

Each clip was sampled every second, and every frame was scored with CLIP (ViT-B/32, LAION-2B) against prompts naming each of the 100 cities ("a photo taken from a car driving through a street in {city}, {country}") and each of their 50 countries. The first frame is the CROWD reference frame as the model rendered it, so it shows how recognisable the real scene is; the later frames show whether the model keeps that look while it invents the rest of the clip.

| Model | Videos | City in CLIP's top 5 of 100: first frame | last frame | Country ranked first: first frame | last frame | Similarity of last to first frame |
|---|---|---|---|---|---|---|
| Veo 3.1 Fast | 109 | 76% | 72% | 56% | 50% | 0.86 |
| MiniMax H3 | 797 | 71% | 71% | 49% | 46% | 0.84 |
| MiniMax H3, the 30 Veo cities | 312 | 74% | 68% | 52% | 47% | 0.85 |

- **The look of the city survives the 8 seconds.** CLIP finds the city about as often in the last frame as in the reference frame, for both models. Whatever the models change, the street stays recognisably that city's.
- **The models are close.** On the same 30 cities, Veo 3.1 Fast and MiniMax H3 lose 5 to 6 points of city recognition over the clip; the difference between them is within the 95% intervals (bands in the figure).
- **MiniMax H3 drifts a little more** from its reference frame (CLIP similarity 0.84 against 0.86 at the end), and 5.6% of its clips end below 0.7, against 0.9% of Veo's. These clips (in `_output/appearance_frames.csv`) are the first to check for artefacts such as pedestrians changing into other road users.
- **No clip is frozen**: every second differs from the one before (mean absolute pixel change 25-34 of 255). Veo 3.1 Fast changes more per second (29-34) than MiniMax H3 (25-26).

[![City recognition over the clip](readme/line_city_recognition.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/line_city_recognition.html)

[![Drift from the reference frame](readme/line_appearance_drift.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/vlm-crossing/blob/main/readme/line_appearance_drift.html)

Per-city values are in `_output/appearance_cities.csv`. With one Veo video, a city's value is that single clip, so only the ten Veo cities with 5-11 videos say something per city.

### Analyses still to run

1. **Current CROWD values**: replace crowd-city's run of 8 October 2026 (made before the tracking files were repaired) with its current per-city table, and rerun `generated_speeds.py` and `speed_generated_vs_crowd.py`.
2. **More videos**: MiniMax H3 generation continues towards 100 cities, and Veo 3.1 Fast has one clip in 20 of its 30 cities; rerun the tracking for new clips (each set in its own output folder, see [Generated videos and the comparison with CROWD](#generated-videos-and-the-comparison-with-crowd)).
3. **Real continuations for Veo 3.1 Fast**: its clips do not record the source video and second of their reference frame; with them, `paired_real_clips.py` could cut Veo's real continuations too.
4. **Track stability and artefacts (RQ1)**: how many pedestrians appear, disappear or change class during a clip, generated against real continuation, and a manual check of the clips.
5. **Side of the road**: the side on which oncoming vehicles pass in left-hand traffic cities (London, Sydney, Melbourne, Tokyo, Hong Kong ...) against right-hand ones, as a check of whether the models keep a basic local rule.

## Configuration

| Key | Meaning |
|---|---|
| `data`, `output` | Input and output folders, relative to the repository or absolute |
| `cities_analyse` | Only these city folders; empty means all |
| `tracking_model`, `bbox_tracker`, `track_buffer_sec` | YOLO weights, BoT-SORT YAML, seconds a lost track is kept |
| `yolo_imgsz`, `device`, `half_precision` | Tracking settings |
| `save_annotated_video`, `display_frame_tracking` | Write `_output/annotated/<city>/<video>.mp4` / show frames live |
| `min_confidence` | Detections below this are ignored by the analysis |
| `boundary_left`, `boundary_right` | Road strip of the `detector` rule, in normalised image x |
| `check_per_sec_time` | Hesitation checks per second |
| `crossing_rule` | `road_crossing` (rule D, needs segmentation) or `detector` |
| `use_segmentation`, `segmentation_is_primary` | Run the road-surface pass; report its values |
| `segmentation_model`, `segmentation_device`, `segmentation_input_width/height`, `segmentation_min_confidence` | SegFormer settings |
| `segmentation_coarse_hz`, `segmentation_refine_hz` | Sampling rates for finding road entry and exit |
| `segmentation_batch_size`, `segmentation_parallel_videos` | GPU batch and videos read at once (lower the batch if the GPU runs out of memory) |
| `seg_data` | Folder of crowd-city's surface-label store |
| `processing_fps`, `ftp_base_url` | crowd-city settings for its YouTube footage; `null` and empty here |
| `min/max_speed_limit`, `min/max_waiting_time` | Values outside these ranges are dropped (crowd-city's 0.3–3.5 for speed) |
| `speed_model` | crowd-city's `crowd_waymo_pipeline_model.json`: speeds in m/s (only tracks that pass its reliability gates get one); empty: the relative index |
| `cpu_worker` | Processes used for crossing detection |
| `save_images` | Also export PNG/EPS figures (needs kaleido) |
| `open_figures` | Open every HTML figure in the default browser when it is written |

Update the copy of crowd-city with `python sync_crowd_city.py /path/to/crowd-city`.
