"""One frame of a generated clip for each of the 100 cities, as a labelled grid for the paper.

For every city of CROWD's per-city table (plus the generated cities it lacks, such as Seattle), takes the MiniMax H3
crossing with the tallest pedestrian (from analysis.py's crossings and main.py's tracks), and crops the frame halfway
through that crossing around the pedestrian, outlined, so that the crossing is visible at thumbnail size. Cities whose
clips have no counted crossing get the middle frame of their first clip; cities without a generated clip yet get a
grey tile, so the grid keeps its 100 places while generation continues. The crossing chosen per city is kept in
results/minimax_h3/city_frames.csv and used when the tracks are not there; the frames themselves need the videos.
Tiles are in alphabetical order with the city name under each.

Writes <output>/figures/frames_cities.jpg (and with --pdf-dir a copy there for the paper). With --readme, also writes
each tile to readme/frames/ and the table of cities in README.md (between its "city table" markers): country, clips
per model, crossings, median speeds, and the tile.

    uv run python city_frames.py --pdf-dir ../overleaf/vlm-crossing-paper/figures
"""

import argparse
import os
import shutil

import cv2
import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

import common
import places
from generated_speeds import folder_cities

MODEL = "minimax_h3"
COLUMNS = 10
TILE = (320, 180)  # width, height of each frame
LABEL = 40  # height of the city name under each frame
FONT_SIZE = 34  # largest font of the city names; long names get a smaller one to fit the tile
SECOND = 4.0  # middle of the 8-second clip, for clips without a counted crossing
ZOOM = 4.5  # crop height in pedestrian heights...
MIN_CROP = 0.45  # ...but at least this share of the frame height
OUTLINE = "#ffd400"
FLAG_SCALE = 1.1  # side of the square flag image in font sizes
README = os.path.join(common.root_dir, "README.md")
MARKERS = ("<!-- city table -->", "<!-- /city table -->")
# TrueType fonts with the letters of all city names (PIL's built-in font lacks e.g. ø and ü); the first found is used.
FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "C:/Windows/Fonts/arial.ttf")


def frame_at(path: str, second: float = None, index: int = None) -> np.ndarray:
    """A frame by time in seconds or by 0-based index."""
    capture = cv2.VideoCapture(path)
    if index is None:
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def best_crossings(crossings: str, bbox_dir: str) -> dict:
    """City -> (video, frame-count, box as x, y, w, h in [0, 1]) of the crossing with the tallest pedestrian, taken
    halfway through the crossing."""
    table = pl.read_csv(crossings, infer_schema_length=None, schema_overrides={"video": pl.Utf8})
    best = {}
    folders = [f for f in os.listdir(bbox_dir) if os.path.isdir(os.path.join(bbox_dir, f))]
    for row in table.iter_rows(named=True):
        paths = [os.path.join(bbox_dir, folder, f) for folder in folders
                 for f in os.listdir(os.path.join(bbox_dir, folder))
                 if f.startswith(row["video"] + "_") and f.endswith(".csv")]
        if not paths:
            continue
        track = (pl.read_csv(paths[0]).filter(pl.col("unique-id") == int(row["track_id"]))
                 .filter(pl.col("frame-count").is_between(row["start_frame"], row["end_frame"])).sort("frame-count"))
        if track.height == 0:
            continue
        middle = track.row(track.height // 2, named=True)
        size = track["height"].median()
        if size > best.get(row["city"], (0,))[0]:
            best[row["city"]] = (size, row["video"], middle["frame-count"],
                                 (middle["x-center"], middle["y-center"], middle["width"], middle["height"]))
    return {city: value[1:] for city, value in best.items()}


def around(frame: np.ndarray, box: tuple) -> Image.Image:
    """Crop the frame to the tile's aspect ratio around the pedestrian's box, outline the box, and resize."""
    image = Image.fromarray(frame)
    width, height = image.size
    x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    crop_h = min(height, max(ZOOM * h, MIN_CROP * height))
    crop_w = min(width, crop_h * TILE[0] / TILE[1])
    crop_h = crop_w * TILE[1] / TILE[0]
    left = min(max(0, x - crop_w / 2), width - crop_w)
    top = min(max(0, y - crop_h / 2), height - crop_h)
    scale = TILE[0] / crop_w
    tile = image.crop((int(left), int(top), int(left + crop_w), int(top + crop_h))).resize(TILE, Image.LANCZOS)
    pad = 3
    ImageDraw.Draw(tile).rectangle(((x - w / 2 - left) * scale - pad, (y - h / 2 - top) * scale - pad,
                                    (x + w / 2 - left) * scale + pad, (y + h / 2 - top) * scale + pad),
                                   outline=OUTLINE, width=3)
    return tile


def fit(frame: np.ndarray) -> Image.Image:
    """Centre-crop to the tile's aspect ratio and resize."""
    image = Image.fromarray(frame)
    width, height = image.size
    target = TILE[0] / TILE[1]
    if width / height > target:
        crop = int(height * target)
        image = image.crop(((width - crop) // 2, 0, (width + crop) // 2, height))
    else:
        crop = int(width / target)
        image = image.crop((0, (height - crop) // 2, width, (height + crop) // 2))
    return image.resize(TILE, Image.LANCZOS)


def readme_table(cities: list, clips: pl.DataFrame, files: dict, tiles: dict, args) -> str:
    """Markdown table of the cities: country, clips per model, MiniMax H3 crossings, median speeds, frame."""
    crowd = pl.read_csv(args.crowd, infer_schema_length=None)
    country = dict(pl.read_csv(os.path.join(common.root_dir, "crowd", "cities.csv"), infer_schema_length=0)
                   .select("locality", "country").iter_rows())
    country.update(zip(crowd["locality"], crowd["country"]))
    crowd_speed = dict(zip(crowd["locality"], crowd["median_crossing_speed_mps"]))
    by_city = os.path.join(common.get_output_dir(), "generated_speed_by_city.csv")
    speed = {}
    if os.path.isfile(by_city):
        speed = {(m, c): (n, s) for m, c, n, s in pl.read_csv(by_city).select(
            "model", "locality", "crossings", "median_crossing_speed_mps").iter_rows()}
    counts = {(m, c): n for m, c, n in clips.group_by("model", "city").len().iter_rows()}

    label = places.city_labels(crowd=args.crowd)

    def mps(value):
        return f"{value:.2f}" if value is not None else ""
    lines = ["| City | Country | Veo 3.1 Fast clips | MiniMax H3 clips | MiniMax H3 crossings | CROWD speed (m/s) "
             "| Veo 3.1 Fast speed | MiniMax H3 speed | Frame |",
             "|---|---|--:|--:|--:|--:|--:|--:|---|"]
    for city in cities:
        frame = ""
        if city in tiles:
            image = os.path.relpath(tiles[city][0], common.root_dir)
            frame = f'<img src="{image}" width="240" alt="{city}">'
        lines.append(" | ".join([
            f"| {label.get(city, city)}", country.get(city, ""), str(counts.get(("veo3", city), "")),
            str(counts.get(("minimax_h3", city), "")), str(speed.get(("minimax_h3", city), ("", None))[0]),
            mps(crowd_speed.get(city)), mps(speed.get(("veo3", city), (0, None))[1]),
            mps(speed.get(("minimax_h3", city), (0, None))[1]), frame]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "crowd", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--generated", default=os.path.join(common.root_dir, "videos", "generated_videos.csv"))
    parser.add_argument("--crossings", default=os.path.join(common.root_dir, "results", MODEL, "crossings.csv"))
    parser.add_argument("--bbox", default=os.path.join(common.root_dir, "_output", MODEL, "bbox"),
                        help="main.py's tracks of the MiniMax H3 clips (without them, tiles show the middle frame)")
    parser.add_argument("--pdf-dir", help="also copy the grid to this folder (e.g. the paper's figures)")
    parser.add_argument("--readme", action="store_true", help="also write the tiles and the table of cities to README")
    args = parser.parse_args()

    used = pl.read_csv(args.generated, infer_schema_length=None).filter(pl.col("status") == "used")
    clips = (used.filter((pl.col("model") == MODEL) & pl.col("file").is_not_null())
             .with_columns(pl.col("file").map_elements(lambda f: os.path.splitext(os.path.basename(f))[0],
                                                       return_dtype=pl.Utf8).alias("video")))
    files = dict(zip(clips["video"], clips["file"]))
    chosen = {c: f for c, f in clips.sort("city", "video").group_by("city", maintain_order=True)
              .agg(pl.col("file").first()).iter_rows()}
    # The crossing shown per city is kept in results/, so the grid is redrawn the same without main.py's tracks.
    chosen_path = os.path.join(os.path.dirname(args.crossings), "city_frames.csv")
    if os.path.isdir(args.bbox):
        crossing = best_crossings(args.crossings, args.bbox)
        pl.DataFrame([{"city": c, "video": v, "frame_count": f, "x": b[0], "y": b[1], "width": b[2], "height": b[3]}
                      for c, (v, f, b) in sorted(crossing.items())]).write_csv(chosen_path)
    elif os.path.isfile(chosen_path):
        crossing = {r["city"]: (r["video"], r["frame_count"], (r["x"], r["y"], r["width"], r["height"]))
                    for r in pl.read_csv(chosen_path, schema_overrides={"video": pl.Utf8}).iter_rows(named=True)}
    else:
        crossing = {}
    cities = sorted(set(pl.read_csv(args.crowd)["locality"]) | set(folder_cities(args.generated).values()))

    rows = -(-len(cities) // COLUMNS)
    grid = Image.new("RGB", (COLUMNS * TILE[0], rows * (TILE[1] + LABEL)), "white")
    draw = ImageDraw.Draw(grid)
    font_path = next((f for f in FONTS if os.path.isfile(f)), None)

    fonts = {}

    def font(size: int):
        if size not in fonts:  # loaded once per size: opening the font file for every label runs out of handles
            fonts[size] = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
        return fonts[size]
    names = places.city_labels(flags=False, crowd=args.crowd)  # the font has no emoji: flags are pasted as images
    iso2 = places.city_iso2(crowd=args.crowd)
    tiles = {}
    tile_dir = os.path.join(common.root_dir, "readme", "frames")
    for i, city in enumerate(cities):
        x, y = (i % COLUMNS) * TILE[0], (i // COLUMNS) * (TILE[1] + LABEL)
        tile, source = None, None
        if city in crossing:
            video, frame_count, box = crossing[city]
            source = files[video]
            frame = frame_at(os.path.join(common.root_dir, source), index=frame_count - 1)
            tile = around(frame, box) if frame is not None else None
        if tile is None and city in chosen:
            source = chosen[city]
            frame = frame_at(os.path.join(common.root_dir, source), SECOND)
            tile = fit(frame) if frame is not None else None
        if tile is not None and args.readme:
            os.makedirs(tile_dir, exist_ok=True)
            path = os.path.join(tile_dir, os.path.basename(os.path.dirname(source)) + ".jpg")
            tile.save(path, quality=85)
            tiles[city] = (path, source)
        if tile is None:
            draw.rectangle((x + 2, y + 2, x + TILE[0] - 3, y + TILE[1] - 3), fill="#e4e3df")
            draw.text((x + TILE[0] / 2, y + TILE[1] / 2), "not generated yet", fill="#6f6e69", font=font(26),
                      anchor="mm")
        else:
            grid.paste(tile.crop((2, 2, TILE[0] - 2, TILE[1] - 2)), (x + 2, y + 2))
        name = names.get(city, city)
        flag_path = places.flag_image(iso2.get(city, ""))
        flag = Image.open(flag_path).convert("RGBA") if flag_path else None
        size = FONT_SIZE
        while True:  # the flag, a gap and the name, together no wider than the tile
            flag_side = round(size * FLAG_SCALE) if flag else 0
            gap = round(size * 0.3) if flag else 0
            width = flag_side + gap + draw.textlength(name, font=font(size))
            if size <= 12 or width <= TILE[0] - 8:
                break
            size -= 1
        left, middle = x + (TILE[0] - width) / 2, y + TILE[1] + LABEL / 2
        if flag:
            flag = flag.resize((flag_side, flag_side), Image.LANCZOS)
            grid.paste(flag, (round(left), round(middle - flag_side / 2)), flag)
        draw.text((left + flag_side + gap, middle), name, fill="black", font=font(size), anchor="lm")

    target = os.path.join(common.get_output_dir(), "figures", "frames_cities.jpg")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    grid.save(target, quality=88)
    common.logger.info(f"{len(chosen)} of {len(cities)} cities have a generated frame, {len(crossing)} of them "
                       f"centred on a crossing; wrote {target}.")
    if args.pdf_dir:
        shutil.copy(target, os.path.join(args.pdf_dir, "frames_cities.jpg"))
    if args.readme:
        shutil.copy(target, os.path.join(common.root_dir, "readme", "frames_cities.jpg"))
        text = open(README).read()
        if MARKERS[0] in text and MARKERS[1] in text:
            start, end = text.index(MARKERS[0]) + len(MARKERS[0]), text.index(MARKERS[1])
            text = text[:start] + "\n" + readme_table(cities, used, files, tiles, args) + "\n" + text[end:]
            open(README, "w").write(text)
        else:
            common.logger.warning(f"No {MARKERS[0]} in README.md; the table of cities was not written.")


if __name__ == "__main__":
    main()
