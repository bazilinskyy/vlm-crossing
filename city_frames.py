"""One frame of a generated clip for each of the 100 cities, as a labelled grid for the paper.

For every city of CROWD's per-city table (plus the generated cities it lacks, such as Seattle), takes the MiniMax H3
crossing with the tallest pedestrian (from analysis.py's crossings and main.py's tracks), and crops the frame halfway
through that crossing around the pedestrian, outlined, so that the crossing is visible at thumbnail size. Cities whose
clips have no counted crossing get the middle frame of their first clip; cities without a generated clip yet get a
grey tile, so the grid keeps its 100 places while generation continues. Tiles are in alphabetical order with the city
name under each. Writes <output>/figures/frames_cities.jpg (and with --pdf-dir a copy there for the paper).

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crowd", default=os.path.join(common.root_dir, "data", "crowd_crossing_speed_by_city.csv"))
    parser.add_argument("--generated", default=os.path.join(common.root_dir, "data", "generated_videos.csv"))
    parser.add_argument("--crossings", default=os.path.join(common.root_dir, "_output", MODEL, "crossings.csv"))
    parser.add_argument("--pdf-dir", help="also copy the grid to this folder (e.g. the paper's figures)")
    args = parser.parse_args()

    clips = (pl.read_csv(args.generated, infer_schema_length=None)
             .filter((pl.col("model") == MODEL) & (pl.col("status") == "used") & pl.col("file").is_not_null())
             .with_columns(pl.col("file").map_elements(lambda f: os.path.splitext(os.path.basename(f))[0],
                                                       return_dtype=pl.Utf8).alias("video")))
    files = dict(zip(clips["video"], clips["file"]))
    chosen = {c: f for c, f in clips.sort("city", "video").group_by("city", maintain_order=True)
              .agg(pl.col("file").first()).iter_rows()}
    bbox_dir = os.path.join(os.path.dirname(args.crossings), "bbox")
    crossing = best_crossings(args.crossings, bbox_dir) if os.path.isdir(bbox_dir) else {}
    cities = sorted(set(pl.read_csv(args.crowd)["locality"]) | set(folder_cities(args.generated).values()))

    rows = -(-len(cities) // COLUMNS)
    grid = Image.new("RGB", (COLUMNS * TILE[0], rows * (TILE[1] + LABEL)), "white")
    draw = ImageDraw.Draw(grid)
    path = next((f for f in FONTS if os.path.isfile(f)), None)

    def font(size: int):
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    for i, city in enumerate(cities):
        x, y = (i % COLUMNS) * TILE[0], (i // COLUMNS) * (TILE[1] + LABEL)
        tile = None
        if city in crossing:
            video, frame_count, box = crossing[city]
            frame = frame_at(os.path.join(common.root_dir, files[video]), index=frame_count - 1)
            tile = around(frame, box) if frame is not None else None
        if tile is None and city in chosen:
            frame = frame_at(os.path.join(common.root_dir, chosen[city]), SECOND)
            tile = fit(frame) if frame is not None else None
        if tile is None:
            draw.rectangle((x + 2, y + 2, x + TILE[0] - 3, y + TILE[1] - 3), fill="#e4e3df")
            draw.text((x + TILE[0] / 2, y + TILE[1] / 2), "not generated yet", fill="#6f6e69", font=font(26),
                      anchor="mm")
        else:
            grid.paste(tile.crop((2, 2, TILE[0] - 2, TILE[1] - 2)), (x + 2, y + 2))
        size = FONT_SIZE
        while size > 12 and draw.textlength(city, font=font(size)) > TILE[0] - 8:
            size -= 1
        draw.text((x + TILE[0] / 2, y + TILE[1] + LABEL / 2), city, fill="black", font=font(size), anchor="mm")

    target = os.path.join(common.get_output_dir(), "figures", "frames_cities.jpg")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    grid.save(target, quality=88)
    common.logger.info(f"{len(chosen)} of {len(cities)} cities have a generated frame, {len(crossing)} of them "
                       f"centred on a crossing; wrote {target}.")
    if args.pdf_dir:
        shutil.copy(target, os.path.join(args.pdf_dir, "frames_cities.jpg"))


if __name__ == "__main__":
    main()
