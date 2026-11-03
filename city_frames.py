"""One frame of a generated clip for each of the 100 cities, as a labelled grid for the paper.

For every city of CROWD's per-city table (plus the generated cities it lacks, such as Seattle), takes the middle frame
of one MiniMax H3 clip, preferring a clip in which the pipeline counted a crossing (first by name), and tiles the
frames alphabetically with the city name under each. Cities without a generated clip yet get a grey tile, so the grid
keeps its 100 places while generation continues. Writes <output>/figures/frames_cities.jpg (and with --pdf-dir a copy
there for the paper).

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
LABEL = 30  # height of the city name under each frame
SECOND = 4.0  # frame taken from the middle of the 8-second clip
# TrueType fonts with the letters of all city names (PIL's built-in font lacks e.g. ø and ü); the first found is used.
FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "C:/Windows/Fonts/arial.ttf")


def frame_at(path: str, second: float) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = capture.read()
    capture.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


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
    crossed = set()
    if os.path.isfile(args.crossings):
        crossed = set(pl.read_csv(args.crossings, columns=["video"], schema_overrides={"video": pl.Utf8})["video"])
    clips = clips.with_columns(pl.col("video").is_in(list(crossed)).alias("crossed")).sort(
        "city", pl.col("crossed").not_(), "video")
    chosen = {c: f for c, f in clips.group_by("city", maintain_order=True).agg(pl.col("file").first()).iter_rows()}
    cities = sorted(set(pl.read_csv(args.crowd)["locality"]) | set(folder_cities(args.generated).values()))

    rows = -(-len(cities) // COLUMNS)
    grid = Image.new("RGB", (COLUMNS * TILE[0], rows * (TILE[1] + LABEL)), "white")
    draw = ImageDraw.Draw(grid)
    path = next((f for f in FONTS if os.path.isfile(f)), None)
    font = ImageFont.truetype(path, 20) if path else ImageFont.load_default(size=20)
    for i, city in enumerate(cities):
        x, y = (i % COLUMNS) * TILE[0], (i // COLUMNS) * (TILE[1] + LABEL)
        frame = frame_at(os.path.join(common.root_dir, chosen[city]), SECOND) if city in chosen else None
        if frame is None:
            draw.rectangle((x + 2, y + 2, x + TILE[0] - 3, y + TILE[1] - 3), fill="#e4e3df")
            draw.text((x + TILE[0] / 2, y + TILE[1] / 2), "not generated yet", fill="#6f6e69", font=font, anchor="mm")
        else:
            grid.paste(fit(frame).crop((2, 2, TILE[0] - 2, TILE[1] - 2)), (x + 2, y + 2))
        draw.text((x + TILE[0] / 2, y + TILE[1] + LABEL / 2), city, fill="black", font=font, anchor="mm")

    target = os.path.join(common.get_output_dir(), "figures", "frames_cities.jpg")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    grid.save(target, quality=88)
    common.logger.info(f"{len(chosen)} of {len(cities)} cities have a generated frame; wrote {target}.")
    if args.pdf_dir:
        shutil.copy(target, os.path.join(args.pdf_dir, "frames_cities.jpg"))


if __name__ == "__main__":
    main()
