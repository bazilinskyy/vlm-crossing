"""Cut the real 8 seconds that follow each MiniMax H3 reference frame from the CROWD video it was taken from.

Every MiniMax H3 clip in videos/generated_videos.csv starts from a frame of a CROWD video (youtube_id, frame_time_s).
This cuts [frame_time_s, frame_time_s + 8 s] of that video to data/real_minimax/<city>/<same file name>, so the
tracking and analysis can compare the generated continuation of a crossing scene with the real one. The cut is
re-encoded (a stream copy would start at the previous keyframe). Run it where the CROWD videos are, e.g. lin4080:

    uv run python paired_real_clips.py --videos '/media/salam/crowd-tue-*/pedestrians-in-youtube/videos'

Clips that already exist are skipped; data/real_minimax/clips.csv lists every clip and why one is missing.
"""

import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import polars as pl

import common

MODEL = "minimax_h3"
OUT = os.path.join(common.root_dir, "data", "real_minimax")


def cut(source: str, start: float, length: float, target: str) -> str:
    """Cut and re-encode one clip; return "ok" or the reason it failed."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part.mp4"
    result = subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                             "-ss", str(start), "-i", source, "-t", str(length), "-an",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "2", part],
                            capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(part):
            os.remove(part)
        return "ffmpeg failed: " + result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed"
    os.replace(part, target)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", action="append", required=True,
                        help="folder (glob) of CROWD videos named <youtube_id>.mp4; repeatable")
    parser.add_argument("--generated", default=os.path.join(common.root_dir, "videos", "generated_videos.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sources = {}
    for pattern in args.videos:
        for folder in sorted(glob.glob(pattern)):
            for path in glob.glob(os.path.join(folder, "*.mp4")):
                sources.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    clips = (pl.read_csv(args.generated, infer_schema_length=None)
             .filter((pl.col("model") == MODEL) & (pl.col("status") == "used"))
             .select("city", "file", "youtube_id", pl.col("frame_time_s").cast(pl.Float64),
                     pl.col("duration_s").cast(pl.Float64)))

    def one(row: dict) -> dict:
        target = os.path.join(OUT, os.path.basename(os.path.dirname(row["file"])), os.path.basename(row["file"]))
        source = sources.get(row["youtube_id"])
        if os.path.exists(target):
            status = "ok"
        elif source is None:
            status = "CROWD video not found"
        else:
            status = cut(source, row["frame_time_s"], row["duration_s"], target)
        return {"city": row["city"], "generated": row["file"], "real": os.path.relpath(target, common.root_dir),
                "youtube_id": row["youtube_id"], "start_s": row["frame_time_s"], "length_s": row["duration_s"],
                "source": source or "", "status": status}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, clips.to_dicts()))
    table = pl.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    table.write_csv(os.path.join(OUT, "clips.csv"))
    common.logger.info(f"{(table['status'] == 'ok').sum()} of {table.height} real clips ready in {OUT}.")
    for status, n in table.filter(pl.col("status") != "ok").group_by("status").len().iter_rows():
        common.logger.warning(f"{n} clips: {status}")


if __name__ == "__main__":
    main()
