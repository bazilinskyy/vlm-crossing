"""Copy one tracked set's crossings and videos from _output/<set>/ to results/<set>/, which the repository keeps.

main.py and analysis.py need a GPU and the videos; their results per set are small, so they are kept in results/ for
the scripts that compare the sets (generated_speeds.py, speed_statistics.py, city_frames.py), which then run from a
fresh checkout. Machine-specific columns (paths, file times) are left out. For the real continuations, the list of
clips with their CROWD video and times (data/real_minimax/clips.csv from paired_real_clips.py) is kept too.

    python export_results.py veo3
"""

import os
import sys

import polars as pl

import common

DROP = ("video_path", "csv", "file_size", "file_mtime_ns", "source")


def main() -> None:
    name = sys.argv[1]
    source = os.path.join(common.root_dir, "_output", name)
    target = os.path.join(common.root_dir, "results", name)
    os.makedirs(target, exist_ok=True)
    files = {"crossings.csv": os.path.join(source, "crossings.csv"), "videos.csv": os.path.join(source, "videos.csv")}
    if name.startswith("real_minimax"):
        files["clips.csv"] = os.path.join(common.root_dir, "data", "real_minimax", "clips.csv")
    for out, path in files.items():
        if not os.path.isfile(path):
            continue
        table = pl.read_csv(path, infer_schema_length=0)
        table.drop([c for c in DROP if c in table.columns]).write_csv(os.path.join(target, out))
        common.logger.info(f"{name}: {table.height} rows to {os.path.join(target, out)}")


if __name__ == "__main__":
    main()
