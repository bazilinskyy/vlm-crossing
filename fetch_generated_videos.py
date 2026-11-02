"""Fetch the generated videos from the project's Google Drive folders and sort them by model and city.

Every file of the shared folders is mirrored to data/_drive/<folder>/ (skipped when already there),
then each generated crossing video is hard-linked to data/<model>/<city>/ so that main.py can track
one model at a time ("data": "data/veo3" in config). data/generated_videos.csv lists every video of
the folders with its model, city, batch and source. Run again to pick up new videos:

    uvx --with gdown python fetch_generated_videos.py

Drive stops serving files to anonymous downloads after a few hundred ("Cannot retrieve the public
link ... many accesses"). Then download the folder in a signed-in browser (select all, Download) and
pass the zip(s), which are unpacked into the mirror before the missing files are fetched:

    uvx --with gdown python fetch_generated_videos.py --zip minimax_h3_zi=~/Downloads/videos-*.zip

Folders (from the #pr-vlm-crossing Discord channel):
    veo3_tanvir    Veo 3 videos by Tanvir: 10 cities with 10-11 videos (08-28-2026), 20 cities with one
                   (09-26-2026), and the first prompt tests (08-08-26, Videos/).
    minimax_h3_zi  MiniMax H3 videos by Zi (turbo 6 steps, 864x480, 8 s), one folder per city;
                   _unused/ holds clips Zi left out.
    minimax_ltx_benchmark_zi  Zi's comparison of MiniMax H3 and LTX 2.3 settings on the same 10 frames (F01-F10).
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import gdown

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MIRROR = os.path.join(DATA, "_drive")
MAPPING = os.path.join(DATA, "generated_videos.csv")
DOWNLOAD_THREADS = 8
FOLDERS = {
    "veo3_tanvir": "1gL40N0urK86SC8I9C765FdWdNJEk8z8O",
    "minimax_h3_zi": "1fTPnYr1DDFre84l-3Yv8ewj6XDM0dYWt",
    "minimax_ltx_benchmark_zi": "1JE4O8XMAjAvmG9d_oM5X8w93au1iTIu0",
}
# Folder names of the generated videos as CROWD's mapping.csv spells the city.
CITY_NAMES = {
    "BuenosAires": "Buenos Aires",
    "Copenhagen": "København",
    "Los-Angeles": "Los Angeles",
    "NewYork": "New York",
    "Saint-Petersburg": "Saint Petersburg",
    "SaoPaulo": "Sao Paulo",
    "Yangun": "Yangon",
    "warsaw": "Warsaw",
}
# ASCII folder names for cities whose CROWD name has letters outside it.
FOLDER_NAMES = {"København": "Copenhagen"}
COLUMNS = ["model", "city", "status", "batch", "file", "source_path", "drive_id", "reference_image",
           "youtube_id", "frame_time_s", "width", "height", "fps", "duration_s", "size_mb"]


def list_folder(folder_id: str) -> list:
    """Return (drive id, path) of every file in a shared folder and its sub-folders."""
    files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
    return [(f.id, f.path) for f in files]


def download(drive_id: str, path: str) -> bool:
    """Download one file unless it is already there; True when the file exists afterwards."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        try:
            if gdown.download(id=drive_id, output=path + ".part", quiet=True):
                os.replace(path + ".part", path)
                return True
        except Exception as e:  # Drive quota or network errors: wait and try again
            print(f"Attempt {attempt + 1} for {path} failed: {e}", flush=True)
        time.sleep(30 * (attempt + 1))
    return False


def city_of(name: str) -> str:
    name = CITY_NAMES.get(name, name)
    return name.replace("_", " ")


def describe(folder: str, path: str) -> dict:
    """Model, city, status and batch of one mirrored file from its place in the Drive folder."""
    parts = path.split("/")
    row = {"model": "", "city": "", "status": "", "batch": "", "youtube_id": "", "frame_time_s": "",
           "reference_image": ""}
    if folder == "veo3_tanvir":
        row["model"] = "veo3"
        if parts[0] == "08-28-2026" and parts[2] == "Videos":
            row.update(city=city_of(parts[1]), status="used", batch="2026-08-28 top 10",
                       reference_image="/".join(parts[:2] + ["Images", os.path.splitext(parts[3])[0] + ".png"]))
        elif parts[0] == "09-26-2026":
            row.update(city=city_of(parts[1]), status="used", batch="2026-09-26 one per city",
                       reference_image="/".join(parts[:2] + [os.path.splitext(parts[2])[0] + ".png"]))
        else:
            row.update(status="prompt test", batch=parts[0])
    elif folder == "minimax_h3_zi":
        row["model"] = "minimax_h3"
        unused = parts[0] == "_unused"
        if unused:
            parts = parts[1:]
        row.update(city=city_of(parts[1]), status="unused" if unused else "used", batch=parts[0])
        match = re.search(r"_([A-Za-z0-9_-]{11})_t(\d+)\.mp4$", path)
        if match:
            row.update(youtube_id=match.group(1), frame_time_s=match.group(2))
    else:
        setting = parts[0]
        row.update(model="minimax_h3" if setting.startswith("mm_") else "ltx_2.3", status="benchmark",
                   batch=setting)
    return row


def probe(path: str) -> dict:
    """Width, height, fps and duration from ffprobe (empty when ffprobe is missing or fails)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate:format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=60).stdout
        info = json.loads(out)
        stream = info["streams"][0]
        num, den = stream["avg_frame_rate"].split("/")
        return {"width": stream["width"], "height": stream["height"],
                "fps": round(float(num) / float(den), 3) if float(den) else "",
                "duration_s": round(float(info["format"]["duration"]), 2)}
    except Exception:
        return {"width": "", "height": "", "fps": "", "duration_s": ""}


def link(source: str, target: str) -> None:
    if os.path.exists(target):
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.link(source, target)


def unzip(folder: str, pattern: str) -> None:
    """Unpack a browser download of a Drive folder (one or more zip parts) into its mirror, keeping files there."""
    for archive in sorted(glob.glob(os.path.expanduser(pattern))):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = os.path.join(MIRROR, folder, member.filename)
                if member.is_dir() or os.path.exists(target):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(member) as source, open(target, "wb") as out:
                    out.write(source.read())
        print(f"Unpacked {archive} into {os.path.join(MIRROR, folder)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", action="append", default=[], metavar="FOLDER=PATTERN",
                        help="zip(s) of a folder downloaded in the browser, e.g. veo3_tanvir=~/Downloads/x*.zip")
    parser.add_argument("--no-download", action="store_true", help="only sort the files already mirrored")
    args = parser.parse_args()
    for spec in args.zip:
        folder, pattern = spec.split("=", 1)
        unzip(folder, pattern)
    files = []
    for folder, folder_id in FOLDERS.items():
        listed = list_folder(folder_id)
        print(f"{folder}: {len(listed)} files", flush=True)
        files += [(folder, drive_id, path, os.path.join(MIRROR, folder, path)) for drive_id, path in listed]
    # Drive serves one file at about 1.5 MB/s, so several are fetched at once.
    fetch = (lambda f: os.path.exists(f[3])) if args.no_download else (lambda f: download(f[1], f[3]))
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        ok = list(pool.map(fetch, files))
    failed = [f[3] for f, done in zip(files, ok) if not done]
    rows = []
    for (folder, drive_id, path, local), done in zip(files, ok):
        if done and path.lower().endswith(".mp4"):
            row = describe(folder, path)
            row.update(source_path=os.path.relpath(local, ROOT), drive_id=drive_id,
                       size_mb=round(os.path.getsize(local) / 1e6, 2), **probe(local))
            if row["reference_image"]:
                row["reference_image"] = os.path.join("data", "_drive", folder, row["reference_image"])
            if row["status"] == "used":
                city_folder = FOLDER_NAMES.get(row["city"], row["city"]).replace(" ", "_")
                target = os.path.join(DATA, row["model"], city_folder, os.path.basename(path))
                link(local, target)
                row["file"] = os.path.relpath(target, ROOT)
            rows.append(row)
    rows.sort(key=lambda r: (r["status"] != "used", r["model"], r["city"], r["batch"], r["source_path"]))
    with open(MAPPING, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    used = [r for r in rows if r["status"] == "used"]
    for model in sorted({r["model"] for r in used}):
        cities = {r["city"] for r in used if r["model"] == model}
        print(f"{model}: {sum(r['model'] == model for r in used)} videos in {len(cities)} cities")
    print(f"Wrote {len(rows)} videos to {os.path.relpath(MAPPING, ROOT)}; {len(failed)} downloads failed")
    if failed:
        print("\n".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
