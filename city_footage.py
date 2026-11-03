"""How much footage of each city CROWD found on YouTube, as a stand-in for how well the city is represented online.

The training data of the video generators is not public, so the YouTube videos and hours of dashcam and walking footage
that CROWD's mapping.csv lists for a city serve as a proxy for its presence in web video, next to the city's population
and gross metropolitan product from the same mapping. Writes crowd/city_footage.csv for the cities of crowd/cities.csv,
which speed_statistics.py relates to how far each city's generated speed is from CROWD's.

    uv run python city_footage.py --mapping ~/repos/crowd/mapping.csv
"""

import argparse
import ast
import csv
import os

import polars as pl

import common

TARGET = os.path.join(common.root_dir, "crowd", "city_footage.csv")


def number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", default=os.path.expanduser("~/repos/crowd/mapping.csv"),
                        help="CROWD's mapping.csv (github.com/Shaadalam9/pedestrians-in-youtube)")
    parser.add_argument("--cities", default=os.path.join(common.root_dir, "crowd", "cities.csv"))
    args = parser.parse_args()

    csv.field_size_limit(10 ** 9)
    wanted = set(pl.read_csv(args.cities, infer_schema_length=0).select("locality", "country").iter_rows())
    footage = {}
    with open(args.mapping, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["locality"], r["country"])
            if key not in wanted:
                continue
            videos = [v for v in r["videos"].strip("[]").split(",") if v]
            # start_time and end_time hold one list of segments (in s) per video
            seconds = sum(e - s for starts, ends in zip(ast.literal_eval(r["start_time"]),
                                                       ast.literal_eval(r["end_time"]))
                          for s, e in zip(starts, ends))
            city = footage.setdefault(key, {"videos": 0, "seconds": 0, "population": None, "gmp": None})
            city["videos"] += len(videos)
            city["seconds"] += seconds
            city["population"] = city["population"] or number(r["population_locality"])
            city["gmp"] = city["gmp"] or number(r["gmp"])
    table = pl.DataFrame([{"locality": c, "country": k, "videos": v["videos"],
                           "footage_h": round(v["seconds"] / 3600, 1), "population": v["population"],
                           "gmp_billion_usd": v["gmp"]} for (c, k), v in footage.items()]).sort("locality")
    table.write_csv(TARGET)
    common.logger.info(f"Footage of {table.height} of {len(wanted)} cities; wrote {TARGET}.")


if __name__ == "__main__":
    main()
