"""Display labels of the cities: the country's flag and, in the United States, Canada and Australia, the state.

The state and country of a city come from CROWD's per-city table (crowd/crowd_crossing_speed_by_city.csv) and, for the
cities it lacks such as Seattle, from crowd/cities.csv, which also gives each country's ISO code for its flag.
"""

import os

import polars as pl

import common

STATE_COUNTRIES = ("United States", "Canada", "Australia")
CROWD = os.path.join(common.root_dir, "crowd", "crowd_crossing_speed_by_city.csv")
CITIES = os.path.join(common.root_dir, "crowd", "cities.csv")


def flag(iso2: str) -> str:
    """Emoji flag of a two-letter ISO country code (regional indicator symbols)."""
    if not iso2 or len(iso2) != 2 or not iso2.isalpha():
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in iso2.upper())


def city_labels(flags: bool = True, crowd: str = CROWD, cities: str = CITIES) -> dict:
    """City name -> label such as "🇺🇸 Portland, OR" (without the flag when flags is False)."""
    mapping = pl.read_csv(cities, infer_schema_length=0)
    iso2 = dict(mapping.select("country", "iso2").unique("country").iter_rows())
    places = {}
    for city, state, country in mapping.select("locality", "state", "country").iter_rows():
        places.setdefault(city, (state, country))
    if os.path.isfile(crowd):  # CROWD's own state and country take precedence
        for city, state, country in pl.read_csv(crowd, infer_schema_length=0).select(
                "locality", "state", "country").iter_rows():
            places[city] = (state, country)
    labels = {}
    for city, (state, country) in places.items():
        label = f"{city}, {state}" if country in STATE_COUNTRIES and state else city
        mark = flag(iso2.get(country, "")) if flags else ""
        labels[city] = f"{mark} {label}" if mark else label
    return labels
