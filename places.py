"""Display labels of the cities: the country's flag and, in the United States, Canada and Australia, the state.

The state and country of a city come from CROWD's per-city table (crowd/crowd_crossing_speed_by_city.csv) and, for the
cities it lacks such as Seattle, from crowd/cities.csv, which also gives each country's ISO code for its flag.
Figures drawn with Pillow cannot render emoji flags (they need text shaping), so they use the Twemoji flag images in
assets/flags/ (CC-BY 4.0, github.com/jdecked/twemoji), fetched once per country when missing.
"""

import os
import urllib.request

import polars as pl

import common

STATE_COUNTRIES = ("United States", "Canada", "Australia")
CROWD = os.path.join(common.root_dir, "crowd", "crowd_crossing_speed_by_city.csv")
CITIES = os.path.join(common.root_dir, "crowd", "cities.csv")
FLAGS = os.path.join(common.root_dir, "assets", "flags")
TWEMOJI = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{}.png"


def flag(iso2: str) -> str:
    """Emoji flag of a two-letter ISO country code (regional indicator symbols)."""
    if not iso2 or len(iso2) != 2 or not iso2.isalpha():
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in iso2.upper())


def flag_image(iso2: str):
    """Path of the Twemoji flag PNG of a two-letter ISO country code, fetched once into assets/flags; None if absent."""
    if not flag(iso2):
        return None
    path = os.path.join(FLAGS, iso2.upper() + ".png")
    if not os.path.isfile(path):
        code = "-".join(f"{ord(c):x}" for c in flag(iso2))
        try:
            os.makedirs(FLAGS, exist_ok=True)
            urllib.request.urlretrieve(TWEMOJI.format(code), path)
        except OSError as error:
            common.logger.warning(f"No flag for {iso2}: {error}")
            return None
    return path


def _places(crowd: str, cities: str) -> tuple:
    """City name -> (state, country), and country -> ISO two-letter code."""
    mapping = pl.read_csv(cities, infer_schema_length=0)
    iso2 = dict(mapping.select("country", "iso2").unique("country").iter_rows())
    places = {}
    for city, state, country in mapping.select("locality", "state", "country").iter_rows():
        places.setdefault(city, (state, country))
    if os.path.isfile(crowd):  # CROWD's own state and country take precedence
        for city, state, country in pl.read_csv(crowd, infer_schema_length=0).select(
                "locality", "state", "country").iter_rows():
            places[city] = (state, country)
    return places, iso2


def city_iso2(crowd: str = CROWD, cities: str = CITIES) -> dict:
    """City name -> two-letter ISO code of its country."""
    places, iso2 = _places(crowd, cities)
    return {city: iso2.get(country, "") for city, (_, country) in places.items()}


def city_flag_images(names, crowd: str = CROWD, cities: str = CITIES) -> dict:
    """City name -> path of its country's flag image (flag_image) for the cities names, for figures without emoji."""
    iso2 = city_iso2(crowd, cities)
    return {city: flag_image(iso2.get(city, "")) for city in names}


def city_labels(flags: bool = True, crowd: str = CROWD, cities: str = CITIES) -> dict:
    """City name -> label such as "🇺🇸 Portland, OR" (without the flag when flags is False)."""
    places, iso2 = _places(crowd, cities)
    labels = {}
    for city, (state, country) in places.items():
        label = f"{city}, {state}" if country in STATE_COUNTRIES and state else city
        mark = flag(iso2.get(country, "")) if flags else ""
        labels[city] = f"{mark} {label}" if mark else label
    return labels
