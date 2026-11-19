"""
Tiny synthetic Tiny-CropNet dataset with the same layout as the real one.

Used by the tests to run the data loaders and both training scripts end to end
without downloading anything.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd

WEATHER = [
    'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)', 'Precipitation (kg m**-2)',
    'Relative Humidity (%)', 'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
    'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)',
]

COUNTIES = [("17091", "KANKAKEE"), ("17155", "PUTNAM")]
STATE, STATE_NAME = "IL", "ILLINOIS"
GRIDS, IMG_SIZE = 2, 64


def _sentinel(root, year, grids, img_size, rng):
    for months in [(4, 5, 6), (7, 8, 9)]:
        start, end = f"{year}-{months[0]:02d}-01", f"{year}-{months[-1]:02d}-30"
        path = os.path.join(root, f"Sentinel-2 Imagery/data/{year}/{STATE}/Agriculture_17_{STATE}_{start}_{end}.h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as hf:
            for fips, county in COUNTIES:
                for m in months:
                    for day in ("01", "15"):
                        g = hf.create_group(f"{fips}/{year}-{m:02d}-{day}")
                        g["data"] = rng.integers(0, 255, (grids, img_size, img_size, 3), dtype=np.uint8)
                        g["coordinates"] = np.zeros((grids, 2, 2))
                        g["county"] = [county.encode()]
                        g["state"] = [STATE_NAME.encode()]


def _hrrr(root, year, grids, rng):
    for month in range(1, 13):
        rows = []
        for fips, county in COUNTIES:
            base = {"Year": year, "Month": month, "State": STATE_NAME, "County": county, "FIPS Code": int(fips)}
            for grid in range(grids):
                for day in range(1, 31):
                    rows.append({**base, "Day": float(day), "Daily/Monthly": "Daily", "Grid Index": float(grid),
                                 "Lat (llcrnr)": 41.0, "Lon (llcrnr)": -88.0, "Lat (urcrnr)": 41.1,
                                 "Lon (urcrnr)": -87.9, **dict(zip(WEATHER, rng.random(len(WEATHER))))})
            rows.append({**base, "Day": np.nan, "Daily/Monthly": "Monthly", "Grid Index": np.nan,
                         **dict(zip(WEATHER, rng.random(len(WEATHER))))})
        path = os.path.join(root, f"WRF-HRRR/data/{year}/{STATE}/HRRR_17_{STATE}_{year}-{month:02d}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def _usda(root, year, rng):
    rows = [{"commodity_desc": "SOYBEANS", "year": year, "state_ansi": fips[:2], "county_ansi": fips[2:],
             "county_name": county, "PRODUCTION, MEASURED IN BU": float(rng.integers(1e5, 1e7)),
             "YIELD, MEASURED IN BU / ACRE": float(rng.uniform(40, 65))} for fips, county in COUNTIES]
    path = os.path.join(root, f"USDA/data/Soybean/{year}/USDA_Soybean_County_{year}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_fake_dataset(root, train_year=2021, val_year=2022, grids=GRIDS, img_size=IMG_SIZE, long_term_years=3, seed=0):
    """Write a synthetic dataset under `root`; returns (train_index_path, val_index_path)."""
    from config.build_config_soybean import get_json_obj

    rng = np.random.default_rng(seed)
    for year in range(train_year - long_term_years, val_year + 1):
        _hrrr(root, year, grids, rng)
    for year in (train_year, val_year):
        _sentinel(root, year, grids, img_size, rng)
        _usda(root, year, rng)

    paths = []
    for split, year in (("train", train_year), ("val", val_year)):
        entries = [get_json_obj(year, {"FIPS": fips, "County": county, "State": STATE}, long_term_years)
                   for fips, county in COUNTIES]
        path = os.path.join(root, f"soybean_{split}.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        paths.append(path)
    return tuple(paths)
