#!/usr/bin/env python
"""
Build the soybean data index files (data/soybean_{train,val}.json).

Each entry lists, for one county-year, the Sentinel-2 files for the growing
season, the daily HRRR files for April-September, the monthly HRRR files for
the `long_term_years` years *before* the prediction year (the paper uses the
previous three years), and the USDA file with the target yield.

Usage (from the repository root):
    python -m config.build_config_soybean
    python -m config.build_config_soybean --long_term_years 3 --fips 17091 19117
"""

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the 10 counties used by the default index
DEFAULT_FIPS = ["22007", "22121", "22043", "22107", "28089", "28015", "17091", "17155", "19117", "19135"]


def build_index(year, csv_path, out_path, target_fips=None, long_term_years=3):
    df = pd.read_csv(csv_path)

    if target_fips:
        df = df[df["FIPS"].isin([int(f) for f in target_fips])]

    counties = json.loads(df.to_json(orient='records', lines=False))
    data = [get_json_obj(year, county_info, long_term_years) for county_info in counties]

    with open(out_path, "w") as write_file:
        json.dump(data, write_file)
    return data


def build_soybean_train(target_fips=None, long_term_years=3):
    return build_index(2021, os.path.join(ROOT, "input/county_info_2021.csv"),
                       os.path.join(ROOT, "data/soybean_train.json"), target_fips, long_term_years)


def build_soybean_val(target_fips=None, long_term_years=3):
    return build_index(2022, os.path.join(ROOT, "input/county_info_2022.csv"),
                       os.path.join(ROOT, "data/soybean_val.json"), target_fips, long_term_years)


def get_json_obj(year, county_info, long_term_years=3):
    fips, county, state = str(county_info["FIPS"]), county_info["County"], county_info["State"]

    short_term = get_short_HRRR_obj(state, fips, year, months=[i for i in range(4, 10)])

    # only years before the prediction year: the current year's monthly data would leak the target season
    long_term_years = [str(y) for y in range(year - long_term_years, year)]
    long_term = get_long_HRRR_obj(state, fips, years=long_term_years)

    obj = {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state,
        "county_ansi": fips[2:],
        "state_ansi": fips[:2],
        "data": {
            "HRRR": {
                "short_term": short_term,
                "long_term": long_term,
            },
            "USDA": "USDA/data/Soybean/{}/USDA_Soybean_County_{}.csv".format(year, year),
            "sentinel": [
                "Sentinel-2 Imagery/data/{}/{}/Agriculture_{}_{}_{}-04-01_{}-06-30.h5".format(year, state, fips[:2],
                                                                                              state, year, year),
                "Sentinel-2 Imagery/data/{}/{}/Agriculture_{}_{}_{}-07-01_{}-09-30.h5".format(year, state, fips[:2],
                                                                                              state, year, year),
            ]
        }

    }

    return obj


def get_short_HRRR_obj(state, fips, year, months=[i + 1 for i in range(12)]):
    file_paths = []
    for month in months:
        month = str(month).zfill(2)
        path = "WRF-HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(year, state, fips[:2], state, year, month)
        file_paths.append(path)
    return file_paths


def get_long_HRRR_obj(state, fips, years, months=[i + 1 for i in range(12)]):
    file_paths = []
    for year in years:
        year_paths = []
        for month in months:
            month = str(month).zfill(2)
            path = "WRF-HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(year, state, fips[:2], state, year, month)
            year_paths.append(path)
        file_paths.append(year_paths)
    return file_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build soybean data index files')
    parser.add_argument('--fips', nargs='+', default=DEFAULT_FIPS, help='County FIPS codes to include')
    parser.add_argument('--long_term_years', type=int, default=3,
                        help='Number of years before the prediction year used as long-term climate input')
    args = parser.parse_args()

    for build in (build_soybean_train, build_soybean_val):
        data = build(target_fips=args.fips, long_term_years=args.long_term_years)
        print(f"{build.__name__}: {len(data)} counties")
