import json
import math
import pickle

import torch

from config.build_config_soybean import get_json_obj
from dataset.hrrr_loader import HRRRDataset
from dataset.sentinel_loader import SentinelDataset
from dataset.sentinel_wrapper import get_data_loader
from dataset.usda_loader import USDADataset
from tests.fake_data import COUNTIES, GRIDS, IMG_SIZE


def test_long_term_window_uses_only_previous_years():
    entry = get_json_obj(2021, {"FIPS": 17091, "County": "KANKAKEE", "State": "IL"}, long_term_years=3)
    years = [paths[0].split("/")[2] for paths in entry["data"]["HRRR"]["long_term"]]
    assert years == ["2018", "2019", "2020"]
    assert all(len(paths) == 12 for paths in entry["data"]["HRRR"]["long_term"])


def test_repo_index_files_do_not_leak_the_prediction_year():
    for path in ("data/soybean_train.json", "data/soybean_val.json"):
        with open(path) as f:
            for entry in json.load(f):
                years = {int(p[0].split("/")[2]) for p in entry["data"]["HRRR"]["long_term"]}
                assert max(years) < entry["year"], (path, entry["FIPS"])


def test_sentinel_loader_keeps_first_of_month_images(fake_dataset):
    root, train, _ = fake_dataset
    x, fips, year = SentinelDataset(root, train)[0]
    # April-September, 1st of each month only (the files also hold the 15th)
    assert tuple(x.shape) == (6, GRIDS, IMG_SIZE, IMG_SIZE, 3)
    assert (fips, year) == (COUNTIES[0][0], 2021)


def test_hrrr_loader_shapes_and_pickling(fake_dataset):
    root, train, _ = fake_dataset
    dataset = pickle.loads(pickle.dumps(HRRRDataset(root, train)))  # must survive DataLoader workers
    x_short, x_long, fips, year = dataset[0]
    assert tuple(x_short.shape) == (6, GRIDS, 28, 9)
    assert tuple(x_long.shape) == (3, 12, 9)
    assert x_short.dtype == torch.float32


def test_usda_loader_returns_log_production_and_yield(fake_dataset):
    root, _, val = fake_dataset
    z, fips, year = USDADataset(root, val)[0]
    assert z.shape == (2,)
    assert 40 <= math.exp(z[1]) <= 65  # yield in bu/acre


def test_simclr_wrapper_yields_two_views_and_weather():
    x = torch.randint(0, 255, (1, 6, 2, 64, 64, 3), dtype=torch.uint8)
    ys = torch.randn(1, 6, 2, 28, 9)
    xi, xj, y = next(iter(get_data_loader(x, ys, batch_size=4)))
    assert xi.shape == xj.shape == (4, 3, 224, 224)
    assert y.shape == (4, 28, 9)
