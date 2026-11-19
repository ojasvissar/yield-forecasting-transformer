import pytest

from tests.fake_data import make_fake_dataset


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    """(root_dir, train_index, val_index) of a tiny synthetic dataset shared by all tests."""
    root = str(tmp_path_factory.mktemp("tinycropnet"))
    train, val = make_fake_dataset(root)
    return root, train, val
