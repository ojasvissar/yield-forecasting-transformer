"""End-to-end: pre-train -> fine-tune -> evaluate on the synthetic dataset, on CPU."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, *args):
    proc = subprocess.run([sys.executable, script, "--device", "cpu", "--num_workers", "0", "--batch_size", "8",
                           *args], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


@pytest.fixture(scope="module")
def pretrained(fake_dataset, tmp_path_factory):
    root, train, _ = fake_dataset
    out = str(tmp_path_factory.mktemp("pvt_simclr"))
    run("main_pretrain_mmst_vit.py", "--root_dir", root, "--data_file", train,
        "--output_dir", out, "--log_dir", out, "--epochs", "1")
    return out


@pytest.fixture(scope="module")
def finetuned(fake_dataset, pretrained, tmp_path_factory):
    root, train, val = fake_dataset
    out = str(tmp_path_factory.mktemp("mmst_vit"))
    run("main_finetune_mmst_vit.py", "--root_dir", root, "--data_file_train", train, "--data_file_val", val,
        "--output_dir", out, "--log_dir", out, "--epochs", "2", "--save_freq", "1",
        "--pvt_simclr", os.path.join(pretrained, "checkpoint-0.pth"))
    return out


def test_pretraining_writes_checkpoint_and_log(pretrained):
    assert os.path.exists(os.path.join(pretrained, "checkpoint-0.pth"))
    with open(os.path.join(pretrained, "log.txt")) as f:
        assert "train_loss" in json.loads(f.readline())


def test_finetuning_writes_metrics_and_predictions(finetuned):
    with open(os.path.join(finetuned, "metrics.txt")) as f:
        rows = [json.loads(line) for line in f]
    assert [r["epoch"] for r in rows] == [0, 1]
    assert all({"rmse", "r2", "corr"} <= r.keys() for r in rows)

    pred = pd.read_csv(os.path.join(finetuned, "predictions.csv"), dtype={"fips": str})
    assert sorted(pred.fips) == ["17091", "17155"]
    assert (pred.true_yield > 0).all() and (pred.pred_yield > 0).all()


def test_eval_scores_the_validation_set(fake_dataset, finetuned):
    root, train, val = fake_dataset
    stdout = run("main_finetune_mmst_vit.py", "--root_dir", root, "--data_file_train", train,
                 "--data_file_val", val, "--output_dir", finetuned, "--log_dir", finetuned,
                 "--eval", "--resume", os.path.join(finetuned, "checkpoint-1.pth"))
    with open(os.path.join(finetuned, "metrics.txt")) as f:
        rows = [json.loads(line) for line in f]
    # same checkpoint, same validation data -> same RMSE as the epoch-1 evaluation during training
    assert rows[-1]["epoch"] is None
    assert rows[-1]["rmse"] == pytest.approx(rows[1]["rmse"], rel=1e-4)
    assert "Metrics: RMSE" in stdout
