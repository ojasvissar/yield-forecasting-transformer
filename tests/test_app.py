import json
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def run_app(root_dir, output_dir):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.text_input[0].set_value(str(root_dir))
    at.sidebar.text_input[1].set_value(str(output_dir)).run()
    assert not at.exception, [e.value for e in at.exception]
    return {m.label: m.value for m in at.metric}


def test_dashboard_renders_without_any_data(tmp_path):
    metrics = run_app(tmp_path / "no-data", tmp_path / "no-runs")
    assert metrics["Sentinel-2 files"] == "0 / 16"
    assert metrics["Fine-tune ckpts"] == "0"


def test_dashboard_shows_finished_run(tmp_path):
    run_dir = tmp_path / "runs" / "mmst_vit"
    run_dir.mkdir(parents=True)
    (run_dir / "log.txt").write_text("".join(json.dumps({"train_loss": 1 / (e + 1), "epoch": e}) + "\n"
                                             for e in range(3)))
    (run_dir / "metrics.txt").write_text(json.dumps({"rmse": 4.2, "r2": 0.5, "corr": 0.7, "epoch": 2}) + "\n")
    (run_dir / "predictions.csv").write_text("epoch,fips,true_yield,pred_yield\n2,17091,59.7,55.1\n2,19117,43.1,47.0\n")

    metrics = run_app(tmp_path / "no-data", tmp_path / "runs")
    assert metrics["Epochs logged"] == "3"
    assert metrics["RMSE (bu/acre)"] == "4.20"
    assert metrics["Eval metrics"] == "logged"
