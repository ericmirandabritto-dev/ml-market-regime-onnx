import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from market_regime_core import (
    build_feature_frame,
    chronological_split,
    export_regime_onnx,
    load_config,
    validate_onnx_model,
)
from walkforward import WalkForwardPolicy, evaluate_walkforward, resolve_purge_bars


def make_candles() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 1000 + rng.normal(0, 2, 1100).cumsum()
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": rng.integers(100, 1000, len(close)).astype(float),
        },
        index=pd.date_range("2025-01-01", periods=len(close), freq="5min", name="time"),
    )


class PortfolioTests(unittest.TestCase):
    def test_holdout_uses_raw_label_timestamps_even_when_feature_rows_are_missing(self):
        cfg = load_config()
        candles = make_candles()
        dataset, features = build_feature_frame(candles, cfg)
        dataset = dataset.drop(index=dataset.index[::17])
        train, test = chronological_split(dataset, cfg["train"]["test_size"])

        self.assertNotIn("label_end_time", features)
        self.assertTrue((train["label_end_time"] < test.index[0]).all())
        candidate_train = dataset.loc[dataset.index < test.index[0]]
        raw_end_times = candles.index.to_series().shift(-cfg["label"]["horizon_bars"])
        expected_index = candidate_train.index[raw_end_times.loc[candidate_train.index] < test.index[0]]
        pd.testing.assert_index_equal(train.index, expected_index)
        self.assertLess(len(train), len(candidate_train))

    def test_feature_builder_rejects_future_return_lags(self):
        cfg = load_config()
        cfg["feature"]["returns_lags"] = [-1, 2]
        with self.assertRaisesRegex(ValueError, "defasagens positivas"):
            build_feature_frame(make_candles(), cfg)

    def test_onnx_validation_fails_for_probability_drift(self):
        sample = make_candles().head(1)
        model = Mock()
        model.onnx_input_array.return_value = np.array([[1]], dtype=np.float32)
        model.predict_proba.return_value = np.array([[0.3, 0.5, 0.2]])
        session = Mock()
        session.get_inputs.return_value = [Mock(name="input")]
        session.get_outputs.return_value = [Mock(name="probabilities")]
        session.run.return_value = [np.array([[0.4, 0.4, 0.2]])]
        with patch("onnxruntime.InferenceSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "fora da tolerancia"):
                validate_onnx_model(model, Path("model.onnx"), sample, ["close"])

    def test_walkforward_requires_a_gap_at_least_as_long_as_the_label_horizon(self):
        cfg = load_config()
        horizon = cfg["label"]["horizon_bars"]
        self.assertEqual(resolve_purge_bars(cfg, "auto"), horizon)
        with self.assertRaisesRegex(ValueError, "horizonte"):
            resolve_purge_bars(cfg, horizon - 1)

    def test_failed_validation_preserves_previous_onnx(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            model_path.write_bytes(b"previous")
            with (
                patch("market_regime_core.export_onnx_model") as export,
                patch("market_regime_core.validate_onnx_model", side_effect=RuntimeError("paridade")),
            ):
                export.side_effect = lambda model, count, staged, opset: staged.write_bytes(b"candidate")
                with self.assertRaisesRegex(RuntimeError, "paridade"):
                    export_regime_onnx(None, model_path, 17, ["feature"], make_candles().head(1))
            self.assertEqual(model_path.read_bytes(), b"previous")

    def test_incremental_walkforward_retrains_if_same_timestamp_data_changes(self):
        cfg = load_config()
        candles = make_candles()
        policy = WalkForwardPolicy(
            train_window=400, validation_window=40, trade_window=100, step_window=150,
            purge_bars=resolve_purge_bars(cfg, "auto"), min_folds=1, max_folds=1,
            min_class_count_per_train=1, objective="balanced_accuracy_stable",
        )
        with TemporaryDirectory() as directory:
            def run():
                dataset, features = build_feature_frame(candles, cfg)
                return evaluate_walkforward(
                    candles, dataset, features, cfg, policy,
                    export_dir=Path(directory), incremental_export=True,
                )["aggregate"]

            self.assertEqual(run()["trained_folds"], 1)
            self.assertEqual(run()["reused_folds"], 1)
            candles.loc[candles.index[-1] + pd.Timedelta(minutes=5)] = candles.iloc[-1]
            self.assertEqual(run()["reused_folds"], 1)
            timestamp = candles.index[500]
            candles.loc[timestamp, ["open", "high", "low", "close"]] += 1
            self.assertEqual(run()["trained_folds"], 1)

    def test_cli_trains_and_validates_onnx_without_a_yaml_file(self):
        with TemporaryDirectory() as directory:
            checkout = Path(directory)
            candles = checkout / "candles.csv"
            artifacts = checkout / "artifacts"
            make_candles().to_csv(candles)
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT / "src/train_market_regime_onnx.py"),
                 "--input", str(candles), "--artifacts", str(artifacts), "--validate-onnx"],
                cwd=checkout, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
            parity = metadata["onnx_validation"]
            self.assertEqual(parity["status"], "ok")
            self.assertEqual(parity["rows_checked"], 32)
            self.assertLessEqual(parity["max_abs_diff_vs_reference"], parity["max_abs_diff_tolerance"])
            self.assertTrue((artifacts / "model.onnx").is_file())
            self.assertTrue((artifacts / "runtime_contract.csv").is_file())


if __name__ == "__main__":
    unittest.main()
