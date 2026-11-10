from __future__ import annotations

import csv
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
LABEL_TO_ID = {"bear": 0, "range": 1, "bull": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
FEATURE_SCHEMA_VERSION = 1
SKLEARN_MODEL_TYPES = {"logistic_regression", "random_forest"}
DEEP_TABULAR_MODEL_TYPES = {"deep_mlp"}
SEQUENCE_MODEL_TYPES = {"lstm", "cnn", "transformer"}
MODEL_TYPE_ALIASES = {
    "mlp": "deep_mlp",
    "deep_network": "deep_mlp",
    "cnn1d": "cnn",
    "transformer_encoder": "transformer",
}


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_candles_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {missing}")

    # Dados exportados do MT5 geralmente usam horario do servidor e nao trazem offset.
    # Evitamos forcar UTC para nao reinterpretar timestamps do broker.
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    return df[["open", "high", "low", "close", "volume"]].astype(float)


def build_feature_frame(candles: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    feature_cfg = cfg["feature"]
    label_cfg = cfg["label"]

    returns_lags = [int(value) for value in feature_cfg["returns_lags"]]
    atr_window = int(feature_cfg["atr_window"])
    vol_window = int(feature_cfg["vol_window"])
    ema_fast = int(feature_cfg["ema_fast"])
    ema_slow = int(feature_cfg["ema_slow"])
    horizon_bars = int(label_cfg["horizon_bars"])
    trend_threshold_atr = float(label_cfg["trend_threshold_atr"])

    if ema_fast >= ema_slow:
        raise ValueError("feature.ema_fast deve ser menor que feature.ema_slow")

    df = candles.copy()
    close_safe = df["close"].replace(0.0, np.nan)

    for lag in returns_lags:
        df[f"ret_{lag}"] = df["close"].pct_change(lag)

    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["range_pct"] = (df["high"] - df["low"]) / close_safe
    df["body_pct"] = body / close_safe
    df["upper_wick_pct"] = upper_wick / close_safe
    df["lower_wick_pct"] = lower_wick / close_safe

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_norm"] = tr.rolling(atr_window).mean() / close_safe

    ret_1 = df["close"].pct_change(1)
    df["realized_vol"] = ret_1.rolling(vol_window).std()

    vol_mean = df["volume"].rolling(vol_window).mean()
    vol_std = df["volume"].rolling(vol_window).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    ema_fast_series = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["ema_fast_gap"] = df["close"] / ema_fast_series - 1.0
    df["ema_slow_gap"] = df["close"] / ema_slow_series - 1.0
    df["ema_spread"] = ema_fast_series / ema_slow_series - 1.0

    df["future_return"] = df["close"].shift(-horizon_bars) / df["close"] - 1.0
    df["future_move_atr"] = df["future_return"] / df["atr_norm"].replace(0.0, np.nan)

    df["regime_id"] = LABEL_TO_ID["range"]
    df.loc[df["future_move_atr"] >= trend_threshold_atr, "regime_id"] = LABEL_TO_ID["bull"]
    df.loc[df["future_move_atr"] <= -trend_threshold_atr, "regime_id"] = LABEL_TO_ID["bear"]
    df["regime_name"] = df["regime_id"].map(ID_TO_LABEL)

    feature_columns = (
        [f"ret_{lag}" for lag in returns_lags]
        + [
            "range_pct",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "atr_norm",
            "realized_vol",
            "vol_z",
            "ema_fast_gap",
            "ema_slow_gap",
            "ema_spread",
        ]
    )

    required = feature_columns + ["future_return", "future_move_atr", "regime_id", "regime_name"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return df, feature_columns


def chronological_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.05 <= test_size < 0.5:
        raise ValueError("train.test_size deve ficar entre 0.05 e 0.50")

    test_rows = max(1, int(len(df) * test_size))
    train = df.iloc[:-test_rows].copy()
    test = df.iloc[-test_rows:].copy()

    if train.empty or test.empty:
        raise ValueError("Split gerou treino ou teste vazio")

    ok, issues = validate_train_classes(train)
    if not ok:
        raise ValueError(" ".join(issues))

    return train, test


def class_distribution(series: pd.Series) -> Dict[str, int]:
    return {
        ID_TO_LABEL[int(label_id)]: int(count)
        for label_id, count in series.astype(int).value_counts().sort_index().items()
    }


def dataset_diagnostics(candles: pd.DataFrame, dataset: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    diagnostics = {
        "candles_rows": int(len(candles)),
        "feature_rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "candles_start": str(candles.index.min()) if len(candles) else None,
        "candles_end": str(candles.index.max()) if len(candles) else None,
        "train_class_distribution": class_distribution(train_df["regime_id"]),
        "test_class_distribution": class_distribution(test_df["regime_id"]),
        "warnings": [],
    }

    if len(dataset) < 5000:
        diagnostics["warnings"].append(
            "Poucas amostras para regime de mercado. Recomendado: pelo menos 5000 a 20000 candles, idealmente mais."
        )
    if len(train_df) < 1000:
        diagnostics["warnings"].append("Treino com menos de 1000 amostras; metricas podem ser instaveis.")
    if train_df["regime_id"].value_counts().min() < 100:
        diagnostics["warnings"].append("Alguma classe tem menos de 100 amostras no treino; aumente historico ou ajuste o threshold.")

    return diagnostics


def validate_train_classes(train_df: pd.DataFrame, min_class_count: int = 1) -> Tuple[bool, List[str]]:
    present_counts = train_df["regime_id"].astype(int).value_counts().sort_index()
    present_classes = set(int(label_id) for label_id in present_counts.index.tolist())
    expected_classes = set(LABEL_TO_ID.values())

    issues: List[str] = []
    missing_classes = sorted(expected_classes - present_classes)
    if missing_classes:
        missing_names = [ID_TO_LABEL[class_id] for class_id in missing_classes]
        issues.append(
            "Treino sem todas as classes necessarias. "
            f"Ajuste o dataset ou reduza label.trend_threshold_atr. Faltando: {missing_names}"
        )

    low_count_classes = [
        ID_TO_LABEL[int(label_id)]
        for label_id, count in present_counts.items()
        if int(count) < int(min_class_count)
    ]
    if low_count_classes:
        issues.append(
            "Treino com classes abaixo do minimo exigido "
            f"({min_class_count}): {low_count_classes}"
        )

    return (len(issues) == 0), issues


def feature_warmup_bars(cfg: Dict) -> int:
    feature_cfg = cfg["feature"]
    returns_lags = [int(value) for value in feature_cfg["returns_lags"]]
    windows = [
        max(returns_lags) if returns_lags else 1,
        int(feature_cfg["atr_window"]),
        int(feature_cfg["vol_window"]),
        int(feature_cfg["ema_slow"]),
    ]
    return int(max(windows))


def export_market_runtime_contract(
    destination: Path,
    cfg: Dict,
    feature_columns: Sequence[str],
    onnx_contract: Dict,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("train", {})
    feature_cfg = cfg.get("feature", {})
    model_type = normalize_model_type(train_cfg.get("model_type", "logistic_regression"))
    input_name = str(onnx_contract.get("input_name", "features"))
    input_rank = 3 if input_name == "sequence" or model_requires_sequence(model_type) else 2
    lookback_bars = int(train_cfg.get("lookback_bars", 32)) if input_rank == 3 else 1
    returns_lags = [int(value) for value in feature_cfg.get("returns_lags", [1, 2, 3, 5, 10])]

    rows = [
        ("key", "value"),
        ("project", "Market Regime ONNX"),
        ("feature_schema_version", str(FEATURE_SCHEMA_VERSION)),
        ("model_type", model_type),
        ("input_name", input_name),
        ("input_rank", str(input_rank)),
        ("lookback_bars", str(lookback_bars)),
        ("feature_count", str(len(feature_columns))),
        ("label_count", str(len(ID_TO_LABEL))),
        ("returns_lag_count", str(len(returns_lags))),
        ("atr_window", str(int(feature_cfg["atr_window"]))),
        ("vol_window", str(int(feature_cfg["vol_window"]))),
        ("ema_fast", str(int(feature_cfg["ema_fast"]))),
        ("ema_slow", str(int(feature_cfg["ema_slow"]))),
    ]

    for index, lag in enumerate(returns_lags):
        rows.append((f"returns_lag_{index}", str(int(lag))))

    for index, feature_name in enumerate(feature_columns):
        rows.append((f"feature_{index}", str(feature_name)))

    for index in sorted(ID_TO_LABEL):
        rows.append((f"label_{index}", ID_TO_LABEL[index]))

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def normalize_model_type(raw_model_type: str | None) -> str:
    model_type = str(raw_model_type or "logistic_regression").strip().lower()
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def model_requires_sequence(model_type: str) -> bool:
    return normalize_model_type(model_type) in SEQUENCE_MODEL_TYPES


def parse_hidden_sizes(raw_value, default: Sequence[int] = (128, 64)) -> Tuple[int, ...]:
    if raw_value is None:
        return tuple(int(value) for value in default)

    if isinstance(raw_value, str):
        values = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
        return tuple(int(value) for value in values) if values else tuple(int(value) for value in default)

    if isinstance(raw_value, (list, tuple)):
        values = [int(value) for value in raw_value]
        return tuple(values) if values else tuple(int(value) for value in default)

    raise ValueError("train.mlp_hidden_sizes deve ser lista ou string separada por virgulas")


def build_sklearn_pipeline(cfg: Dict) -> Pipeline:
    train_cfg = cfg["train"]
    model_type = normalize_model_type(train_cfg.get("model_type", "logistic_regression"))

    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=int(train_cfg.get("max_iter", 1500)),
                        C=float(train_cfg.get("C", 1.0)),
                        class_weight="balanced",
                        random_state=int(train_cfg.get("random_state", 42)),
                    ),
                ),
            ]
        )

    if model_type == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(train_cfg.get("n_estimators", 300)),
                        max_depth=int(train_cfg.get("max_depth", 8)),
                        min_samples_leaf=int(train_cfg.get("min_samples_leaf", 10)),
                        class_weight="balanced_subsample",
                        random_state=int(train_cfg.get("random_state", 42)),
                        n_jobs=int(train_cfg.get("n_jobs", 1)),
                    ),
                ),
            ]
        )

    raise ValueError(f"Modelo sklearn nao suportado: {model_type}")


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Modelos profundos exigem torch no interpretador atual. Instale a dependencia antes de usar train.model_type=deep_mlp."
        ) from exc

    return torch, nn, DataLoader, TensorDataset


def _build_torch_mlp_module(nn, input_size: int, hidden_sizes: Sequence[int], class_count: int, dropout: float):
    layers = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(previous_size, int(hidden_size)))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous_size = int(hidden_size)
    layers.append(nn.Linear(previous_size, class_count))

    class TorchMLPModule(nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.network = nn.Sequential(*seq)

        def forward(self, features):
            return self.network(features)

    return TorchMLPModule(layers)


def _build_torch_export_wrapper(nn, torch, base_model, medians, means, scales):
    class TorchTabularExportWrapper(nn.Module):
        def __init__(self, model, fill_values, feature_means, feature_scales):
            super().__init__()
            self.model = model
            self.register_buffer("fill_values", torch.tensor(fill_values, dtype=torch.float32))
            self.register_buffer("feature_means", torch.tensor(feature_means, dtype=torch.float32))
            self.register_buffer("feature_scales", torch.tensor(feature_scales, dtype=torch.float32))

        def forward(self, features):
            fill_values = self.fill_values.unsqueeze(0).expand_as(features)
            clean_features = torch.where(torch.isnan(features), fill_values, features)
            scaled_features = (clean_features - self.feature_means) / self.feature_scales
            logits = self.model(scaled_features)
            probabilities = torch.softmax(logits, dim=1)
            labels = torch.argmax(probabilities, dim=1)
            return labels, probabilities

    return TorchTabularExportWrapper(base_model, medians, means, scales)


def _build_torch_sequence_export_wrapper(nn, torch, base_model, medians, means, scales):
    class TorchSequenceExportWrapper(nn.Module):
        def __init__(self, model, fill_values, feature_means, feature_scales):
            super().__init__()
            self.model = model
            self.register_buffer("fill_values", torch.tensor(fill_values, dtype=torch.float32).view(1, 1, -1))
            self.register_buffer("feature_means", torch.tensor(feature_means, dtype=torch.float32).view(1, 1, -1))
            self.register_buffer("feature_scales", torch.tensor(feature_scales, dtype=torch.float32).view(1, 1, -1))

        def forward(self, sequence):
            fill_values = self.fill_values.expand_as(sequence)
            clean_sequence = torch.where(torch.isnan(sequence), fill_values, sequence)
            scaled_sequence = (clean_sequence - self.feature_means) / self.feature_scales
            logits = self.model(scaled_sequence)
            probabilities = torch.softmax(logits, dim=1)
            labels = torch.argmax(probabilities, dim=1)
            return labels, probabilities

    return TorchSequenceExportWrapper(base_model, medians, means, scales)


def _ensure_enough_rows_for_sequence(row_count: int, lookback_bars: int, context: str) -> None:
    if row_count < lookback_bars:
        raise ValueError(
            f"{context} insuficiente para modelo sequencial: necessario pelo menos {lookback_bars} rows, recebido {row_count}."
        )


def _build_sequence_windows_from_matrix(
    matrix: np.ndarray,
    lookback_bars: int,
    start_index: int,
    target_count: int,
) -> np.ndarray:
    if start_index < lookback_bars - 1:
        raise ValueError(
            f"Contexto insuficiente para montar janelas sequenciais: start_index={start_index}, lookback_bars={lookback_bars}."
        )
    windows = [
        matrix[(end_index - lookback_bars + 1) : (end_index + 1)]
        for end_index in range(start_index, start_index + target_count)
    ]
    return np.stack(windows).astype(np.float32, copy=False)


def _build_lstm_sequence_module(nn, input_size: int, hidden_size: int, num_layers: int, class_count: int, dropout: float):
    effective_dropout = dropout if int(num_layers) > 1 else 0.0

    class TorchLSTMModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=effective_dropout,
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.head = nn.Linear(int(hidden_size), class_count)

        def forward(self, sequence):
            _output, (hidden_state, _cell_state) = self.encoder(sequence)
            embedding = self.dropout(hidden_state[-1])
            return self.head(embedding)

    return TorchLSTMModule()


def _build_cnn_sequence_module(nn, input_size: int, channels: int, kernel_size: int, class_count: int, dropout: float):
    padding = max(0, int(kernel_size) // 2)

    class TorchCNNModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_size, int(channels), kernel_size=int(kernel_size), padding=padding),
                nn.ReLU(),
                nn.Conv1d(int(channels), int(channels), kernel_size=int(kernel_size), padding=padding),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.head = nn.Linear(int(channels), class_count)

        def forward(self, sequence):
            channels_first = sequence.transpose(1, 2)
            encoded = self.network(channels_first)
            pooled = self.pool(encoded).squeeze(-1)
            pooled = self.dropout(pooled)
            return self.head(pooled)

    return TorchCNNModule()


def _build_transformer_sequence_module(
    torch,
    nn,
    input_size: int,
    lookback_bars: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    class_count: int,
    dropout: float,
):
    class TorchTransformerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, int(d_model))
            self.position_embedding = nn.Parameter(
                nn.init.normal_(torch.empty(1, int(lookback_bars), int(d_model)), mean=0.0, std=0.02)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
            self.norm = nn.LayerNorm(int(d_model))
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.head = nn.Linear(int(d_model), class_count)

        def forward(self, sequence):
            encoded = self.input_proj(sequence) + self.position_embedding[:, : sequence.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = encoded.mean(dim=1)
            pooled = self.dropout(self.norm(pooled))
            return self.head(pooled)

    return TorchTransformerModule()


class BaseRegimeModel(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model_type = normalize_model_type(cfg.get("train", {}).get("model_type", "logistic_regression"))
        self._class_ids = [LABEL_TO_ID["bear"], LABEL_TO_ID["range"], LABEL_TO_ID["bull"]]
        self._feature_columns: List[str] = []

    @property
    def class_ids(self) -> List[int]:
        return [int(class_id) for class_id in self._class_ids]

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_columns)

    @property
    def history_rows_needed(self) -> int:
        return 0

    def _remember_feature_columns(self, feature_columns: Sequence[str]) -> None:
        self._feature_columns = [str(column) for column in feature_columns]

    def _validate_feature_columns(self, feature_columns: Sequence[str]) -> None:
        incoming = [str(column) for column in feature_columns]
        if self._feature_columns and incoming != self._feature_columns:
            raise ValueError(
                "Ordem das features divergente da usada no treino. "
                f"Esperado: {self._feature_columns} | recebido: {incoming}"
            )

    def onnx_input_array(
        self,
        sample_df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        self._validate_feature_columns(feature_columns)
        return sample_df[list(feature_columns)].to_numpy(dtype=np.float32)

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def export_onnx(self, output_path: Path, opset: int, feature_count: int | None = None) -> Dict[str, List[str] | str]:
        raise NotImplementedError


class SklearnPipelineRegimeModel(BaseRegimeModel):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.pipeline = build_sklearn_pipeline(cfg)

    def fit(self, train_df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        self._remember_feature_columns(feature_columns)
        self.pipeline.fit(train_df[list(feature_columns)], train_df["regime_id"].astype(int))
        classifier = self.pipeline.named_steps["classifier"]
        self._class_ids = [int(class_id) for class_id in classifier.classes_]

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        self._validate_feature_columns(feature_columns)
        return self.pipeline.predict(df[list(feature_columns)])

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        self._validate_feature_columns(feature_columns)
        return self.pipeline.predict_proba(df[list(feature_columns)])

    def export_onnx(self, output_path: Path, opset: int, feature_count: int | None = None) -> Dict[str, List[str] | str]:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise RuntimeError(
                "Exportacao ONNX indisponivel: instale skl2onnx no interpretador atual."
            ) from exc

        if feature_count is None:
            feature_count = len(self._feature_columns)
        classifier = self.pipeline.named_steps["classifier"]
        onnx_model = convert_sklearn(
            self.pipeline,
            initial_types=[("features", FloatTensorType([None, feature_count]))],
            target_opset=opset,
            options={classifier.__class__: {"zipmap": False}},
        )
        output_path.write_bytes(onnx_model.SerializeToString())
        return {
            "input_name": onnx_model.graph.input[0].name,
            "output_names": [output.name for output in onnx_model.graph.output],
        }


class TorchTabularRegimeModel(BaseRegimeModel):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        train_cfg = cfg.get("train", {})
        self.hidden_sizes = parse_hidden_sizes(train_cfg.get("mlp_hidden_sizes"), default=(128, 64))
        self.dropout = float(train_cfg.get("dropout", 0.10))
        self.epochs = int(train_cfg.get("epochs", 25))
        self.batch_size = int(train_cfg.get("batch_size", 256))
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        self.device_name = str(train_cfg.get("device", "cpu"))
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._torch_model = None
        self._torch = None
        self._device = None
        self._input_size = 0

    def _resolve_device(self, torch):
        requested = str(self.device_name).strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_training_features(self, train_df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
        raw_features = train_df[list(feature_columns)].to_numpy(dtype=np.float32)
        filled_features = self._imputer.fit_transform(raw_features)
        scaled_features = self._scaler.fit_transform(filled_features)
        return scaled_features.astype(np.float32, copy=False)

    def _transform_features(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
        self._validate_feature_columns(feature_columns)
        raw_features = df[list(feature_columns)].to_numpy(dtype=np.float32)
        filled_features = self._imputer.transform(raw_features)
        scaled_features = self._scaler.transform(filled_features)
        return scaled_features.astype(np.float32, copy=False)

    def fit(self, train_df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        torch, nn, DataLoader, TensorDataset = _require_torch()
        self._remember_feature_columns(feature_columns)
        self._torch = torch
        self._device = self._resolve_device(torch)

        x_train = self._prepare_training_features(train_df, feature_columns)
        y_train = train_df["regime_id"].astype(np.int64).to_numpy(copy=True)
        self._input_size = int(x_train.shape[1])

        model = _build_torch_mlp_module(
            nn=nn,
            input_size=self._input_size,
            hidden_sizes=self.hidden_sizes,
            class_count=len(LABEL_TO_ID),
            dropout=self.dropout,
        ).to(self._device)

        counts = np.bincount(y_train, minlength=len(LABEL_TO_ID)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self._device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        data_loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, len(dataset))),
            shuffle=True,
        )

        model.train()
        for _epoch in range(max(1, self.epochs)):
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self._device)
                batch_targets = batch_targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

        model.eval()
        self._torch_model = model
        self._class_ids = [LABEL_TO_ID["bear"], LABEL_TO_ID["range"], LABEL_TO_ID["bull"]]

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(df, feature_columns, history_df=history_df)
        return probabilities.argmax(axis=1).astype(np.int64, copy=False)

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if self._torch_model is None or self._torch is None or self._device is None:
            raise RuntimeError("Modelo deep_mlp ainda nao foi treinado")

        x_test = self._transform_features(df, feature_columns)
        if len(x_test) == 0:
            return np.zeros((0, len(LABEL_TO_ID)), dtype=np.float64)

        batch_size = max(1, min(self.batch_size, len(x_test)))
        probabilities: List[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                end = start + batch_size
                batch = self._torch.tensor(x_test[start:end], dtype=self._torch.float32, device=self._device)
                logits = self._torch_model(batch)
                batch_probabilities = self._torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(batch_probabilities.astype(np.float64, copy=False))

        return np.vstack(probabilities)

    def export_onnx(self, output_path: Path, opset: int, feature_count: int | None = None) -> Dict[str, List[str] | str]:
        torch, nn, _DataLoader, _TensorDataset = _require_torch()
        if self._torch_model is None:
            raise RuntimeError("Modelo deep_mlp ainda nao foi treinado")

        if feature_count is None:
            feature_count = self._input_size
        model_cpu = self._torch_model.to("cpu").eval()
        fill_values = self._imputer.statistics_.astype(np.float32, copy=False)
        feature_means = self._scaler.mean_.astype(np.float32, copy=False)
        feature_scales = np.where(self._scaler.scale_ == 0.0, 1.0, self._scaler.scale_).astype(np.float32, copy=False)
        wrapper = _build_torch_export_wrapper(
            nn=nn,
            torch=torch,
            base_model=model_cpu,
            medians=fill_values,
            means=feature_means,
            scales=feature_scales,
        ).eval()

        sample_input = torch.zeros((1, int(feature_count)), dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            sample_input,
            output_path.as_posix(),
            input_names=["features"],
            output_names=["label", "probabilities"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "label": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
            opset_version=int(opset),
            dynamo=False,
        )
        return {
            "input_name": "features",
            "output_names": ["label", "probabilities"],
        }


class TorchSequenceRegimeModel(BaseRegimeModel):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        train_cfg = cfg.get("train", {})
        self.lookback_bars = int(train_cfg.get("lookback_bars", 32))
        self.dropout = float(train_cfg.get("dropout", 0.10))
        self.epochs = int(train_cfg.get("epochs", 25))
        self.batch_size = int(train_cfg.get("batch_size", 256))
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))
        self.device_name = str(train_cfg.get("device", "cpu"))
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self._torch_model = None
        self._torch = None
        self._device = None
        self._input_size = 0

    @property
    def history_rows_needed(self) -> int:
        return max(0, self.lookback_bars - 1)

    def _resolve_device(self, torch):
        requested = str(self.device_name).strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_training_sequences(
        self,
        train_df: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_features = train_df[list(feature_columns)].to_numpy(dtype=np.float32)
        filled_features = self._imputer.fit_transform(raw_features)
        scaled_features = self._scaler.fit_transform(filled_features).astype(np.float32, copy=False)
        y_train = train_df["regime_id"].astype(np.int64).to_numpy(copy=True)
        _ensure_enough_rows_for_sequence(len(scaled_features), self.lookback_bars, "Treino")

        sequence_targets = y_train[self.lookback_bars - 1 :]
        sequence_features = _build_sequence_windows_from_matrix(
            matrix=scaled_features,
            lookback_bars=self.lookback_bars,
            start_index=self.lookback_bars - 1,
            target_count=len(sequence_targets),
        )
        return sequence_features, sequence_targets

    def _transform_inference_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        self._validate_feature_columns(feature_columns)
        history_rows = int(self.history_rows_needed)
        if history_rows <= 0:
            raw_features = df[list(feature_columns)].to_numpy(dtype=np.float32)
            filled_features = self._imputer.transform(raw_features)
            scaled_features = self._scaler.transform(filled_features).astype(np.float32, copy=False)
            return scaled_features[:, None, :]

        history_slice = (
            history_df.iloc[-history_rows:].copy()
            if history_df is not None and not history_df.empty
            else df.iloc[0:0].copy()
        )
        if len(history_slice) < history_rows:
            raise ValueError(
                "Historico insuficiente para inferencia sequencial. "
                f"Necessario {history_rows} rows anteriores, recebido {len(history_slice)}."
            )
        combined_df = pd.concat([history_slice, df], axis=0)
        _ensure_enough_rows_for_sequence(len(combined_df), self.lookback_bars, "Inferencia")

        raw_features = combined_df[list(feature_columns)].to_numpy(dtype=np.float32)
        filled_features = self._imputer.transform(raw_features)
        scaled_features = self._scaler.transform(filled_features).astype(np.float32, copy=False)
        return _build_sequence_windows_from_matrix(
            matrix=scaled_features,
            lookback_bars=self.lookback_bars,
            start_index=len(history_slice),
            target_count=len(df),
        )

    def _build_torch_sequence_module(self, torch, nn, input_size: int):
        train_cfg = self.cfg.get("train", {})
        if self.model_type == "lstm":
            return _build_lstm_sequence_module(
                nn=nn,
                input_size=input_size,
                hidden_size=int(train_cfg.get("lstm_hidden_size", 64)),
                num_layers=int(train_cfg.get("lstm_num_layers", 2)),
                class_count=len(LABEL_TO_ID),
                dropout=self.dropout,
            )
        if self.model_type == "cnn":
            return _build_cnn_sequence_module(
                nn=nn,
                input_size=input_size,
                channels=int(train_cfg.get("cnn_channels", 64)),
                kernel_size=int(train_cfg.get("cnn_kernel_size", 3)),
                class_count=len(LABEL_TO_ID),
                dropout=self.dropout,
            )
        if self.model_type == "transformer":
            return _build_transformer_sequence_module(
                torch=torch,
                nn=nn,
                input_size=input_size,
                lookback_bars=self.lookback_bars,
                d_model=int(train_cfg.get("transformer_d_model", 64)),
                nhead=int(train_cfg.get("transformer_nhead", 4)),
                num_layers=int(train_cfg.get("transformer_num_layers", 2)),
                dim_feedforward=int(train_cfg.get("transformer_dim_feedforward", 128)),
                class_count=len(LABEL_TO_ID),
                dropout=self.dropout,
            )
        raise ValueError(f"Modelo sequencial nao suportado: {self.model_type}")

    def fit(self, train_df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        torch, nn, DataLoader, TensorDataset = _require_torch()
        self._remember_feature_columns(feature_columns)
        self._torch = torch
        self._device = self._resolve_device(torch)

        x_train, y_train = self._prepare_training_sequences(train_df, feature_columns)
        self._input_size = int(x_train.shape[2])

        model = self._build_torch_sequence_module(torch=torch, nn=nn, input_size=self._input_size).to(self._device)
        counts = np.bincount(y_train, minlength=len(LABEL_TO_ID)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        class_weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self._device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        data_loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, len(dataset))),
            shuffle=True,
        )

        model.train()
        for _epoch in range(max(1, self.epochs)):
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self._device)
                batch_targets = batch_targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

        model.eval()
        self._torch_model = model
        self._class_ids = [LABEL_TO_ID["bear"], LABEL_TO_ID["range"], LABEL_TO_ID["bull"]]

    def onnx_input_array(
        self,
        sample_df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        return self._transform_inference_sequences(sample_df, feature_columns, history_df=history_df)

    def predict(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(df, feature_columns, history_df=history_df)
        return probabilities.argmax(axis=1).astype(np.int64, copy=False)

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        history_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if self._torch_model is None or self._torch is None or self._device is None:
            raise RuntimeError(f"Modelo {self.model_type} ainda nao foi treinado")

        x_test = self._transform_inference_sequences(df, feature_columns, history_df=history_df)
        if len(x_test) == 0:
            return np.zeros((0, len(LABEL_TO_ID)), dtype=np.float64)

        batch_size = max(1, min(self.batch_size, len(x_test)))
        probabilities: List[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                end = start + batch_size
                batch = self._torch.tensor(x_test[start:end], dtype=self._torch.float32, device=self._device)
                logits = self._torch_model(batch)
                batch_probabilities = self._torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(batch_probabilities.astype(np.float64, copy=False))

        return np.vstack(probabilities)

    def export_onnx(self, output_path: Path, opset: int, feature_count: int | None = None) -> Dict[str, List[str] | str]:
        torch, nn, _DataLoader, _TensorDataset = _require_torch()
        if self._torch_model is None:
            raise RuntimeError(f"Modelo {self.model_type} ainda nao foi treinado")

        if feature_count is None:
            feature_count = self._input_size
        model_cpu = self._torch_model.to("cpu").eval()
        fill_values = self._imputer.statistics_.astype(np.float32, copy=False)
        feature_means = self._scaler.mean_.astype(np.float32, copy=False)
        feature_scales = np.where(self._scaler.scale_ == 0.0, 1.0, self._scaler.scale_).astype(np.float32, copy=False)
        wrapper = _build_torch_sequence_export_wrapper(
            nn=nn,
            torch=torch,
            base_model=model_cpu,
            medians=fill_values,
            means=feature_means,
            scales=feature_scales,
        ).eval()

        sample_input = torch.zeros((1, int(self.lookback_bars), int(feature_count)), dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            sample_input,
            output_path.as_posix(),
            input_names=["sequence"],
            output_names=["label", "probabilities"],
            dynamic_axes={
                "sequence": {0: "batch_size"},
                "label": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
            opset_version=int(opset),
            dynamo=False,
        )
        return {
            "input_name": "sequence",
            "output_names": ["label", "probabilities"],
        }


def build_model_backend(cfg: Dict) -> BaseRegimeModel:
    train_cfg = cfg.get("train", {})
    model_type = normalize_model_type(train_cfg.get("model_type", "logistic_regression"))
    cfg["train"]["model_type"] = model_type

    if model_type in SKLEARN_MODEL_TYPES:
        return SklearnPipelineRegimeModel(cfg)

    if model_type in DEEP_TABULAR_MODEL_TYPES:
        return TorchTabularRegimeModel(cfg)

    if model_type in SEQUENCE_MODEL_TYPES:
        return TorchSequenceRegimeModel(cfg)

    supported = sorted(SKLEARN_MODEL_TYPES | DEEP_TABULAR_MODEL_TYPES | SEQUENCE_MODEL_TYPES)
    raise ValueError(f"train.model_type invalido: {model_type}. Opcoes atuais: {supported}")


def build_pipeline(cfg: Dict) -> Pipeline:
    model = build_model_backend(cfg)
    if not isinstance(model, SklearnPipelineRegimeModel):
        raise ValueError(
            "build_pipeline suporta apenas modelos sklearn. Use build_model_backend para deep_mlp e proximos backends."
        )
    return model.pipeline


def evaluate_model(
    model: BaseRegimeModel,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    history_df: pd.DataFrame | None = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    y_true = test_df["regime_id"].astype(int)
    y_pred = model.predict(test_df, feature_columns, history_df=history_df)
    y_proba = model.predict_proba(test_df, feature_columns, history_df=history_df)
    proba_full = np.zeros((len(test_df), len(LABEL_TO_ID)), dtype=np.float64)
    for source_index, class_id in enumerate(model.class_ids):
        proba_full[:, int(class_id)] = y_proba[:, source_index]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "class_distribution_test": {
                ID_TO_LABEL[int(label_id)]: int(count)
                for label_id, count in y_true.value_counts().sort_index().items()
            },
        }
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1], ID_TO_LABEL[2]],
            zero_division=0,
            output_dict=True,
        )

    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{ID_TO_LABEL[label_id]}" for label_id in [0, 1, 2]],
        columns=[f"pred_{ID_TO_LABEL[label_id]}" for label_id in [0, 1, 2]],
    )

    predictions = test_df[["open", "high", "low", "close", "volume", "regime_id", "regime_name"]].copy()
    predictions["predicted_regime_id"] = y_pred.astype(int)
    predictions["predicted_regime_name"] = predictions["predicted_regime_id"].map(ID_TO_LABEL)
    predictions["prob_bear"] = proba_full[:, LABEL_TO_ID["bear"]]
    predictions["prob_range"] = proba_full[:, LABEL_TO_ID["range"]]
    predictions["prob_bull"] = proba_full[:, LABEL_TO_ID["bull"]]

    return metrics, confusion_df, predictions


def export_onnx_model(
    model: BaseRegimeModel,
    feature_count: int,
    output_path: Path,
    opset: int,
) -> Dict[str, List[str] | str]:
    return model.export_onnx(
        output_path=output_path,
        opset=opset,
        feature_count=feature_count,
    )


def validate_onnx_model(
    model: BaseRegimeModel,
    onnx_path: Path,
    sample_df: pd.DataFrame,
    feature_columns: Sequence[str],
    history_df: pd.DataFrame | None = None,
) -> Dict:
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None

    if ort is None:
        return {"status": "skipped", "reason": "onnxruntime nao instalado"}

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    x = model.onnx_input_array(sample_df, feature_columns, history_df=history_df)
    onnx_outputs = session.run(None, {input_name: x})
    reference_proba = model.predict_proba(sample_df, feature_columns, history_df=history_df)

    probability_output = None
    for output in onnx_outputs:
        if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[1] == len(LABEL_TO_ID):
            probability_output = output.astype(np.float64)
            break

    if probability_output is None:
        return {
            "status": "warning",
            "reason": "saida de probabilidade [N,3] nao encontrada no ONNX",
            "output_names": output_names,
        }

    max_abs_diff = float(np.max(np.abs(probability_output - reference_proba)))
    return {
        "status": "ok",
        "output_names": output_names,
        "rows_checked": int(len(sample_df)),
        "max_abs_diff_vs_reference": max_abs_diff,
    }
