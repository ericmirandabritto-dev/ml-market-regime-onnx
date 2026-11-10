from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.pipeline import Pipeline

from market_regime_core import build_pipeline


POSITION_MANAGEMENT_MODEL_SCHEMA_VERSION = 1
DEFAULT_FALLBACK_ACTION = "hold_base"


def load_position_management_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_parts(path: Path) -> Tuple[str, ...]:
    return tuple(str(part).lower() for part in path.parts)


def resolve_manifest_data_path(manifest_path: Path, manifest: Dict, raw_path: str) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate

    output_root = Path(str(manifest.get("output_root", "")))
    candidate_parts = candidate.parts
    output_root_parts = output_root.parts

    if output_root_parts and _normalize_parts(candidate)[: len(output_root_parts)] == _normalize_parts(output_root):
        candidate = Path(*candidate_parts[len(output_root_parts) :])

    return (manifest_path.parent / candidate).resolve()


def load_position_management_training_samples(
    manifest_path: Path,
    training_samples_override: Path | None = None,
) -> Tuple[Dict, pd.DataFrame, Path]:
    manifest = load_position_management_manifest(manifest_path)
    if training_samples_override is not None:
        samples_path = training_samples_override.resolve()
    else:
        raw_samples_path = str(manifest["paths"]["training_samples"])
        samples_path = resolve_manifest_data_path(manifest_path, manifest, raw_samples_path)

    samples_df = pd.read_csv(samples_path)
    for column in ["bar_time", "entry_time", "actual_exit_time"]:
        if column in samples_df.columns:
            samples_df[column] = pd.to_datetime(samples_df[column], errors="coerce")

    return manifest, samples_df, samples_path


def resolve_feature_columns(cfg: Dict, manifest: Dict, samples_df: pd.DataFrame) -> List[str]:
    raw_value = cfg.get("data", {}).get("feature_columns", "auto")
    if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
        feature_columns = [str(column) for column in manifest.get("recommended_training_columns", [])]
    else:
        feature_columns = [str(column) for column in raw_value]

    if not feature_columns:
        raise ValueError("Nenhuma feature configurada para o treino do modelo de gestao.")

    missing = [column for column in feature_columns if column not in samples_df.columns]
    if missing:
        raise ValueError(f"training_samples sem as features esperadas: {missing}")

    return feature_columns


def resolve_action_names(samples_df: pd.DataFrame, manifest: Dict, cfg: Dict) -> List[str]:
    if "best_action" not in samples_df.columns:
        raise ValueError("training_samples precisa conter a coluna best_action.")

    observed_actions = {
        str(value).strip()
        for value in samples_df["best_action"].dropna().astype(str).tolist()
        if str(value).strip() != ""
    }
    if not observed_actions:
        raise ValueError("Nao ha labels best_action no dataset.")

    allowed_actions = [str(value).strip() for value in cfg.get("train", {}).get("allowed_actions", []) if str(value).strip()]
    if allowed_actions:
        blocked = sorted(observed_actions - set(allowed_actions))
        if blocked:
            raise ValueError(
                "O dataset contem labels fora de train.allowed_actions. "
                f"Revise a configuracao ou regenere labels. Faltando: {blocked}"
            )

    manifest_order = [
        str(item.get("name", "")).strip()
        for item in manifest.get("simulation_actions", [])
        if str(item.get("name", "")).strip()
    ]
    action_order = allowed_actions or manifest_order or sorted(observed_actions)

    resolved: List[str] = []
    for action in action_order:
        if action in observed_actions and action not in resolved:
            resolved.append(action)

    for action in sorted(observed_actions):
        if action not in resolved:
            resolved.append(action)

    if len(resolved) < 2:
        raise ValueError("O modelo de gestao precisa de pelo menos 2 acoes rotuladas no dataset.")

    return resolved


def encode_action_targets(samples_df: pd.DataFrame, action_names: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    action_to_id = {action: index for index, action in enumerate(action_names)}
    encoded = samples_df.copy()
    encoded["best_action"] = encoded["best_action"].astype(str)
    encoded["target_id"] = encoded["best_action"].map(action_to_id)

    missing_mask = encoded["target_id"].isna()
    if missing_mask.any():
        missing_actions = sorted(encoded.loc[missing_mask, "best_action"].dropna().unique().tolist())
        raise ValueError(f"Labels best_action fora do space de treino: {missing_actions}")

    encoded["target_id"] = encoded["target_id"].astype(int)
    return encoded, action_to_id


def _resolve_sort_column(samples_df: pd.DataFrame, configured_column: str) -> str:
    candidate = str(configured_column or "").strip()
    if candidate and candidate in samples_df.columns:
        return candidate
    if "entry_time" in samples_df.columns:
        return "entry_time"
    if "bar_time" in samples_df.columns:
        return "bar_time"
    raise ValueError("Nao foi possivel determinar a coluna temporal para ordenar as trades.")


def chronological_trade_split(samples_df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    trade_column = str(data_cfg.get("trade_group_column", "trade_id"))
    if trade_column not in samples_df.columns:
        raise ValueError(f"training_samples sem a coluna de grupo {trade_column}")

    test_size = float(train_cfg.get("test_size_trades", 0.20))
    if not 0.05 <= test_size < 0.5:
        raise ValueError("train.test_size_trades deve ficar entre 0.05 e 0.50")

    sort_column = _resolve_sort_column(samples_df, str(data_cfg.get("trade_sort_column", "entry_time")))
    order_df = (
        samples_df[[trade_column, sort_column]]
        .drop_duplicates(subset=[trade_column])
        .sort_values([sort_column, trade_column], kind="mergesort")
        .reset_index(drop=True)
    )

    trade_count = int(len(order_df))
    if trade_count < 10:
        raise ValueError("O treino de gestao precisa de ao menos 10 trades para split cronologico.")

    test_trade_count = max(1, int(trade_count * test_size))
    train_trade_count = trade_count - test_trade_count

    min_train_trades = int(train_cfg.get("min_trades_train", 30))
    min_test_trades = int(train_cfg.get("min_trades_test", 10))
    if train_trade_count < min_train_trades:
        raise ValueError(
            f"Split gerou apenas {train_trade_count} trades no treino, abaixo de train.min_trades_train={min_train_trades}."
        )
    if test_trade_count < min_test_trades:
        raise ValueError(
            f"Split gerou apenas {test_trade_count} trades no teste, abaixo de train.min_trades_test={min_test_trades}."
        )

    train_trade_ids = order_df.iloc[:train_trade_count][trade_column].tolist()
    test_trade_ids = order_df.iloc[train_trade_count:][trade_column].tolist()

    sort_columns = [column for column in [sort_column, "bar_time", "snapshot_id"] if column in samples_df.columns]
    train_df = samples_df.loc[samples_df[trade_column].isin(train_trade_ids)].copy().sort_values(sort_columns, kind="mergesort")
    test_df = samples_df.loc[samples_df[trade_column].isin(test_trade_ids)].copy().sort_values(sort_columns, kind="mergesort")

    if train_df.empty or test_df.empty:
        raise ValueError("Split cronologico gerou treino ou teste vazio.")

    return train_df, test_df, order_df


def validate_train_action_classes(train_df: pd.DataFrame, action_names: Sequence[str], min_class_count: int) -> Tuple[bool, List[str]]:
    counts = train_df["target_id"].astype(int).value_counts().sort_index()
    issues: List[str] = []

    missing = [action_names[index] for index in range(len(action_names)) if index not in counts.index]
    if missing:
        issues.append(
            "Treino sem todas as acoes necessarias. "
            f"Ajuste o split temporal ou labels. Faltando: {missing}"
        )

    low_count = [
        action_names[int(label_id)]
        for label_id, count in counts.items()
        if int(count) < int(min_class_count)
    ]
    if low_count:
        issues.append(
            "Treino com acoes abaixo do minimo exigido "
            f"({min_class_count}): {low_count}"
        )

    return (len(issues) == 0), issues


def fit_position_management_model(
    cfg: Dict,
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Pipeline:
    model = build_pipeline(cfg)
    fit_kwargs = {}

    sample_weight_column = str(cfg.get("data", {}).get("sample_weight_column", "sample_weight"))
    use_sample_weight = bool(cfg.get("train", {}).get("use_sample_weight", True))
    if use_sample_weight and sample_weight_column in train_df.columns:
        sample_weights = train_df[sample_weight_column].astype(float).fillna(1.0).to_numpy(dtype=np.float64)
        fit_kwargs["classifier__sample_weight"] = sample_weights

    model.fit(train_df[list(feature_columns)], train_df["target_id"].astype(int), **fit_kwargs)
    return model


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _utility_column_name(action: str) -> str:
    return f"{action}_utility"


def _validity_column_name(action: str) -> str:
    return f"{action}_is_valid"


def build_position_dataset_diagnostics(
    samples_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    action_names: Sequence[str],
) -> Dict:
    diagnostics = {
        "rows_total": int(len(samples_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "trades_total": int(samples_df["trade_id"].nunique()) if "trade_id" in samples_df.columns else None,
        "trades_train": int(train_df["trade_id"].nunique()) if "trade_id" in train_df.columns else None,
        "trades_test": int(test_df["trade_id"].nunique()) if "trade_id" in test_df.columns else None,
        "time_start": str(samples_df["bar_time"].min()) if "bar_time" in samples_df.columns else None,
        "time_end": str(samples_df["bar_time"].max()) if "bar_time" in samples_df.columns else None,
        "train_action_distribution": {
            action_names[int(label_id)]: int(count)
            for label_id, count in train_df["target_id"].astype(int).value_counts().sort_index().items()
        },
        "test_action_distribution": {
            action_names[int(label_id)]: int(count)
            for label_id, count in test_df["target_id"].astype(int).value_counts().sort_index().items()
        },
        "warnings": [],
    }

    if "sample_weight" in train_df.columns:
        diagnostics["train_sample_weight_sum"] = float(train_df["sample_weight"].astype(float).fillna(0.0).sum())
    if "sample_weight" in test_df.columns:
        diagnostics["test_sample_weight_sum"] = float(test_df["sample_weight"].astype(float).fillna(0.0).sum())

    for distribution_name in ["train_action_distribution", "test_action_distribution"]:
        distribution = diagnostics[distribution_name]
        if any(count < 100 for count in distribution.values()):
            diagnostics["warnings"].append(
                f"{distribution_name} contem ao menos uma acao com menos de 100 amostras."
            )
            break

    return diagnostics


def evaluate_position_management_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    action_names: Sequence[str],
    cfg: Dict,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    x_test = test_df[list(feature_columns)]
    y_true = test_df["target_id"].astype(int).to_numpy(dtype=np.int64)
    raw_proba = model.predict_proba(x_test)
    model_classes = [int(class_id) for class_id in model.named_steps["classifier"].classes_]

    full_proba = np.zeros((len(x_test), len(action_names)), dtype=np.float64)
    for source_index, class_id in enumerate(model_classes):
        full_proba[:, class_id] = raw_proba[:, source_index]

    raw_pred_ids = full_proba.argmax(axis=1).astype(np.int64)

    validity_matrix = np.ones((len(test_df), len(action_names)), dtype=np.int8)
    for action_index, action in enumerate(action_names):
        validity_column = _validity_column_name(action)
        if validity_column in test_df.columns:
            validity_matrix[:, action_index] = test_df[validity_column].fillna(0).astype(int).to_numpy(dtype=np.int8)

    utility_matrix = np.full((len(test_df), len(action_names)), np.nan, dtype=np.float64)
    for action_index, action in enumerate(action_names):
        utility_column = _utility_column_name(action)
        if utility_column in test_df.columns:
            utility_matrix[:, action_index] = test_df[utility_column].astype(float).to_numpy(dtype=np.float64)

    fallback_action = str(cfg.get("train", {}).get("fallback_action", DEFAULT_FALLBACK_ACTION))
    fallback_id = action_names.index(fallback_action) if fallback_action in action_names else int(raw_pred_ids[0])
    use_validity_mask = bool(cfg.get("train", {}).get("use_validity_mask_on_inference", True))

    selected_ids = np.zeros(len(test_df), dtype=np.int64)
    raw_invalid_flags = np.zeros(len(test_df), dtype=np.int8)
    selected_invalid_flags = np.zeros(len(test_df), dtype=np.int8)

    for row_index in range(len(test_df)):
        raw_pred_id = int(raw_pred_ids[row_index])
        if validity_matrix[row_index, raw_pred_id] <= 0:
            raw_invalid_flags[row_index] = 1

        selected_id = raw_pred_id
        if use_validity_mask:
            valid_ids = np.flatnonzero(validity_matrix[row_index] > 0)
            if valid_ids.size > 0:
                row_proba = full_proba[row_index, valid_ids]
                selected_id = int(valid_ids[int(np.argmax(row_proba))])
            else:
                selected_id = fallback_id

        selected_ids[row_index] = selected_id
        if validity_matrix[row_index, selected_id] <= 0:
            selected_invalid_flags[row_index] = 1

    predicted_utility = utility_matrix[np.arange(len(test_df)), selected_ids]
    best_utility = test_df["best_action_utility_score"].astype(float).to_numpy(dtype=np.float64)
    utility_regret = best_utility - predicted_utility

    hold_base_gap = None
    if "hold_base" in action_names:
        hold_base_id = action_names.index("hold_base")
        hold_base_gap = predicted_utility - utility_matrix[:, hold_base_id]

    exit_now_gap = None
    if "exit_now" in action_names:
        exit_now_id = action_names.index("exit_now")
        exit_now_gap = predicted_utility - utility_matrix[:, exit_now_id]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metrics = {
            "accuracy": float(accuracy_score(y_true, selected_ids)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, selected_ids)),
            "class_distribution_test": {
                action_names[int(label_id)]: int(count)
                for label_id, count in pd.Series(y_true).value_counts().sort_index().items()
            },
            "predicted_action_distribution": {
                action_names[int(label_id)]: int(count)
                for label_id, count in pd.Series(selected_ids).value_counts().sort_index().items()
            },
            "mean_predicted_utility": _safe_mean(predicted_utility),
            "mean_best_utility": _safe_mean(best_utility),
            "mean_utility_regret": _safe_mean(utility_regret),
            "mean_predicted_minus_hold_base": _safe_mean(hold_base_gap) if hold_base_gap is not None else None,
            "mean_predicted_minus_exit_now": _safe_mean(exit_now_gap) if exit_now_gap is not None else None,
            "raw_invalid_rate": float(raw_invalid_flags.mean()) if len(raw_invalid_flags) else 0.0,
            "selected_invalid_rate": float(selected_invalid_flags.mean()) if len(selected_invalid_flags) else 0.0,
        }

        if len(action_names) > 2:
            metrics["top2_accuracy"] = float(
                top_k_accuracy_score(y_true, full_proba, k=2, labels=list(range(len(action_names))))
            )
        else:
            metrics["top2_accuracy"] = None

        metrics["classification_report"] = classification_report(
            y_true,
            selected_ids,
            labels=list(range(len(action_names))),
            target_names=list(action_names),
            zero_division=0,
            output_dict=True,
        )

    exit_actions = [str(value).strip() for value in cfg.get("runtime", {}).get("exit_actions", ["exit_now"]) if str(value).strip()]
    exit_action_ids = [action_names.index(action) for action in exit_actions if action in action_names]
    predicted_should_exit = np.isin(selected_ids, exit_action_ids).astype(np.int8)
    if exit_action_ids and "label_should_exit" in test_df.columns:
        metrics["exit_decision_accuracy"] = float(
            accuracy_score(test_df["label_should_exit"].astype(int), predicted_should_exit)
        )
    else:
        metrics["exit_decision_accuracy"] = None

    confusion = confusion_matrix(y_true, selected_ids, labels=list(range(len(action_names))))
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{action}" for action in action_names],
        columns=[f"pred_{action}" for action in action_names],
    )

    predictions_df = test_df[
        [column for column in ["snapshot_id", "trade_id", "bar_time", "best_action", "label_should_exit"] if column in test_df.columns]
    ].copy()
    predictions_df["predicted_action"] = [action_names[int(label_id)] for label_id in selected_ids]
    predictions_df["raw_predicted_action"] = [action_names[int(label_id)] for label_id in raw_pred_ids]
    predictions_df["predicted_action_probability"] = full_proba[np.arange(len(test_df)), selected_ids]
    predictions_df["raw_predicted_action_probability"] = full_proba[np.arange(len(test_df)), raw_pred_ids]
    predictions_df["predicted_action_is_valid"] = 1 - selected_invalid_flags
    predictions_df["raw_predicted_action_is_valid"] = 1 - raw_invalid_flags
    predictions_df["predicted_action_utility"] = predicted_utility
    predictions_df["best_action_utility_score"] = best_utility
    predictions_df["utility_regret"] = utility_regret
    predictions_df["predicted_should_exit"] = predicted_should_exit

    for action_index, action in enumerate(action_names):
        predictions_df[f"prob_{action}"] = full_proba[:, action_index]

    return metrics, confusion_df, predictions_df


def validate_position_management_onnx_model(
    model: Pipeline,
    onnx_path: Path,
    sample_df: pd.DataFrame,
    feature_columns: Sequence[str],
    class_count: int,
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

    x = sample_df[list(feature_columns)].to_numpy(dtype=np.float32)
    onnx_outputs = session.run(None, {input_name: x})
    sklearn_proba = model.predict_proba(sample_df[list(feature_columns)])

    probability_output = None
    for output in onnx_outputs:
        if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[1] == int(class_count):
            probability_output = output.astype(np.float64)
            break

    if probability_output is None:
        return {
            "status": "warning",
            "reason": f"saida de probabilidade [N,{class_count}] nao encontrada no ONNX",
            "output_names": output_names,
        }

    max_abs_diff = float(np.max(np.abs(probability_output - sklearn_proba)))
    return {
        "status": "ok",
        "output_names": output_names,
        "rows_checked": int(len(sample_df)),
        "max_abs_diff_vs_sklearn": max_abs_diff,
    }
