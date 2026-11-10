from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from market_regime_core import (
    FEATURE_SCHEMA_VERSION,
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_model_backend,
    class_distribution,
    ensure_dir,
    evaluate_model,
    export_market_runtime_contract,
    export_regime_onnx,
    save_json,
    validate_train_classes,
)


DEFAULT_OBJECTIVE = "balanced_accuracy_stable"


@dataclass(frozen=True)
class WalkForwardPolicy:
    train_window: int
    validation_window: int
    trade_window: int
    step_window: int
    purge_bars: int
    min_folds: int = 3
    max_folds: int = 0
    min_class_count_per_train: int = 1
    objective: str = DEFAULT_OBJECTIVE


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: str
    anchor_index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    trade_start: int
    trade_end: int


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def should_log_progress(index: int, total: int, every: int) -> bool:
    every = max(1, int(every))
    return index == 1 or index == total or index % every == 0


def policy_signature(policy: WalkForwardPolicy) -> str:
    return (
        f"train{policy.train_window}_"
        f"valid{policy.validation_window}_"
        f"trade{policy.trade_window}_"
        f"step{policy.step_window}_"
        f"purge{policy.purge_bars}_"
        f"maxfolds{policy.max_folds}"
    )


def resolve_purge_bars(cfg: Dict, raw_value: str | int | None) -> int:
    if raw_value is None:
        raw_value = cfg.get("walkforward", {}).get("purge_bars", "auto")

    horizon = int(cfg["label"]["horizon_bars"])
    if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
        return horizon

    purge_bars = int(raw_value)
    if purge_bars < horizon:
        raise ValueError(f"walkforward.purge_bars deve ser pelo menos {horizon} (horizonte do rotulo)")
    return purge_bars


def cfg_hash(cfg: Dict, policy: WalkForwardPolicy, export_options: Dict) -> str:
    payload = {
        "config": cfg,
        "policy": asdict(policy),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "export_options": export_options,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _dataset_hash(dataset: pd.DataFrame) -> str:
    rows = pd.util.hash_pandas_object(dataset, index=True)
    return hashlib.sha256(rows.to_numpy(dtype="<u8").tobytes()).hexdigest()


def format_mql_time(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y.%m.%d %H:%M:%S")


def path_to_windows_string(path: Path) -> str:
    return str(path).replace("/", "\\")


def build_fold_definition(policy: WalkForwardPolicy, fold_number: int) -> FoldDefinition:
    anchor = fold_number * policy.step_window
    train_start = anchor
    train_end = train_start + policy.train_window
    validation_start = train_end + policy.purge_bars
    validation_end = validation_start + policy.validation_window
    trade_start = validation_end + policy.purge_bars
    trade_end = trade_start + policy.trade_window
    return FoldDefinition(
        fold_id=f"fold_{fold_number + 1:04d}",
        anchor_index=anchor,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        trade_start=trade_start,
        trade_end=trade_end,
    )


def _validate_walkforward_policy(policy: WalkForwardPolicy) -> None:
    if policy.purge_bars < 0:
        raise ValueError("walkforward.purge_bars nao pode ser negativo")
    if policy.train_window <= 0:
        raise ValueError("walkforward.train_window deve ser maior que zero")
    if policy.validation_window < 0:
        raise ValueError("walkforward.validation_window nao pode ser negativo")
    if policy.trade_window <= 0:
        raise ValueError("walkforward.trade_window deve ser maior que zero")
    if policy.step_window <= 0:
        raise ValueError("walkforward.step_window deve ser maior que zero")


def generate_walkforward_folds(dataset: pd.DataFrame, policy: WalkForwardPolicy, apply_cap: bool = True) -> List[FoldDefinition]:
    _validate_walkforward_policy(policy)

    folds: List[FoldDefinition] = []
    fold_number = 0
    total_rows = len(dataset)

    while True:
        fold = build_fold_definition(policy, fold_number)

        if fold.trade_end > total_rows:
            break

        folds.append(fold)

        fold_number += 1

    if apply_cap and policy.max_folds > 0 and len(folds) > policy.max_folds:
        return folds[-policy.max_folds :]

    return folds


def generate_train_ready_folds(dataset: pd.DataFrame, policy: WalkForwardPolicy) -> List[FoldDefinition]:
    _validate_walkforward_policy(policy)

    folds: List[FoldDefinition] = []
    fold_number = 0
    total_rows = len(dataset)

    while True:
        fold = build_fold_definition(policy, fold_number)
        if fold.train_end > total_rows:
            break

        folds.append(fold)
        fold_number += 1

    return folds


def score_walkforward_metrics(
    balanced_accuracies: Sequence[float],
    valid_folds: int,
    planned_folds: int,
    min_folds: int,
) -> float | None:
    if valid_folds == 0:
        return None

    mean_balanced_accuracy = float(np.mean(balanced_accuracies))
    std_balanced_accuracy = float(np.std(balanced_accuracies))
    coverage = float(valid_folds / planned_folds) if planned_folds else 0.0
    fold_penalty = max(0, min_folds - valid_folds) * 0.05
    coverage_penalty = max(0.0, 1.0 - coverage) * 0.10
    return mean_balanced_accuracy - (0.50 * std_balanced_accuracy) - fold_penalty - coverage_penalty


def aggregate_walkforward_summary(summary_df: pd.DataFrame, policy: WalkForwardPolicy) -> Dict:
    planned_folds = int(len(summary_df))
    ok_df = summary_df.loc[summary_df["status"] == "ok"].copy()
    valid_folds = int(len(ok_df))

    aggregate = {
        "objective": policy.objective,
        "available_folds": planned_folds,
        "planned_folds": planned_folds,
        "valid_folds": valid_folds,
        "coverage": float(valid_folds / planned_folds) if planned_folds else 0.0,
        "eligible": bool(valid_folds >= policy.min_folds),
        "train_window": int(policy.train_window),
        "validation_window": int(policy.validation_window),
        "trade_window": int(policy.trade_window),
        "step_window": int(policy.step_window),
        "purge_bars": int(policy.purge_bars),
        "min_folds": int(policy.min_folds),
        "max_folds": int(policy.max_folds),
        "min_class_count_per_train": int(policy.min_class_count_per_train),
        "score": None,
        "mean_accuracy_trade": None,
        "mean_balanced_accuracy_trade": None,
        "std_balanced_accuracy_trade": None,
        "mean_accuracy_validation": None,
        "mean_balanced_accuracy_validation": None,
        "trade_rows_total": int(ok_df["trade_rows"].sum()) if valid_folds else 0,
    }

    if valid_folds == 0:
        return aggregate

    balanced_accuracies = ok_df["trade_balanced_accuracy"].astype(float).tolist()
    aggregate["score"] = score_walkforward_metrics(
        balanced_accuracies=balanced_accuracies,
        valid_folds=valid_folds,
        planned_folds=planned_folds,
        min_folds=policy.min_folds,
    )
    aggregate["mean_accuracy_trade"] = float(ok_df["trade_accuracy"].mean())
    aggregate["mean_balanced_accuracy_trade"] = float(ok_df["trade_balanced_accuracy"].mean())
    aggregate["std_balanced_accuracy_trade"] = float(ok_df["trade_balanced_accuracy"].std(ddof=0))

    validation_df = ok_df.loc[ok_df["validation_rows"] > 0].copy()
    if not validation_df.empty:
        aggregate["mean_accuracy_validation"] = float(validation_df["validation_accuracy"].mean())
        aggregate["mean_balanced_accuracy_validation"] = float(validation_df["validation_balanced_accuracy"].mean())

    return aggregate


def _df_slice(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if end <= start:
        return df.iloc[0:0].copy()
    return df.iloc[start:end].copy()


def _history_slice(df: pd.DataFrame, target_start: int, history_rows_needed: int) -> pd.DataFrame | None:
    if history_rows_needed <= 0:
        return None

    history_start = max(0, int(target_start) - int(history_rows_needed))
    history_end = max(history_start, int(target_start))
    history_df = _df_slice(df, history_start, history_end)
    return history_df if not history_df.empty else None


def _runtime_relative_path(runtime_root: str, *parts: str) -> str:
    clean_root = runtime_root.strip("\\/")
    raw = Path(clean_root)
    for part in parts:
        raw /= part
    return path_to_windows_string(raw)


def _clean_record_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_csv_record_lookup(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty or "fold_id" not in df.columns:
        return {}

    lookup: Dict[str, Dict] = {}
    for raw_record in df.to_dict(orient="records"):
        record = {key: _clean_record_value(value) for key, value in raw_record.items()}
        fold_id = str(record.get("fold_id", "")).strip()
        if fold_id:
            lookup[fold_id] = record
    return lookup


def _load_json(path: Path) -> Dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_existing_predictions(fold_dir: Path) -> pd.DataFrame | None:
    predictions_path = fold_dir / "trade_predictions.csv"
    if not predictions_path.exists():
        return None

    predictions_df = pd.read_csv(predictions_path)
    if "time" in predictions_df.columns:
        predictions_df["time"] = pd.to_datetime(predictions_df["time"], errors="coerce")
    return predictions_df


def _stringify_optional(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _can_reuse_exported_fold(
    fold_dir: Path,
    fold_record: Dict,
    configuration_hash: str,
    data_hash: str,
) -> bool:
    if not (fold_dir / "model.onnx").is_file():
        return False
    if not (fold_dir / "model_metadata.json").exists():
        return False
    if not (fold_dir / "runtime_contract.csv").exists():
        return False

    fold_info = _load_json(fold_dir / "fold_info.json")
    if not fold_info:
        return False

    if fold_info.get("config_hash") != configuration_hash or fold_info.get("data_hash") != data_hash:
        return False
    if int(fold_info.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
        return False

    expected_pairs = {
        "train_start": fold_record["train_start"],
        "train_end": fold_record["train_end"],
        "trained_until": fold_record["trained_until"],
        "validation_start": fold_record["validation_start"],
        "validation_end": fold_record["validation_end"],
        "trade_start": fold_record["trade_start"],
        "trade_end": fold_record["trade_end"],
        "active_from": fold_record["active_from"],
        "active_to": fold_record["active_to"],
    }
    for key, expected_value in expected_pairs.items():
        if _stringify_optional(fold_info.get(key)) != _stringify_optional(expected_value):
            return False

    expected_rows = {
        "rows_train": fold_record["train_rows"],
        "rows_validation": fold_record["validation_rows"],
        "rows_trade": fold_record["trade_rows"],
    }
    for key, expected_value in expected_rows.items():
        if int(fold_info.get(key, -1)) != int(expected_value):
            return False

    return True


def _load_existing_export_state(
    export_dir: Path | None,
    policy: WalkForwardPolicy,
    configuration_hash: str,
    incremental_export: bool,
) -> Dict[str, Dict]:
    empty_state = {
        "enabled": False,
        "summary_by_fold": {},
        "manifest_by_fold": {},
    }

    if export_dir is None or not incremental_export:
        return empty_state

    manifest_payload = _load_json(export_dir / "research_manifest.json")
    if manifest_payload is None:
        manifest_payload = _load_json(export_dir / "manifest.json")
    if manifest_payload:
        manifest_policy = manifest_payload.get("policy", {})
        if manifest_payload.get("config_hash") != configuration_hash:
            return empty_state
        if manifest_policy != asdict(policy):
            return empty_state
        if int(manifest_payload.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
            return empty_state

    summary_by_fold = _load_csv_record_lookup(export_dir / "summary.csv")
    manifest_by_fold = _load_csv_record_lookup(export_dir / "research_manifest.csv")
    if not manifest_by_fold:
        manifest_by_fold = _load_csv_record_lookup(export_dir / "manifest.csv")
    if not summary_by_fold or not manifest_by_fold:
        return empty_state

    return {
        "enabled": True,
        "summary_by_fold": summary_by_fold,
        "manifest_by_fold": manifest_by_fold,
    }


def _format_mql_optional(value: str | None) -> str | None:
    if not value:
        return None
    return format_mql_time(pd.Timestamp(value))


def _classify_runtime_fold_state(total_rows: int, fold: FoldDefinition) -> str:
    if total_rows >= fold.trade_end:
        return "closed"
    if total_rows > fold.trade_start:
        return "open"
    if total_rows >= fold.train_end:
        return "staged"
    return "pending"


def _build_runtime_fold_record(dataset: pd.DataFrame, fold: FoldDefinition) -> Dict:
    total_rows = len(dataset)
    state = _classify_runtime_fold_state(total_rows, fold)

    train_df = _df_slice(dataset, fold.train_start, min(fold.train_end, total_rows))
    validation_df = _df_slice(dataset, fold.validation_start, min(fold.validation_end, total_rows))
    trade_df = _df_slice(dataset, fold.trade_start, min(fold.trade_end, total_rows))

    train_rows = int(fold.train_end - fold.train_start)
    validation_rows = int(max(0, fold.validation_end - fold.validation_start))
    trade_rows = int(fold.trade_end - fold.trade_start)

    train_start = str(train_df.index.min()) if not train_df.empty else None
    train_end = str(train_df.index.max()) if len(train_df) == train_rows and not train_df.empty else None
    validation_start = str(validation_df.index.min()) if not validation_df.empty else None
    validation_end = str(validation_df.index.max()) if len(validation_df) == validation_rows and not validation_df.empty else None
    active_from = str(trade_df.index.min()) if not trade_df.empty else None
    active_to = str(trade_df.index.max()) if state == "closed" and not trade_df.empty else None

    return {
        "fold_id": fold.fold_id,
        "runtime_status": state,
        "anchor_index": int(fold.anchor_index),
        "train_ready_row_count": int(fold.train_end),
        "activation_row_count": int(fold.trade_start + 1),
        "close_row_count": int(fold.trade_end),
        "train_start": train_start,
        "train_end": train_end,
        "trained_until": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "trade_start": active_from,
        "trade_end": active_to,
        "active_from": active_from,
        "active_to": active_to,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "trade_rows": trade_rows,
    }


def _build_runtime_manifest_record(
    fold_record: Dict,
    runtime_root: str,
    symbol: str,
    timeframe: str,
) -> Dict:
    return {
        "fold_id": fold_record["fold_id"],
        "active_from": _format_mql_optional(fold_record["active_from"]),
        "active_to": _format_mql_optional(fold_record["active_to"]),
        "trained_until": _format_mql_optional(fold_record["trained_until"]),
        "model_path": _runtime_relative_path(runtime_root, "folds", fold_record["fold_id"], "model.onnx"),
        "metadata_path": _runtime_relative_path(runtime_root, "folds", fold_record["fold_id"], "model_metadata.json"),
        "train_rows": int(fold_record["train_rows"]),
        "validation_rows": int(fold_record["validation_rows"]),
        "trade_rows": int(fold_record["trade_rows"]),
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "status": fold_record["runtime_status"],
    }


def _can_reuse_runtime_fold_export(
    fold_dir: Path,
    fold_record: Dict,
    configuration_hash: str,
    data_hash: str,
) -> bool:
    if not (fold_dir / "model.onnx").is_file():
        return False
    if not (fold_dir / "model_metadata.json").is_file():
        return False
    if not (fold_dir / "runtime_contract.csv").is_file():
        return False

    fold_info = _load_json(fold_dir / "fold_info.json")
    if not fold_info:
        return False

    if fold_info.get("config_hash") != configuration_hash or fold_info.get("data_hash") != data_hash:
        return False
    if int(fold_info.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
        return False

    expected_pairs = {
        "train_start": fold_record["train_start"],
        "train_end": fold_record["train_end"],
        "trained_until": fold_record["trained_until"],
    }
    for key, expected_value in expected_pairs.items():
        if _stringify_optional(fold_info.get(key)) != _stringify_optional(expected_value):
            return False

    expected_rows = {
        "rows_train": fold_record["train_rows"],
        "rows_validation": fold_record["validation_rows"],
        "rows_trade": fold_record["trade_rows"],
        "planned_train_ready_row_count": fold_record["train_ready_row_count"],
        "planned_activation_row_count": fold_record["activation_row_count"],
        "planned_close_row_count": fold_record["close_row_count"],
    }
    for key, expected_value in expected_rows.items():
        if int(fold_info.get(key, -1)) != int(expected_value):
            return False

    return True


def _load_existing_live_tail_info(export_dir: Path | None) -> Dict | None:
    if export_dir is None:
        return None
    return _load_json(export_dir / "live" / "live_tail_info.json")


def _can_reuse_live_tail(
    existing_live_info: Dict | None,
    export_dir: Path,
    configuration_hash: str,
    train_start: str | None,
    trained_until: str | None,
    rows_train: int,
    data_hash: str,
) -> bool:
    if not existing_live_info:
        return False

    if existing_live_info.get("config_hash") != configuration_hash or existing_live_info.get("data_hash") != data_hash:
        return False
    if int(existing_live_info.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
        return False
    if _stringify_optional(existing_live_info.get("train_start")) != _stringify_optional(train_start):
        return False
    if _stringify_optional(existing_live_info.get("trained_until")) != _stringify_optional(trained_until):
        return False
    if int(existing_live_info.get("rows_train", -1)) != int(rows_train):
        return False

    return (
        (export_dir / "live" / "model.onnx").exists()
        and (export_dir / "live" / "model_metadata.json").exists()
        and (export_dir / "live" / "runtime_contract.csv").exists()
    )


def _training_validation_sample(
    train_df: pd.DataFrame, history_rows: int, cfg: Dict, validate_onnx: bool
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not validate_onnx:
        return None, None
    validation_rows = int(cfg["export"].get("validate_rows", 32))
    if validation_rows < 1:
        raise ValueError("export.validate_rows deve ser maior que zero")
    sample_start = max(history_rows, len(train_df) - validation_rows)
    return train_df.iloc[sample_start:], _history_slice(train_df, sample_start, history_rows)


def export_live_tail_model(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    cfg: Dict,
    policy: WalkForwardPolicy,
    export_dir: Path | None,
    validate_onnx: bool,
    runtime_root: str,
    symbol: str,
    timeframe: str,
    input_csv: str,
    configuration_hash: str,
    incremental_export: bool,
    enable_progress: bool,
    progress_label: str,
) -> Dict:
    result = {
        "available": False,
        "reused": False,
        "train_start": None,
        "trained_until": None,
        "rows_train": 0,
        "runtime_model_path": _runtime_relative_path(runtime_root, "live", "model.onnx"),
        "runtime_metadata_path": _runtime_relative_path(runtime_root, "live", "model_metadata.json"),
        "runtime_contract_path": _runtime_relative_path(runtime_root, "live", "runtime_contract.csv"),
    }

    if export_dir is None or dataset.empty:
        return result

    train_end = len(dataset)
    train_start = max(0, train_end - policy.train_window)
    train_df = _df_slice(dataset, train_start, train_end)
    if train_df.empty:
        return result

    result["train_start"] = str(train_df.index.min())
    result["trained_until"] = str(train_df.index.max())
    result["rows_train"] = int(len(train_df))

    live_dir = export_dir / "live"
    ensure_dir(live_dir)

    data_hash = _dataset_hash(train_df)
    existing_live_info = _load_existing_live_tail_info(export_dir)
    if incremental_export and _can_reuse_live_tail(
        existing_live_info=existing_live_info,
        export_dir=export_dir,
        configuration_hash=configuration_hash,
        train_start=result["train_start"],
        trained_until=result["trained_until"],
        rows_train=result["rows_train"],
        data_hash=data_hash,
    ):
        result["available"] = True
        result["reused"] = True
        result["runtime_model_path"] = existing_live_info.get("runtime_model_path", result["runtime_model_path"])
        result["runtime_metadata_path"] = existing_live_info.get("runtime_metadata_path", result["runtime_metadata_path"])
        if enable_progress:
            label = progress_label if progress_label else "walkforward"
            print(
                "[wf] "
                f"stage=live_tail "
                f"label={label} "
                f"status=reused "
                f"rows_train={result['rows_train']} "
                f"trained_until={result['trained_until']}",
                flush=True,
            )
        return result

    ok, issues = validate_train_classes(train_df, min_class_count=policy.min_class_count_per_train)
    if not ok:
        result["skip_reason"] = " | ".join(issues)
        return result

    model = build_model_backend(cfg)
    model.fit(train_df, feature_columns)

    sample_df, sample_history_df = _training_validation_sample(train_df, model.history_rows_needed, cfg, validate_onnx)
    onnx_contract, validation_payload = export_regime_onnx(
        model=model,
        output_path=live_dir / "model.onnx",
        opset=int(cfg["export"]["onnx_opset"]),
        feature_columns=feature_columns,
        sample_df=sample_df,
        history_df=sample_history_df,
    )
    export_market_runtime_contract(
        destination=live_dir / "runtime_contract.csv",
        cfg=cfg,
        feature_columns=feature_columns,
        onnx_contract=onnx_contract,
    )

    live_tail_info = {
        "mode": "live_tail",
        "symbol": symbol,
        "timeframe": timeframe,
        "train_start": result["train_start"],
        "train_end": result["trained_until"],
        "trained_until": result["trained_until"],
        "rows_train": result["rows_train"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": configuration_hash,
        "data_hash": data_hash,
        "runtime_model_path": result["runtime_model_path"],
        "runtime_metadata_path": result["runtime_metadata_path"],
        "runtime_contract_path": result["runtime_contract_path"],
        "export_dir": str(live_dir),
        "input_csv": input_csv,
    }

    metadata = {
        "project": "Market Regime ONNX Live Tail",
        "input_csv": input_csv,
        "input_name": onnx_contract["input_name"],
        "output_names": onnx_contract["output_names"],
        "feature_columns": list(feature_columns),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "symbol": symbol,
        "timeframe": timeframe,
        "train_rows": int(len(train_df)),
        "validation_rows": 0,
        "trade_rows": 0,
        "metrics": {},
        "onnx_validation": validation_payload,
        "live_tail_info": live_tail_info,
        "config": cfg,
        "walkforward_policy": asdict(policy),
    }

    save_json(live_dir / "live_tail_info.json", live_tail_info)
    save_json(live_dir / "model_metadata.json", metadata)

    result["available"] = True
    result["reused"] = False

    if enable_progress:
        label = progress_label if progress_label else "walkforward"
        print(
            "[wf] "
            f"stage=live_tail "
            f"label={label} "
            f"status={'reused' if result['reused'] else 'trained'} "
            f"rows_train={result['rows_train']} "
            f"trained_until={result['trained_until']}",
            flush=True,
        )

    return result


def export_runtime_fold_model(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    cfg: Dict,
    policy: WalkForwardPolicy,
    fold: FoldDefinition,
    export_dir: Path | None,
    validate_onnx: bool,
    runtime_root: str,
    symbol: str,
    timeframe: str,
    input_csv: str,
    configuration_hash: str,
    incremental_export: bool,
    enable_progress: bool,
    progress_label: str,
) -> Dict:
    fold_record = _build_runtime_fold_record(dataset, fold)
    runtime_status = fold_record["runtime_status"]
    result = {
        "available": False,
        "reused": False,
        "runtime_status": runtime_status,
        "fold_record": fold_record,
        "manifest_record": None,
        "skip_reason": "",
    }

    if export_dir is None or runtime_status not in {"staged", "open"}:
        return result

    train_df = _df_slice(dataset, fold.train_start, fold.train_end)
    ok, issues = validate_train_classes(train_df, min_class_count=policy.min_class_count_per_train)
    if not ok:
        result["skip_reason"] = " | ".join(issues)
        return result

    fold_dir = export_dir / "folds" / fold.fold_id
    ensure_dir(fold_dir)
    runtime_model_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "model.onnx")
    runtime_metadata_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "model_metadata.json")
    runtime_contract_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "runtime_contract.csv")

    validation_payload = {"status": "disabled"}
    existing_metadata = None
    data_hash = _dataset_hash(train_df)
    reused = incremental_export and _can_reuse_runtime_fold_export(fold_dir, fold_record, configuration_hash, data_hash)
    if reused:
        existing_metadata = _load_json(fold_dir / "model_metadata.json") or {}
        validation_payload = existing_metadata.get("onnx_validation", validation_payload)
        input_name = existing_metadata.get("input_name")
        output_names = existing_metadata.get("output_names", [])
        metadata_feature_columns = existing_metadata.get("feature_columns", list(feature_columns))
    else:
        model = build_model_backend(cfg)
        model.fit(train_df, feature_columns)

        sample_df, sample_history_df = _training_validation_sample(
            train_df, model.history_rows_needed, cfg, validate_onnx
        )
        onnx_contract, validation_payload = export_regime_onnx(
            model=model,
            output_path=fold_dir / "model.onnx",
            opset=int(cfg["export"]["onnx_opset"]),
            feature_columns=feature_columns,
            sample_df=sample_df,
            history_df=sample_history_df,
        )
        export_market_runtime_contract(
            destination=fold_dir / "runtime_contract.csv",
            cfg=cfg,
            feature_columns=feature_columns,
            onnx_contract=onnx_contract,
        )
        input_name = onnx_contract["input_name"]
        output_names = onnx_contract["output_names"]
        metadata_feature_columns = list(feature_columns)

    fold_info = {
        "fold_id": fold.fold_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "train_start": fold_record["train_start"],
        "train_end": fold_record["train_end"],
        "trained_until": fold_record["trained_until"],
        "validation_start": fold_record["validation_start"],
        "validation_end": fold_record["validation_end"],
        "trade_start": fold_record["trade_start"],
        "trade_end": fold_record["trade_end"],
        "active_from": fold_record["active_from"],
        "active_to": fold_record["active_to"],
        "rows_train": fold_record["train_rows"],
        "rows_validation": fold_record["validation_rows"],
        "rows_trade": fold_record["trade_rows"],
        "purge_bars": int(policy.purge_bars),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": configuration_hash,
        "data_hash": data_hash,
        "runtime_model_path": runtime_model_path,
        "runtime_metadata_path": runtime_metadata_path,
        "runtime_contract_path": runtime_contract_path,
        "runtime_status": runtime_status,
        "planned_train_ready_row_count": fold_record["train_ready_row_count"],
        "planned_activation_row_count": fold_record["activation_row_count"],
        "planned_close_row_count": fold_record["close_row_count"],
        "metrics": {},
    }
    save_json(fold_dir / "fold_info.json", fold_info)

    metadata = {
        "project": "Market Regime ONNX Runtime Fold",
        "input_csv": input_csv,
        "input_name": input_name,
        "output_names": output_names,
        "feature_columns": metadata_feature_columns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "symbol": symbol,
        "timeframe": timeframe,
        "train_rows": int(fold_record["train_rows"]),
        "validation_rows": int(fold_record["validation_rows"]),
        "trade_rows": int(fold_record["trade_rows"]),
        "metrics": {},
        "onnx_validation": validation_payload,
        "fold_info": fold_info,
        "config": cfg,
        "walkforward_policy": asdict(policy),
        "runtime_status": runtime_status,
    }
    save_json(fold_dir / "model_metadata.json", metadata)

    result["available"] = True
    result["reused"] = reused
    result["manifest_record"] = _build_runtime_manifest_record(
        fold_record=fold_record,
        runtime_root=runtime_root,
        symbol=symbol,
        timeframe=timeframe,
    )

    if enable_progress:
        label = progress_label if progress_label else "walkforward"
        print(
            "[wf] "
            f"stage=runtime_fold "
            f"label={label} "
            f"fold_id={fold.fold_id} "
            f"runtime_status={runtime_status} "
            f"status={'reused' if reused else 'trained'} "
            f"active_from={fold_record['active_from'] if fold_record['active_from'] else 'pending'}",
            flush=True,
        )

    return result


def build_next_runtime_transition(
    dataset: pd.DataFrame,
    policy: WalkForwardPolicy,
    runtime_ready_records: Sequence[Dict],
) -> Dict | None:
    total_rows = len(dataset)
    candidates: List[Dict] = []

    for fold_record in runtime_ready_records:
        runtime_status = str(fold_record.get("runtime_status", ""))
        if runtime_status == "staged":
            candidates.append(
                {
                    "fold_id": fold_record["fold_id"],
                    "stage": "activate",
                    "reason": "runtime_fold_boundary",
                    "row_count": int(fold_record["activation_row_count"]),
                    "train_ready_row_count": int(fold_record["train_ready_row_count"]),
                    "activation_row_count": int(fold_record["activation_row_count"]),
                    "close_row_count": int(fold_record["close_row_count"]),
                    "trained_until": fold_record["trained_until"],
                    "active_from": fold_record["active_from"],
                    "active_to": fold_record["active_to"],
                }
            )
        elif runtime_status == "open":
            candidates.append(
                {
                    "fold_id": fold_record["fold_id"],
                    "stage": "close",
                    "reason": "runtime_fold_boundary",
                    "row_count": int(fold_record["close_row_count"]),
                    "train_ready_row_count": int(fold_record["train_ready_row_count"]),
                    "activation_row_count": int(fold_record["activation_row_count"]),
                    "close_row_count": int(fold_record["close_row_count"]),
                    "trained_until": fold_record["trained_until"],
                    "active_from": fold_record["active_from"],
                    "active_to": fold_record["active_to"],
                }
            )

    next_pending_fold = build_fold_definition(policy, len(runtime_ready_records))
    if next_pending_fold.train_end > total_rows:
        next_pending_record = _build_runtime_fold_record(dataset, next_pending_fold)
        candidates.append(
            {
                "fold_id": next_pending_record["fold_id"],
                "stage": "train_ready",
                "reason": "runtime_fold_boundary",
                "row_count": int(next_pending_record["train_ready_row_count"]),
                "train_ready_row_count": int(next_pending_record["train_ready_row_count"]),
                "activation_row_count": int(next_pending_record["activation_row_count"]),
                "close_row_count": int(next_pending_record["close_row_count"]),
                "trained_until": next_pending_record["trained_until"],
                "active_from": next_pending_record["active_from"],
                "active_to": next_pending_record["active_to"],
            }
        )

    if not candidates:
        return None

    return min(candidates, key=lambda item: (int(item["row_count"]), str(item["fold_id"])))


def evaluate_walkforward(
    candles: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    cfg: Dict,
    policy: WalkForwardPolicy,
    export_dir: Path | None = None,
    validate_onnx: bool = False,
    runtime_root: str = "ML_MarketRegime_ONNX\\walkforward",
    symbol: str = "",
    timeframe: str = "",
    input_csv: str = "",
    enable_progress: bool = False,
    progress_label: str = "",
    fold_progress_every: int = 1,
    incremental_export: bool = False,
) -> Dict:
    resolve_purge_bars(cfg, policy.purge_bars)
    available_folds = generate_walkforward_folds(dataset, policy, apply_cap=False)
    folds = generate_walkforward_folds(dataset, policy, apply_cap=True)
    runtime_ready_folds = generate_train_ready_folds(dataset, policy)
    summary_records: List[Dict] = []
    oos_predictions: List[pd.DataFrame] = []
    research_manifest_records: List[Dict] = []
    configuration_hash = cfg_hash(
        cfg,
        policy,
        {
            "validate_onnx": validate_onnx,
            "runtime_root": runtime_root,
            "symbol": symbol,
            "timeframe": timeframe,
            "input_csv": input_csv,
        },
    )
    walkforward_start = time.perf_counter()
    existing_state = _load_existing_export_state(
        export_dir=export_dir,
        policy=policy,
        configuration_hash=configuration_hash,
        incremental_export=incremental_export,
    )
    reused_folds = 0
    trained_folds = 0
    skipped_folds = 0

    if enable_progress:
        label = progress_label if progress_label else "walkforward"
        print(
            "[wf] "
            f"stage=start "
            f"label={label} "
            f"available_folds={len(available_folds)} "
            f"planned_folds={len(folds)} "
            f"runtime_ready_folds={len(runtime_ready_folds)} "
            f"train_window={policy.train_window} "
            f"trade_window={policy.trade_window} "
            f"step_window={policy.step_window} "
            f"min_class_count={policy.min_class_count_per_train} "
            f"max_folds={policy.max_folds} "
            f"incremental_export={int(incremental_export)} "
            f"reuse_ready={int(existing_state['enabled'])}",
            flush=True,
        )

    if export_dir is not None:
        ensure_dir(export_dir)
        ensure_dir(export_dir / "folds")

    total_folds = len(folds)
    for fold_index, fold in enumerate(folds, start=1):
        fold_start_time = time.perf_counter()
        train_df = _df_slice(dataset, fold.train_start, fold.train_end)
        validation_df = _df_slice(dataset, fold.validation_start, fold.validation_end)
        trade_df = _df_slice(dataset, fold.trade_start, fold.trade_end)

        fold_record = {
            "fold_id": fold.fold_id,
            "status": "ok",
            "skip_reason": "",
            "train_start": str(train_df.index.min()) if not train_df.empty else None,
            "train_end": str(train_df.index.max()) if not train_df.empty else None,
            "trained_until": str(train_df.index.max()) if not train_df.empty else None,
            "validation_start": str(validation_df.index.min()) if not validation_df.empty else None,
            "validation_end": str(validation_df.index.max()) if not validation_df.empty else None,
            "trade_start": str(trade_df.index.min()) if not trade_df.empty else None,
            "trade_end": str(trade_df.index.max()) if not trade_df.empty else None,
            "active_from": str(trade_df.index.min()) if not trade_df.empty else None,
            "active_to": str(trade_df.index.max()) if not trade_df.empty else None,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "trade_rows": int(len(trade_df)),
            "train_class_distribution": json.dumps(class_distribution(train_df["regime_id"])) if not train_df.empty else "{}",
            "validation_class_distribution": json.dumps(class_distribution(validation_df["regime_id"])) if not validation_df.empty else "{}",
            "trade_class_distribution": json.dumps(class_distribution(trade_df["regime_id"])) if not trade_df.empty else "{}",
            "validation_accuracy": None,
            "validation_balanced_accuracy": None,
            "trade_accuracy": None,
            "trade_balanced_accuracy": None,
        }

        ok, issues = validate_train_classes(train_df, min_class_count=policy.min_class_count_per_train)
        if not ok:
            fold_record["status"] = "skipped"
            fold_record["skip_reason"] = " | ".join(issues)
            summary_records.append(fold_record)
            skipped_folds += 1
            if enable_progress and should_log_progress(fold_index, total_folds, fold_progress_every):
                elapsed = time.perf_counter() - walkforward_start
                avg_per_fold = elapsed / fold_index
                eta = avg_per_fold * (total_folds - fold_index)
                label = progress_label if progress_label else "walkforward"
                print(
                    "[wf] "
                    f"stage=fold "
                    f"label={label} "
                    f"fold={fold_index}/{total_folds} "
                    f"fold_id={fold.fold_id} "
                    f"status=skipped "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)} "
                    f"reason={fold_record['skip_reason']}",
                    flush=True,
                )
            continue

        fold_dir = export_dir / "folds" / fold.fold_id if export_dir is not None else None
        existing_summary = existing_state["summary_by_fold"].get(fold.fold_id)
        existing_manifest = existing_state["manifest_by_fold"].get(fold.fold_id)
        data_hash = _dataset_hash(dataset.iloc[fold.train_start:fold.trade_end]) if fold_dir is not None else ""
        existing_predictions = None
        if (
            existing_state["enabled"]
            and existing_summary is not None
            and existing_manifest is not None
            and fold_dir is not None
            and _can_reuse_exported_fold(fold_dir, fold_record, configuration_hash, data_hash)
        ):
            existing_predictions = _load_existing_predictions(fold_dir)

        if existing_predictions is not None:
            summary_records.append(existing_summary)
            research_manifest_records.append(
                {
                    "fold_id": str(existing_manifest.get("fold_id", fold.fold_id)),
                    "active_from": existing_manifest.get("active_from"),
                    "active_to": existing_manifest.get("active_to"),
                    "trained_until": existing_manifest.get("trained_until"),
                    "model_path": existing_manifest.get("model_path"),
                    "metadata_path": existing_manifest.get("metadata_path"),
                    "train_rows": int(existing_manifest.get("train_rows", fold_record["train_rows"])),
                    "validation_rows": int(existing_manifest.get("validation_rows", fold_record["validation_rows"])),
                    "trade_rows": int(existing_manifest.get("trade_rows", fold_record["trade_rows"])),
                    "symbol": str(existing_manifest.get("symbol", symbol)),
                    "timeframe": str(existing_manifest.get("timeframe", timeframe)),
                    "feature_schema_version": int(existing_manifest.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
                }
            )

            oos_predictions.append(existing_predictions)

            reused_folds += 1
            if enable_progress and should_log_progress(fold_index, total_folds, fold_progress_every):
                elapsed = time.perf_counter() - walkforward_start
                avg_per_fold = elapsed / fold_index
                eta = avg_per_fold * (total_folds - fold_index)
                label = progress_label if progress_label else "walkforward"
                print(
                    "[wf] "
                    f"stage=fold "
                    f"label={label} "
                    f"fold={fold_index}/{total_folds} "
                    f"fold_id={fold.fold_id} "
                    f"status=reused "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)}",
                    flush=True,
                )
            continue

        model = build_model_backend(cfg)
        model.fit(train_df, feature_columns)
        validation_history_df = _history_slice(dataset, fold.validation_start, model.history_rows_needed)
        trade_history_df = _history_slice(dataset, fold.trade_start, model.history_rows_needed)

        validation_metrics = None
        if not validation_df.empty:
            validation_metrics, _validation_confusion_df, _validation_predictions_df = evaluate_model(
                model,
                validation_df,
                feature_columns,
                history_df=validation_history_df,
            )
            fold_record["validation_accuracy"] = float(validation_metrics["accuracy"])
            fold_record["validation_balanced_accuracy"] = float(validation_metrics["balanced_accuracy"])

        trade_metrics, confusion_df, predictions_df = evaluate_model(
            model,
            trade_df,
            feature_columns,
            history_df=trade_history_df,
        )
        fold_record["trade_accuracy"] = float(trade_metrics["accuracy"])
        fold_record["trade_balanced_accuracy"] = float(trade_metrics["balanced_accuracy"])

        predictions_with_fold = predictions_df.reset_index().copy()
        predictions_with_fold["fold_id"] = fold.fold_id
        predictions_with_fold["active_from"] = fold_record["active_from"]
        predictions_with_fold["active_to"] = fold_record["active_to"]
        predictions_with_fold["trained_until"] = fold_record["trained_until"]
        oos_predictions.append(predictions_with_fold)

        if export_dir is not None:
            ensure_dir(fold_dir)

            sample_df = None
            if validate_onnx:
                validation_rows = int(cfg["export"].get("validate_rows", 32))
                if validation_rows < 1:
                    raise ValueError("export.validate_rows deve ser maior que zero")
                sample_df = trade_df.iloc[:validation_rows]
            onnx_contract, validation_payload = export_regime_onnx(
                model=model,
                output_path=fold_dir / "model.onnx",
                opset=int(cfg["export"]["onnx_opset"]),
                feature_columns=feature_columns,
                sample_df=sample_df,
                history_df=trade_history_df,
            )
            export_market_runtime_contract(
                destination=fold_dir / "runtime_contract.csv",
                cfg=cfg,
                feature_columns=feature_columns,
                onnx_contract=onnx_contract,
            )

            confusion_df.to_csv(fold_dir / "confusion_matrix.csv")
            predictions_with_fold.to_csv(fold_dir / "trade_predictions.csv", index=False)
            save_json(fold_dir / "classification_report.json", trade_metrics["classification_report"])

            runtime_model_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "model.onnx")
            runtime_metadata_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "model_metadata.json")
            runtime_contract_path = _runtime_relative_path(runtime_root, "folds", fold.fold_id, "runtime_contract.csv")

            fold_info = {
                "fold_id": fold.fold_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "train_start": fold_record["train_start"],
                "train_end": fold_record["train_end"],
                "trained_until": fold_record["trained_until"],
                "validation_start": fold_record["validation_start"],
                "validation_end": fold_record["validation_end"],
                "trade_start": fold_record["trade_start"],
                "trade_end": fold_record["trade_end"],
                "active_from": fold_record["active_from"],
                "active_to": fold_record["active_to"],
                "rows_train": fold_record["train_rows"],
                "rows_validation": fold_record["validation_rows"],
                "rows_trade": fold_record["trade_rows"],
                "purge_bars": int(policy.purge_bars),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "config_hash": configuration_hash,
                "data_hash": data_hash,
                "runtime_model_path": runtime_model_path,
                "runtime_metadata_path": runtime_metadata_path,
                "runtime_contract_path": runtime_contract_path,
                "runtime_status": "closed",
                "planned_train_ready_row_count": int(fold.train_end),
                "planned_activation_row_count": int(fold.trade_start + 1),
                "planned_close_row_count": int(fold.trade_end),
                "metrics": {
                    "trade_accuracy": trade_metrics["accuracy"],
                    "trade_balanced_accuracy": trade_metrics["balanced_accuracy"],
                    "validation_accuracy": validation_metrics["accuracy"] if validation_metrics else None,
                    "validation_balanced_accuracy": validation_metrics["balanced_accuracy"] if validation_metrics else None,
                },
            }
            save_json(fold_dir / "fold_info.json", fold_info)

            metadata = {
                "project": "Market Regime ONNX WalkForward",
                "input_csv": input_csv,
                "input_name": onnx_contract["input_name"],
                "output_names": onnx_contract["output_names"],
                "feature_columns": list(feature_columns),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
                "symbol": symbol,
                "timeframe": timeframe,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "trade_rows": int(len(trade_df)),
                "metrics": {
                    "trade_accuracy": trade_metrics["accuracy"],
                    "trade_balanced_accuracy": trade_metrics["balanced_accuracy"],
                    "validation_accuracy": validation_metrics["accuracy"] if validation_metrics else None,
                    "validation_balanced_accuracy": validation_metrics["balanced_accuracy"] if validation_metrics else None,
                },
                "onnx_validation": validation_payload,
                "fold_info": fold_info,
                "config": cfg,
                "walkforward_policy": asdict(policy),
                "runtime_status": "closed",
            }
            save_json(fold_dir / "model_metadata.json", metadata)

            research_manifest_records.append(
                {
                    "fold_id": fold.fold_id,
                    "active_from": format_mql_time(trade_df.index.min()),
                    "active_to": format_mql_time(trade_df.index.max()),
                    "trained_until": format_mql_time(train_df.index.max()),
                    "model_path": runtime_model_path,
                    "metadata_path": runtime_metadata_path,
                    "train_rows": int(len(train_df)),
                    "validation_rows": int(len(validation_df)),
                    "trade_rows": int(len(trade_df)),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                }
            )

        summary_records.append(fold_record)
        trained_folds += 1
        if enable_progress and should_log_progress(fold_index, total_folds, fold_progress_every):
            elapsed = time.perf_counter() - walkforward_start
            avg_per_fold = elapsed / fold_index
            eta = avg_per_fold * (total_folds - fold_index)
            label = progress_label if progress_label else "walkforward"
            print(
                "[wf] "
                f"stage=fold "
                f"label={label} "
                f"fold={fold_index}/{total_folds} "
                f"fold_id={fold.fold_id} "
                f"status=ok "
                f"trade_bal_acc={fold_record['trade_balanced_accuracy']:.4f} "
                f"last_fold={format_duration(time.perf_counter() - fold_start_time)} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)}",
                flush=True,
            )

    summary_df = pd.DataFrame(summary_records)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "fold_id",
                "status",
                "skip_reason",
                "train_start",
                "train_end",
                "trained_until",
                "validation_start",
                "validation_end",
                "trade_start",
                "trade_end",
                "active_from",
                "active_to",
                "train_rows",
                "validation_rows",
                "trade_rows",
                "train_class_distribution",
                "validation_class_distribution",
                "trade_class_distribution",
                "validation_accuracy",
                "validation_balanced_accuracy",
                "trade_accuracy",
                "trade_balanced_accuracy",
            ]
        )

    aggregate = aggregate_walkforward_summary(summary_df, policy)
    aggregate["available_folds"] = int(len(available_folds))
    aggregate["reused_folds"] = int(reused_folds)
    aggregate["trained_folds"] = int(trained_folds)
    aggregate["skipped_folds"] = int(skipped_folds)

    live_tail = export_live_tail_model(
        dataset=dataset,
        feature_columns=feature_columns,
        cfg=cfg,
        policy=policy,
        export_dir=export_dir,
        validate_onnx=validate_onnx,
        runtime_root=runtime_root,
        symbol=symbol,
        timeframe=timeframe,
        input_csv=input_csv,
        configuration_hash=configuration_hash,
        incremental_export=incremental_export,
        enable_progress=enable_progress,
        progress_label=progress_label,
    )
    aggregate["live_tail_available"] = bool(live_tail["available"])
    aggregate["live_tail_reused"] = bool(live_tail["reused"])
    aggregate["live_tail_trained_until"] = live_tail["trained_until"]

    runtime_manifest_records: List[Dict] = []
    runtime_fold_records: List[Dict] = []
    staged_folds: List[Dict] = []
    runtime_unavailable_folds: List[Dict] = []
    runtime_reused_folds = 0
    runtime_trained_folds = 0

    research_manifest_lookup = {record["fold_id"]: record for record in research_manifest_records}
    for fold in runtime_ready_folds:
        runtime_fold_record = _build_runtime_fold_record(dataset, fold)
        runtime_fold_records.append(runtime_fold_record)
        runtime_status = runtime_fold_record["runtime_status"]

        if runtime_status == "closed":
            existing_record = research_manifest_lookup.get(fold.fold_id)
            if existing_record is not None:
                runtime_manifest_records.append({**existing_record, "status": "closed"})
            continue

        runtime_export = export_runtime_fold_model(
            dataset=dataset,
            feature_columns=feature_columns,
            cfg=cfg,
            policy=policy,
            fold=fold,
            export_dir=export_dir,
            validate_onnx=validate_onnx,
            runtime_root=runtime_root,
            symbol=symbol,
            timeframe=timeframe,
            input_csv=input_csv,
            configuration_hash=configuration_hash,
            incremental_export=incremental_export,
            enable_progress=enable_progress,
            progress_label=progress_label,
        )

        if not runtime_export["available"]:
            runtime_unavailable_folds.append(
                {
                    "fold_id": fold.fold_id,
                    "status": runtime_status,
                    "skip_reason": runtime_export["skip_reason"],
                    "train_ready_row_count": int(runtime_fold_record["train_ready_row_count"]),
                    "activation_row_count": int(runtime_fold_record["activation_row_count"]),
                    "close_row_count": int(runtime_fold_record["close_row_count"]),
                }
            )
            continue

        if runtime_export["reused"]:
            runtime_reused_folds += 1
        else:
            runtime_trained_folds += 1

        manifest_record = runtime_export["manifest_record"]
        if runtime_status == "open" and manifest_record is not None:
            runtime_manifest_records.append(manifest_record)
        elif runtime_status == "staged" and manifest_record is not None:
            staged_folds.append(
                {
                    **manifest_record,
                    "train_ready_row_count": int(runtime_fold_record["train_ready_row_count"]),
                    "activation_row_count": int(runtime_fold_record["activation_row_count"]),
                    "close_row_count": int(runtime_fold_record["close_row_count"]),
                }
            )

    runtime_manifest_records.sort(key=lambda record: str(record.get("fold_id", "")))
    next_runtime_transition = build_next_runtime_transition(
        dataset=dataset,
        policy=policy,
        runtime_ready_records=runtime_fold_records,
    )

    oos_df = pd.concat(oos_predictions, ignore_index=True) if oos_predictions else pd.DataFrame()

    research_manifest_payload = {
        "project": "Market Regime ONNX WalkForward Research",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": configuration_hash,
        "policy": asdict(policy),
        "aggregate_metrics": aggregate,
        "folds": research_manifest_records,
        "live_tail": live_tail,
    }

    runtime_manifest_payload = {
        "project": "Market Regime ONNX WalkForward Runtime",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": configuration_hash,
        "policy": asdict(policy),
        "latest_dataset_time": str(dataset.index.max()) if not dataset.empty else None,
        "latest_dataset_rows": int(len(dataset)),
        "runtime_ready_folds": int(len(runtime_ready_folds)),
        "runtime_closed_folds": int(sum(1 for record in runtime_manifest_records if record.get("status") == "closed")),
        "runtime_open_folds": int(sum(1 for record in runtime_manifest_records if record.get("status") == "open")),
        "runtime_staged_folds": int(len(staged_folds)),
        "runtime_reused_folds": int(runtime_reused_folds),
        "runtime_trained_folds": int(runtime_trained_folds),
        "folds": runtime_manifest_records,
        "staged_folds": staged_folds,
        "unavailable_folds": runtime_unavailable_folds,
        "next_refresh_reason": next_runtime_transition["reason"] if next_runtime_transition else None,
        "next_refresh_stage": next_runtime_transition["stage"] if next_runtime_transition else None,
        "next_refresh_row_count": int(next_runtime_transition["row_count"]) if next_runtime_transition else None,
        "next_fold_id": next_runtime_transition["fold_id"] if next_runtime_transition else None,
        "next_fold_train_ready_at": _format_mql_optional(next_runtime_transition["trained_until"]) if next_runtime_transition else None,
        "next_fold_active_from": _format_mql_optional(next_runtime_transition["active_from"]) if next_runtime_transition else None,
        "next_fold_active_to": _format_mql_optional(next_runtime_transition["active_to"]) if next_runtime_transition else None,
        "next_fold_train_ready_row_count": int(next_runtime_transition["train_ready_row_count"]) if next_runtime_transition else None,
        "next_fold_active_row_count": int(next_runtime_transition["activation_row_count"]) if next_runtime_transition else None,
        "next_fold_close_row_count": int(next_runtime_transition["close_row_count"]) if next_runtime_transition else None,
        "live_tail": live_tail,
    }

    if export_dir is not None:
        summary_df.to_csv(export_dir / "summary.csv", index=False)
        oos_df.to_csv(export_dir / "oos_predictions.csv", index=False)
        save_json(export_dir / "research_manifest.json", research_manifest_payload)
        save_json(export_dir / "manifest.json", runtime_manifest_payload)
        save_json(export_dir / "walkforward_report.json", aggregate)
        pd.DataFrame(
            research_manifest_records,
            columns=[
                "fold_id",
                "active_from",
                "active_to",
                "trained_until",
                "model_path",
                "metadata_path",
                "train_rows",
                "validation_rows",
                "trade_rows",
                "symbol",
                "timeframe",
                "feature_schema_version",
            ],
        ).to_csv(export_dir / "research_manifest.csv", index=False)
        pd.DataFrame(
            runtime_manifest_records,
            columns=[
                "fold_id",
                "active_from",
                "active_to",
                "trained_until",
                "model_path",
                "metadata_path",
                "train_rows",
                "validation_rows",
                "trade_rows",
                "symbol",
                "timeframe",
                "feature_schema_version",
                "status",
            ],
        ).to_csv(export_dir / "manifest.csv", index=False)

    if enable_progress:
        label = progress_label if progress_label else "walkforward"
        print(
            "[wf] "
            f"stage=done "
            f"label={label} "
            f"planned_folds={aggregate['planned_folds']} "
            f"valid_folds={aggregate['valid_folds']} "
            f"reused_folds={aggregate['reused_folds']} "
            f"trained_folds={aggregate['trained_folds']} "
            f"runtime_open_folds={runtime_manifest_payload['runtime_open_folds']} "
            f"runtime_staged_folds={runtime_manifest_payload['runtime_staged_folds']} "
            f"elapsed={format_duration(time.perf_counter() - walkforward_start)} "
            f"score={aggregate['score'] if aggregate['score'] is not None else 'nan'} "
            f"mean_bal_acc={aggregate['mean_balanced_accuracy_trade'] if aggregate['mean_balanced_accuracy_trade'] is not None else 'nan'}",
            flush=True,
        )

    return {
        "policy": asdict(policy),
        "summary": summary_df,
        "aggregate": aggregate,
        "oos_predictions": oos_df,
        "research_manifest": research_manifest_payload,
        "runtime_manifest": runtime_manifest_payload,
        "manifest": runtime_manifest_payload,
    }
