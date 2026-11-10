from __future__ import annotations

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from market_regime_core import (
    build_feature_frame,
    build_model_backend,
    chronological_split,
    dataset_diagnostics,
    evaluate_model,
    load_candles_csv,
    load_config,
    normalize_model_type,
)
from walkforward import WalkForwardPolicy, evaluate_walkforward, format_duration, resolve_purge_bars


@dataclass(frozen=True)
class SearchFamilySpace:
    model_type: str
    evaluation_mode: str
    gene_order: Tuple[str, ...]
    genes: Dict[str, Tuple[Any, ...]]

    @property
    def total_candidates(self) -> int:
        if not self.gene_order:
            return 0
        total = 1
        for gene_name in self.gene_order:
            total *= len(self.genes.get(gene_name, ()))
        return total


@dataclass(frozen=True)
class CandidateSpec:
    model_type: str
    horizon: int
    threshold: float
    train_window: int | None
    validation_window: int | None
    trade_window: int | None
    step_window: int | None
    min_class_count: int | None
    cfg_updates: Dict[str, Any]
    record_fields: Dict[str, Any]
    genome: Dict[str, Any]
    signature: str
    search_origin: str
    search_generation: int


@dataclass
class EvaluatedCandidate:
    candidate: CandidateSpec
    record: Dict[str, Any]


@dataclass
class SweepState:
    args: argparse.Namespace
    base_cfg: Dict[str, Any]
    candles: pd.DataFrame
    evaluation_mode: str
    objective: str
    search_mode: str
    progress_every: int
    dataset_cache: Dict[Tuple[int, float], Tuple[pd.DataFrame, List[str], Dict]] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    sweep_start: float = field(default_factory=time.perf_counter)
    total_feature_sets: int = 0
    total_experiments: int = 0
    built_feature_sets: int = 0
    completed_experiments: int = 0


def parse_list(raw: str, cast) -> List:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_optional_grid(raw: str, cast, default_value):
    if raw and raw.strip():
        return parse_list(raw, cast)
    return [default_value]


def parse_hidden_size_token(token: str) -> Tuple[int, ...]:
    cleaned = str(token).strip()
    if not cleaned:
        raise ValueError("Token de hidden sizes vazio")

    separators = ["-", "x", ":", ";", "/"]
    normalized = cleaned
    for separator in separators:
        normalized = normalized.replace(separator, ",")

    values = [int(part.strip()) for part in normalized.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Token invalido de hidden sizes: {token}")
    return tuple(values)


def format_hidden_sizes(values) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    return "-".join(str(int(value)) for value in values)


def parse_hidden_size_grid(raw: str, default_value) -> List[Tuple[int, ...]]:
    if raw and raw.strip():
        return [parse_hidden_size_token(token) for token in raw.split(",") if token.strip()]
    return [parse_hidden_size_token(format_hidden_sizes(default_value))]


def to_tuple(values: List[Any]) -> Tuple[Any, ...]:
    return tuple(values)


def unique_values(values: List[Any]) -> List[Any]:
    return list(dict.fromkeys(values))


def value_sort_key(value: Any):
    if isinstance(value, tuple):
        return tuple(value)
    return value


def parse_active_window_values(trade_windows_raw: str, step_windows_raw: str) -> List[int]:
    trade_windows = parse_list(trade_windows_raw, int)
    step_windows = parse_list(step_windows_raw, int)
    return sorted(set(trade_windows).intersection(step_windows))


def record_model_hyperparameters(record: Dict, extra_fields: Dict | None = None) -> Dict:
    if extra_fields:
        record.update(extra_fields)
    return record


def metric_sort_columns(metric: str) -> List[str]:
    if metric == "score":
        return ["eligible_sort", "score", "mean_balanced_accuracy_trade", "mean_accuracy_trade"]
    if metric == "mean_balanced_accuracy_trade":
        return ["eligible_sort", "mean_balanced_accuracy_trade", "score", "mean_accuracy_trade"]
    if metric == "mean_accuracy_trade":
        return ["eligible_sort", "mean_accuracy_trade", "score", "mean_balanced_accuracy_trade"]

    return ["eligible_sort", metric, "score", "mean_balanced_accuracy_trade", "mean_accuracy_trade"]


def rank_report(report: pd.DataFrame, metric: str) -> pd.DataFrame:
    if report.empty:
        return report.copy()

    ranked = report.copy()
    if "eligible" in ranked.columns:
        ranked["eligible_sort"] = ranked["eligible"].fillna(False).astype(int)
    else:
        ranked["eligible_sort"] = 0

    ranked = ranked.sort_values(
        metric_sort_columns(metric),
        ascending=[False, False, False, False] + ([False] if len(metric_sort_columns(metric)) == 5 else []),
        na_position="last",
    )
    return ranked.drop(columns=["eligible_sort"], errors="ignore")


def export_rankings(report: pd.DataFrame, output: Path, metrics: List[str], top_k: int) -> None:
    valid_report = report.loc[report["error"].isna()].copy() if "error" in report.columns else report.copy()
    summary_rows = []

    for metric in metrics:
        if metric not in valid_report.columns:
            continue

        ranked = rank_report(valid_report, metric)
        metric_output = output.with_name(f"{output.stem}.top_{metric}{output.suffix}")
        ranked.head(top_k).to_csv(metric_output, index=False)

        if not ranked.empty:
            top_row = ranked.iloc[0].to_dict()
            top_row["ranking_metric"] = metric
            summary_rows.append(top_row)

    if summary_rows:
        summary_output = output.with_name(f"{output.stem}.best_summary{output.suffix}")
        pd.DataFrame(summary_rows).to_csv(summary_output, index=False)


def build_parameter_signature(record: Dict) -> str:
    signature = (
        f"{record.get('model_type', 'na')}"
        f"|h={record.get('horizon_bars', 'na')}"
        f"|thr={record.get('trend_threshold_atr', 'na')}"
        f"|train={record.get('train_window', 'na')}"
        f"|valid={record.get('validation_window', 'na')}"
        f"|trade={record.get('trade_window', 'na')}"
        f"|step={record.get('step_window', 'na')}"
        f"|maxfolds={record.get('max_folds', 'na')}"
        f"|minclass={record.get('min_class_count_per_train', 'na')}"
    )
    extras = [
        ("C", "C"),
        ("max_iter", "maxiter"),
        ("n_estimators", "trees"),
        ("max_depth", "depth"),
        ("min_samples_leaf", "leaf"),
        ("lookback_bars", "lookback"),
        ("epochs", "epochs"),
        ("batch_size", "batch"),
        ("learning_rate", "lr"),
        ("weight_decay", "wd"),
        ("dropout", "drop"),
        ("mlp_hidden_sizes", "mlp"),
        ("lstm_hidden_size", "lstmh"),
        ("lstm_num_layers", "lstml"),
        ("cnn_channels", "cnnc"),
        ("cnn_kernel_size", "cnnk"),
        ("transformer_d_model", "trdm"),
        ("transformer_nhead", "trhead"),
        ("transformer_num_layers", "trlayers"),
        ("transformer_dim_feedforward", "trff"),
    ]
    for field_name, token in extras:
        value = record.get(field_name)
        if value is None or value == "":
            continue
        signature += f"|{token}={value}"
    return signature


def build_candidate_signature(candidate: CandidateSpec, max_folds: int) -> str:
    record = {
        "model_type": candidate.model_type,
        "horizon_bars": candidate.horizon,
        "trend_threshold_atr": candidate.threshold,
        "train_window": candidate.train_window,
        "validation_window": candidate.validation_window,
        "trade_window": candidate.trade_window,
        "step_window": candidate.step_window,
        "max_folds": max_folds,
        "min_class_count_per_train": candidate.min_class_count,
        **candidate.record_fields,
    }
    return build_parameter_signature(record)


def should_log_progress(index: int, total: int, every: int) -> bool:
    every = max(1, int(every))
    return index == 1 or index == total or index % every == 0


def maybe_log_best(record: Dict) -> str:
    if record.get("error"):
        return f"status=error last_error={record['error']}"

    score = record.get("score")
    mean_bal_acc = record.get("mean_balanced_accuracy_trade")
    mean_acc = record.get("mean_accuracy_trade")
    return (
        f"status=ok "
        f"score={score if score is not None else 'nan'} "
        f"mean_bal_acc={mean_bal_acc if mean_bal_acc is not None else 'nan'} "
        f"mean_acc={mean_acc if mean_acc is not None else 'nan'}"
    )


def build_experiment_record(
    evaluation_mode: str,
    objective: str,
    model_type: str,
    horizon: int,
    threshold: float,
    train_window,
    validation_window,
    trade_window,
    step_window,
    min_class_count,
    max_folds,
    purge_bars,
    feature_rows,
    extra_fields: Dict | None = None,
) -> Dict:
    record = {
        "evaluation_mode": evaluation_mode,
        "objective": objective,
        "model_type": model_type,
        "horizon_bars": horizon,
        "trend_threshold_atr": threshold,
        "train_window": train_window,
        "validation_window": validation_window,
        "trade_window": trade_window,
        "step_window": step_window,
        "min_class_count_per_train": min_class_count,
        "max_folds": max_folds,
        "purge_bars": purge_bars,
        "feature_rows": feature_rows,
    }
    return record_model_hyperparameters(record, extra_fields)


def build_model_search_variants(base_cfg: Dict, model_type: str, args: argparse.Namespace) -> List[Dict]:
    train_cfg = base_cfg["train"]
    normalized_model_type = normalize_model_type(model_type)
    variants: List[Dict] = []

    if normalized_model_type == "logistic_regression":
        c_values = parse_optional_grid(args.logistic_cs, float, float(train_cfg.get("C", 1.0)))
        max_iter_values = parse_optional_grid(args.logistic_max_iters, int, int(train_cfg.get("max_iter", 1500)))
        for c_value, max_iter in product(c_values, max_iter_values):
            variants.append(
                {
                    "cfg_updates": {
                        "C": float(c_value),
                        "max_iter": int(max_iter),
                    },
                    "record_fields": {
                        "C": float(c_value),
                        "max_iter": int(max_iter),
                    },
                }
            )
        return variants

    if normalized_model_type == "random_forest":
        estimator_values = parse_optional_grid(args.rf_n_estimators, int, int(train_cfg.get("n_estimators", 300)))
        depth_values = parse_optional_grid(args.rf_max_depths, int, int(train_cfg.get("max_depth", 8)))
        leaf_values = parse_optional_grid(args.rf_min_samples_leaf, int, int(train_cfg.get("min_samples_leaf", 10)))
        for estimator_count, max_depth, min_leaf in product(estimator_values, depth_values, leaf_values):
            variants.append(
                {
                    "cfg_updates": {
                        "n_estimators": int(estimator_count),
                        "max_depth": int(max_depth),
                        "min_samples_leaf": int(min_leaf),
                    },
                    "record_fields": {
                        "n_estimators": int(estimator_count),
                        "max_depth": int(max_depth),
                        "min_samples_leaf": int(min_leaf),
                    },
                }
            )
        return variants

    epochs_values = parse_optional_grid(args.epochs, int, int(train_cfg.get("epochs", 25)))
    batch_size_values = parse_optional_grid(args.batch_sizes, int, int(train_cfg.get("batch_size", 256)))
    learning_rate_values = parse_optional_grid(args.learning_rates, float, float(train_cfg.get("learning_rate", 1e-3)))
    weight_decay_values = parse_optional_grid(args.weight_decays, float, float(train_cfg.get("weight_decay", 1e-5)))
    dropout_values = parse_optional_grid(args.dropouts, float, float(train_cfg.get("dropout", 0.10)))

    if normalized_model_type == "deep_mlp":
        hidden_size_values = parse_hidden_size_grid(args.mlp_hidden_sizes, train_cfg.get("mlp_hidden_sizes", [128, 64]))
        for epochs, batch_size, learning_rate, weight_decay, dropout, hidden_sizes in product(
            epochs_values,
            batch_size_values,
            learning_rate_values,
            weight_decay_values,
            dropout_values,
            hidden_size_values,
        ):
            formatted_hidden_sizes = format_hidden_sizes(hidden_sizes)
            variants.append(
                {
                    "cfg_updates": {
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "mlp_hidden_sizes": list(hidden_sizes),
                    },
                    "record_fields": {
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "mlp_hidden_sizes": formatted_hidden_sizes,
                    },
                }
            )
        return variants

    lookback_values = parse_optional_grid(args.lookback_bars, int, int(train_cfg.get("lookback_bars", 32)))

    if normalized_model_type == "lstm":
        hidden_size_values = parse_optional_grid(args.lstm_hidden_sizes, int, int(train_cfg.get("lstm_hidden_size", 64)))
        num_layer_values = parse_optional_grid(args.lstm_num_layers, int, int(train_cfg.get("lstm_num_layers", 2)))
        for lookback, epochs, batch_size, learning_rate, weight_decay, dropout, hidden_size, num_layers in product(
            lookback_values,
            epochs_values,
            batch_size_values,
            learning_rate_values,
            weight_decay_values,
            dropout_values,
            hidden_size_values,
            num_layer_values,
        ):
            variants.append(
                {
                    "cfg_updates": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "lstm_hidden_size": int(hidden_size),
                        "lstm_num_layers": int(num_layers),
                    },
                    "record_fields": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "lstm_hidden_size": int(hidden_size),
                        "lstm_num_layers": int(num_layers),
                    },
                }
            )
        return variants

    if normalized_model_type == "cnn":
        channel_values = parse_optional_grid(args.cnn_channels, int, int(train_cfg.get("cnn_channels", 64)))
        kernel_values = parse_optional_grid(args.cnn_kernel_sizes, int, int(train_cfg.get("cnn_kernel_size", 3)))
        for lookback, epochs, batch_size, learning_rate, weight_decay, dropout, channels, kernel_size in product(
            lookback_values,
            epochs_values,
            batch_size_values,
            learning_rate_values,
            weight_decay_values,
            dropout_values,
            channel_values,
            kernel_values,
        ):
            variants.append(
                {
                    "cfg_updates": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "cnn_channels": int(channels),
                        "cnn_kernel_size": int(kernel_size),
                    },
                    "record_fields": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "cnn_channels": int(channels),
                        "cnn_kernel_size": int(kernel_size),
                    },
                }
            )
        return variants

    if normalized_model_type == "transformer":
        d_model_values = parse_optional_grid(args.transformer_d_models, int, int(train_cfg.get("transformer_d_model", 64)))
        nhead_values = parse_optional_grid(args.transformer_nheads, int, int(train_cfg.get("transformer_nhead", 4)))
        layer_values = parse_optional_grid(args.transformer_num_layers, int, int(train_cfg.get("transformer_num_layers", 2)))
        ff_values = parse_optional_grid(
            args.transformer_dim_feedforwards,
            int,
            int(train_cfg.get("transformer_dim_feedforward", 128)),
        )
        for lookback, epochs, batch_size, learning_rate, weight_decay, dropout, d_model, nhead, num_layers, dim_ff in product(
            lookback_values,
            epochs_values,
            batch_size_values,
            learning_rate_values,
            weight_decay_values,
            dropout_values,
            d_model_values,
            nhead_values,
            layer_values,
            ff_values,
        ):
            if int(d_model) % int(nhead) != 0:
                continue
            variants.append(
                {
                    "cfg_updates": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "transformer_d_model": int(d_model),
                        "transformer_nhead": int(nhead),
                        "transformer_num_layers": int(num_layers),
                        "transformer_dim_feedforward": int(dim_ff),
                    },
                    "record_fields": {
                        "lookback_bars": int(lookback),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "dropout": float(dropout),
                        "transformer_d_model": int(d_model),
                        "transformer_nhead": int(nhead),
                        "transformer_num_layers": int(num_layers),
                        "transformer_dim_feedforward": int(dim_ff),
                    },
                }
            )
        return variants

    raise ValueError(f"Modelo nao suportado no sweep: {normalized_model_type}")


def build_family_search_space(
    base_cfg: Dict,
    model_type: str,
    args: argparse.Namespace,
    evaluation_mode: str,
) -> SearchFamilySpace:
    train_cfg = base_cfg["train"]
    normalized_model_type = normalize_model_type(model_type)
    genes: Dict[str, Tuple[Any, ...]] = {}
    gene_order: List[str] = []

    def add_gene(gene_name: str, values: List[Any]) -> None:
        unique = unique_values(values)
        genes[gene_name] = to_tuple(unique)
        gene_order.append(gene_name)

    add_gene("horizon_bars", parse_list(args.horizons, int))
    add_gene("trend_threshold_atr", parse_list(args.thresholds, float))

    if evaluation_mode != "holdout":
        add_gene("train_window", parse_list(args.train_windows, int))
        add_gene("validation_window", parse_list(args.validation_windows, int))
        add_gene("active_window", parse_active_window_values(args.trade_windows, args.step_windows))
        min_class_counts = (
            parse_list(args.min_class_counts_per_train, int)
            if args.min_class_counts_per_train.strip()
            else [int(args.min_class_count_per_train)]
        )
        add_gene("min_class_count_per_train", min_class_counts)

    if normalized_model_type == "logistic_regression":
        add_gene("C", parse_optional_grid(args.logistic_cs, float, float(train_cfg.get("C", 1.0))))
        add_gene("max_iter", parse_optional_grid(args.logistic_max_iters, int, int(train_cfg.get("max_iter", 1500))))
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    if normalized_model_type == "random_forest":
        add_gene("n_estimators", parse_optional_grid(args.rf_n_estimators, int, int(train_cfg.get("n_estimators", 300))))
        add_gene("max_depth", parse_optional_grid(args.rf_max_depths, int, int(train_cfg.get("max_depth", 8))))
        add_gene(
            "min_samples_leaf",
            parse_optional_grid(args.rf_min_samples_leaf, int, int(train_cfg.get("min_samples_leaf", 10))),
        )
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    epochs_values = parse_optional_grid(args.epochs, int, int(train_cfg.get("epochs", 25)))
    batch_size_values = parse_optional_grid(args.batch_sizes, int, int(train_cfg.get("batch_size", 256)))
    learning_rate_values = parse_optional_grid(args.learning_rates, float, float(train_cfg.get("learning_rate", 1e-3)))
    weight_decay_values = parse_optional_grid(args.weight_decays, float, float(train_cfg.get("weight_decay", 1e-5)))
    dropout_values = parse_optional_grid(args.dropouts, float, float(train_cfg.get("dropout", 0.10)))

    if normalized_model_type == "deep_mlp":
        add_gene("epochs", epochs_values)
        add_gene("batch_size", batch_size_values)
        add_gene("learning_rate", learning_rate_values)
        add_gene("weight_decay", weight_decay_values)
        add_gene("dropout", dropout_values)
        add_gene(
            "mlp_hidden_sizes",
            parse_hidden_size_grid(args.mlp_hidden_sizes, train_cfg.get("mlp_hidden_sizes", [128, 64])),
        )
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    lookback_values = parse_optional_grid(args.lookback_bars, int, int(train_cfg.get("lookback_bars", 32)))

    if normalized_model_type == "lstm":
        add_gene("lookback_bars", lookback_values)
        add_gene("epochs", epochs_values)
        add_gene("batch_size", batch_size_values)
        add_gene("learning_rate", learning_rate_values)
        add_gene("weight_decay", weight_decay_values)
        add_gene("dropout", dropout_values)
        add_gene(
            "lstm_hidden_size",
            parse_optional_grid(args.lstm_hidden_sizes, int, int(train_cfg.get("lstm_hidden_size", 64))),
        )
        add_gene(
            "lstm_num_layers",
            parse_optional_grid(args.lstm_num_layers, int, int(train_cfg.get("lstm_num_layers", 2))),
        )
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    if normalized_model_type == "cnn":
        add_gene("lookback_bars", lookback_values)
        add_gene("epochs", epochs_values)
        add_gene("batch_size", batch_size_values)
        add_gene("learning_rate", learning_rate_values)
        add_gene("weight_decay", weight_decay_values)
        add_gene("dropout", dropout_values)
        add_gene("cnn_channels", parse_optional_grid(args.cnn_channels, int, int(train_cfg.get("cnn_channels", 64))))
        add_gene(
            "cnn_kernel_size",
            parse_optional_grid(args.cnn_kernel_sizes, int, int(train_cfg.get("cnn_kernel_size", 3))),
        )
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    if normalized_model_type == "transformer":
        d_model_values = parse_optional_grid(args.transformer_d_models, int, int(train_cfg.get("transformer_d_model", 64)))
        nhead_values = parse_optional_grid(args.transformer_nheads, int, int(train_cfg.get("transformer_nhead", 4)))
        valid_head_pairs = [
            (int(d_model), int(nhead))
            for d_model, nhead in product(d_model_values, nhead_values)
            if int(d_model) % int(nhead) == 0
        ]
        add_gene("lookback_bars", lookback_values)
        add_gene("epochs", epochs_values)
        add_gene("batch_size", batch_size_values)
        add_gene("learning_rate", learning_rate_values)
        add_gene("weight_decay", weight_decay_values)
        add_gene("dropout", dropout_values)
        add_gene("transformer_head_pair", valid_head_pairs)
        add_gene(
            "transformer_num_layers",
            parse_optional_grid(args.transformer_num_layers, int, int(train_cfg.get("transformer_num_layers", 2))),
        )
        add_gene(
            "transformer_dim_feedforward",
            parse_optional_grid(
                args.transformer_dim_feedforwards,
                int,
                int(train_cfg.get("transformer_dim_feedforward", 128)),
            ),
        )
        return SearchFamilySpace(normalized_model_type, evaluation_mode, tuple(gene_order), genes)

    raise ValueError(f"Modelo nao suportado no sweep: {normalized_model_type}")


def build_family_search_spaces(
    base_cfg: Dict,
    model_values: List[str],
    args: argparse.Namespace,
    evaluation_mode: str,
) -> Dict[str, SearchFamilySpace]:
    return {
        model_type: build_family_search_space(base_cfg, model_type, args, evaluation_mode)
        for model_type in model_values
    }


def candidate_from_genome(
    space: SearchFamilySpace,
    genome: Dict[str, Any],
    max_folds: int,
    origin: str,
    generation: int,
) -> CandidateSpec:
    model_type = space.model_type
    horizon = int(genome["horizon_bars"])
    threshold = float(genome["trend_threshold_atr"])

    train_window = None
    validation_window = None
    trade_window = None
    step_window = None
    min_class_count = None
    if space.evaluation_mode != "holdout":
        train_window = int(genome["train_window"])
        validation_window = int(genome["validation_window"])
        trade_window = int(genome["active_window"])
        step_window = int(genome["active_window"])
        min_class_count = int(genome["min_class_count_per_train"])

    cfg_updates: Dict[str, Any] = {}
    record_fields: Dict[str, Any] = {}

    if model_type == "logistic_regression":
        cfg_updates = {
            "C": float(genome["C"]),
            "max_iter": int(genome["max_iter"]),
        }
        record_fields = {
            "C": float(genome["C"]),
            "max_iter": int(genome["max_iter"]),
        }
    elif model_type == "random_forest":
        cfg_updates = {
            "n_estimators": int(genome["n_estimators"]),
            "max_depth": int(genome["max_depth"]),
            "min_samples_leaf": int(genome["min_samples_leaf"]),
        }
        record_fields = {
            "n_estimators": int(genome["n_estimators"]),
            "max_depth": int(genome["max_depth"]),
            "min_samples_leaf": int(genome["min_samples_leaf"]),
        }
    elif model_type == "deep_mlp":
        hidden_sizes = tuple(int(value) for value in genome["mlp_hidden_sizes"])
        cfg_updates = {
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "mlp_hidden_sizes": list(hidden_sizes),
        }
        record_fields = {
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "mlp_hidden_sizes": format_hidden_sizes(hidden_sizes),
        }
    elif model_type == "lstm":
        cfg_updates = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "lstm_hidden_size": int(genome["lstm_hidden_size"]),
            "lstm_num_layers": int(genome["lstm_num_layers"]),
        }
        record_fields = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "lstm_hidden_size": int(genome["lstm_hidden_size"]),
            "lstm_num_layers": int(genome["lstm_num_layers"]),
        }
    elif model_type == "cnn":
        cfg_updates = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "cnn_channels": int(genome["cnn_channels"]),
            "cnn_kernel_size": int(genome["cnn_kernel_size"]),
        }
        record_fields = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "cnn_channels": int(genome["cnn_channels"]),
            "cnn_kernel_size": int(genome["cnn_kernel_size"]),
        }
    elif model_type == "transformer":
        d_model, nhead = genome["transformer_head_pair"]
        cfg_updates = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "transformer_d_model": int(d_model),
            "transformer_nhead": int(nhead),
            "transformer_num_layers": int(genome["transformer_num_layers"]),
            "transformer_dim_feedforward": int(genome["transformer_dim_feedforward"]),
        }
        record_fields = {
            "lookback_bars": int(genome["lookback_bars"]),
            "epochs": int(genome["epochs"]),
            "batch_size": int(genome["batch_size"]),
            "learning_rate": float(genome["learning_rate"]),
            "weight_decay": float(genome["weight_decay"]),
            "dropout": float(genome["dropout"]),
            "transformer_d_model": int(d_model),
            "transformer_nhead": int(nhead),
            "transformer_num_layers": int(genome["transformer_num_layers"]),
            "transformer_dim_feedforward": int(genome["transformer_dim_feedforward"]),
        }
    else:
        raise ValueError(f"Modelo nao suportado na decodificacao do sweep: {model_type}")

    draft_candidate = CandidateSpec(
        model_type=model_type,
        horizon=horizon,
        threshold=threshold,
        train_window=train_window,
        validation_window=validation_window,
        trade_window=trade_window,
        step_window=step_window,
        min_class_count=min_class_count,
        cfg_updates=cfg_updates,
        record_fields=record_fields,
        genome=dict(genome),
        signature="",
        search_origin=origin,
        search_generation=generation,
    )
    signature = build_candidate_signature(draft_candidate, max_folds)
    return CandidateSpec(
        model_type=draft_candidate.model_type,
        horizon=draft_candidate.horizon,
        threshold=draft_candidate.threshold,
        train_window=draft_candidate.train_window,
        validation_window=draft_candidate.validation_window,
        trade_window=draft_candidate.trade_window,
        step_window=draft_candidate.step_window,
        min_class_count=draft_candidate.min_class_count,
        cfg_updates=draft_candidate.cfg_updates,
        record_fields=draft_candidate.record_fields,
        genome=draft_candidate.genome,
        signature=signature,
        search_origin=origin,
        search_generation=generation,
    )


def iter_family_candidates(
    space: SearchFamilySpace,
    max_folds: int,
    origin: str,
    generation: int,
):
    gene_values = [space.genes[gene_name] for gene_name in space.gene_order]
    if not gene_values or any(len(values) == 0 for values in gene_values):
        return
    for values in product(*gene_values):
        genome = dict(zip(space.gene_order, values))
        yield candidate_from_genome(space, genome, max_folds, origin, generation)


def random_genome(space: SearchFamilySpace, rng: random.Random) -> Dict[str, Any]:
    return {
        gene_name: rng.choice(space.genes[gene_name])
        for gene_name in space.gene_order
    }


def sample_random_candidates(
    space: SearchFamilySpace,
    count: int,
    max_folds: int,
    rng: random.Random,
    seen_signatures: set[str],
    origin: str,
    generation: int,
) -> List[CandidateSpec]:
    if count <= 0 or space.total_candidates == 0:
        return []

    target = min(count, space.total_candidates)
    if target <= 0:
        return []

    selected: List[CandidateSpec] = []
    local_signatures: set[str] = set()
    max_attempts = max(100, target * 50)
    attempts = 0

    while len(selected) < target and attempts < max_attempts:
        attempts += 1
        candidate = candidate_from_genome(
            space,
            random_genome(space, rng),
            max_folds,
            origin=origin,
            generation=generation,
        )
        if candidate.signature in seen_signatures or candidate.signature in local_signatures:
            continue
        local_signatures.add(candidate.signature)
        selected.append(candidate)

    if len(selected) >= target:
        return selected

    fallback_candidates = [
        candidate
        for candidate in iter_family_candidates(space, max_folds, origin=origin, generation=generation) or []
        if candidate.signature not in seen_signatures and candidate.signature not in local_signatures
    ]
    rng.shuffle(fallback_candidates)
    selected.extend(fallback_candidates[: target - len(selected)])
    return selected


def allocate_family_budgets(spaces: Dict[str, SearchFamilySpace], total_budget: int) -> Dict[str, int]:
    families = [
        space
        for space in spaces.values()
        if space.total_candidates > 0
    ]
    budgets = {space.model_type: 0 for space in spaces.values()}
    if total_budget <= 0 or not families:
        return budgets

    capped_budget = min(total_budget, sum(space.total_candidates for space in families))
    remaining_capacity = {space.model_type: space.total_candidates for space in families}

    if capped_budget < len(families):
        for space in sorted(families, key=lambda item: item.total_candidates, reverse=True)[:capped_budget]:
            budgets[space.model_type] = 1
        return budgets

    for space in families:
        budgets[space.model_type] = 1
        remaining_capacity[space.model_type] -= 1

    remaining_budget = capped_budget - len(families)
    if remaining_budget <= 0:
        return budgets

    equal_pool = remaining_budget // 2
    proportional_pool = remaining_budget - equal_pool

    if equal_pool > 0:
        equal_each = equal_pool // len(families)
        for space in families:
            extra = min(equal_each, remaining_capacity[space.model_type])
            budgets[space.model_type] += extra
            remaining_capacity[space.model_type] -= extra
            equal_pool -= extra

        for space in sorted(
            families,
            key=lambda item: (
                -remaining_capacity[item.model_type],
                budgets[item.model_type],
                item.model_type,
            ),
        ):
            if equal_pool <= 0:
                break
            if remaining_capacity[space.model_type] <= 0:
                continue
            budgets[space.model_type] += 1
            remaining_capacity[space.model_type] -= 1
            equal_pool -= 1

    if proportional_pool > 0:
        total_capacity = sum(remaining_capacity.values())
        remainders: Dict[str, float] = {space.model_type: 0.0 for space in families}
        if total_capacity > 0:
            provisional: Dict[str, int] = {}
            for space in families:
                family = space.model_type
                share = proportional_pool * (remaining_capacity[family] / total_capacity)
                floor_share = min(remaining_capacity[family], int(math.floor(share)))
                provisional[family] = floor_share
                remainders[family] = share - floor_share

            used = 0
            for family, extra in provisional.items():
                budgets[family] += extra
                remaining_capacity[family] -= extra
                used += extra
            proportional_pool -= used

        for space in sorted(
            families,
            key=lambda item: (
                -remainders.get(item.model_type, 0.0),
                budgets[item.model_type],
                -remaining_capacity[item.model_type],
                item.model_type,
            ),
        ):
            if proportional_pool <= 0:
                break
            family = space.model_type
            if remaining_capacity[family] <= 0:
                continue
            budgets[family] += 1
            remaining_capacity[family] -= 1
            proportional_pool -= 1

    return budgets


def numeric_or_default(value: Any, default: float = float("-inf")) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def candidate_rank_key(record: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        1.0 if bool(record.get("eligible")) else 0.0,
        numeric_or_default(record.get("score")),
        numeric_or_default(record.get("score_raw")),
        numeric_or_default(record.get("mean_balanced_accuracy_trade")),
        numeric_or_default(record.get("mean_accuracy_trade")),
    )


def tournament_select(
    evaluated: List[EvaluatedCandidate],
    rng: random.Random,
    tournament_size: int,
) -> EvaluatedCandidate:
    contenders = rng.sample(evaluated, min(len(evaluated), max(2, tournament_size)))
    return max(contenders, key=lambda item: candidate_rank_key(item.record))


def crossover_genomes(
    space: SearchFamilySpace,
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    child: Dict[str, Any] = {}
    for gene_name in space.gene_order:
        child[gene_name] = parent_a[gene_name] if rng.random() < 0.5 else parent_b[gene_name]
    return child


def mutate_genome(
    space: SearchFamilySpace,
    genome: Dict[str, Any],
    mutation_rate: float,
    rng: random.Random,
) -> Dict[str, Any]:
    mutated = dict(genome)
    mutable_genes = [gene_name for gene_name in space.gene_order if len(space.genes[gene_name]) > 1]
    mutated_any = False
    for gene_name in mutable_genes:
        if rng.random() >= mutation_rate:
            continue
        alternatives = [value for value in space.genes[gene_name] if value != mutated[gene_name]]
        if not alternatives:
            continue
        mutated[gene_name] = rng.choice(alternatives)
        mutated_any = True

    if not mutated_any and mutable_genes and mutation_rate > 0.0:
        gene_name = rng.choice(mutable_genes)
        alternatives = [value for value in space.genes[gene_name] if value != mutated[gene_name]]
        if alternatives:
            mutated[gene_name] = rng.choice(alternatives)

    return mutated
def validate_single_model_policy(policy: WalkForwardPolicy) -> str | None:
    if policy.step_window > policy.trade_window:
        return (
            "Politica invalida para runtime com modelo unico por candle: "
            f"step_window ({policy.step_window}) maior que trade_window ({policy.trade_window}) cria buraco entre modelos."
        )
    if policy.step_window < policy.trade_window:
        return (
            "Politica invalida para runtime com modelo unico por candle: "
            f"step_window ({policy.step_window}) menor que trade_window ({policy.trade_window}) cria sobreposicao entre modelos."
        )
    return None


def fold_number_from_id(fold_id: str) -> int:
    raw_number = str(fold_id).strip().split("_")[-1]
    if not raw_number.isdigit():
        raise ValueError(f"fold_id invalido para cobertura temporal: {fold_id}")
    fold_number = int(raw_number) - 1
    if fold_number < 0:
        raise ValueError(f"fold_id invalido para cobertura temporal: {fold_id}")
    return fold_number


def merge_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not segments:
        return []

    merged: List[Tuple[int, int]] = []
    for start, end in sorted(segments):
        if not merged:
            merged.append((start, end))
            continue

        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def summarize_temporal_coverage(summary_df: pd.DataFrame, policy: WalkForwardPolicy) -> Dict:
    empty_summary = {
        "temporal_valid": False,
        "temporal_error": "Walk-forward sem folds planejados para avaliar cobertura temporal.",
        "coverage_start_row": None,
        "coverage_end_row": None,
        "coverage_target_bars": 0,
        "covered_bars": 0,
        "uncovered_bars": 0,
        "model_coverage_ratio": 0.0,
        "max_gap_bars": 0,
        "overlap_bars": 0,
        "first_gap_start_row": None,
        "first_gap_end_row": None,
        "ok_folds_for_coverage": 0,
        "skipped_folds_for_coverage": 0,
    }

    if summary_df.empty:
        return empty_summary

    trade_offset = policy.train_window + policy.validation_window + (2 * policy.purge_bars)
    planned_segments: List[Tuple[int, int]] = []
    ok_segments: List[Tuple[int, int]] = []

    summary_records = summary_df.to_dict(orient="records")
    for record in summary_records:
        fold_id = str(record.get("fold_id", "")).strip()
        if not fold_id:
            continue

        fold_number = fold_number_from_id(fold_id)
        trade_start = trade_offset + (fold_number * policy.step_window)
        trade_end = trade_start + policy.trade_window
        planned_segments.append((trade_start, trade_end))

        if str(record.get("status", "")).lower() == "ok":
            ok_segments.append((trade_start, trade_end))

    if not planned_segments:
        return empty_summary

    coverage_start = min(start for start, _end in planned_segments)
    coverage_end = max(end for _start, end in planned_segments)
    coverage_target_bars = max(0, coverage_end - coverage_start)

    merged_ok_segments = merge_segments(ok_segments)
    covered_bars = sum(end - start for start, end in merged_ok_segments)
    raw_ok_bars = sum(end - start for start, end in ok_segments)
    overlap_bars = max(0, raw_ok_bars - covered_bars)

    uncovered_segments: List[Tuple[int, int]] = []
    cursor = coverage_start
    for start, end in merged_ok_segments:
        if start > cursor:
            uncovered_segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < coverage_end:
        uncovered_segments.append((cursor, coverage_end))

    uncovered_bars = sum(end - start for start, end in uncovered_segments)
    max_gap_bars = max((end - start for start, end in uncovered_segments), default=0)
    first_gap_start = uncovered_segments[0][0] if uncovered_segments else None
    first_gap_end = uncovered_segments[0][1] if uncovered_segments else None
    model_coverage_ratio = (
        float(covered_bars / coverage_target_bars)
        if coverage_target_bars > 0
        else 1.0
    )

    temporal_valid = bool(ok_segments) and uncovered_bars == 0 and overlap_bars == 0
    temporal_error = None
    if not ok_segments:
        temporal_error = "Nenhum fold valido produz modelo para cobrir a janela avaliada."
    elif uncovered_bars > 0:
        temporal_error = (
            "Cobertura temporal invalida: existem candles sem modelo entre os folds "
            f"(max_gap_bars={max_gap_bars}, uncovered_bars={uncovered_bars})."
        )
    elif overlap_bars > 0:
        temporal_error = (
            "Cobertura temporal invalida: existem candles com mais de um modelo ativo "
            f"(overlap_bars={overlap_bars})."
        )

    return {
        "temporal_valid": temporal_valid,
        "temporal_error": temporal_error,
        "coverage_start_row": int(coverage_start),
        "coverage_end_row": int(coverage_end),
        "coverage_target_bars": int(coverage_target_bars),
        "covered_bars": int(covered_bars),
        "uncovered_bars": int(uncovered_bars),
        "model_coverage_ratio": model_coverage_ratio,
        "max_gap_bars": int(max_gap_bars),
        "overlap_bars": int(overlap_bars),
        "first_gap_start_row": int(first_gap_start) if first_gap_start is not None else None,
        "first_gap_end_row": int(first_gap_end) if first_gap_end is not None else None,
        "ok_folds_for_coverage": int(len(ok_segments)),
        "skipped_folds_for_coverage": int(len(planned_segments) - len(ok_segments)),
    }


def dataset_for_label_combo(candles: pd.DataFrame, base_cfg: Dict, horizon: int, threshold: float) -> Tuple[pd.DataFrame, List[str], Dict]:
    cfg = copy.deepcopy(base_cfg)
    cfg["label"]["horizon_bars"] = horizon
    cfg["label"]["trend_threshold_atr"] = threshold
    dataset, feature_columns = build_feature_frame(candles, cfg)
    return dataset, feature_columns, cfg


def evaluate_holdout(cfg: Dict, candles: pd.DataFrame, dataset: pd.DataFrame, feature_columns: List[str]) -> Dict:
    train_df, test_df = chronological_split(dataset, float(cfg["train"]["test_size"]))
    model = build_model_backend(cfg)
    model.fit(train_df, feature_columns)
    test_history_df = dataset.iloc[:-len(test_df)].tail(model.history_rows_needed)
    metrics, _confusion_df, _predictions_df = evaluate_model(
        model,
        test_df,
        feature_columns,
        history_df=test_history_df,
    )
    diagnostics = dataset_diagnostics(candles, dataset, train_df, test_df)

    return {
        "available_folds": 1,
        "planned_folds": 1,
        "valid_folds": 1,
        "coverage": 1.0,
        "eligible": True,
        "score": float(metrics["balanced_accuracy"]),
        "score_raw": float(metrics["balanced_accuracy"]),
        "mean_accuracy_trade": float(metrics["accuracy"]),
        "mean_balanced_accuracy_trade": float(metrics["balanced_accuracy"]),
        "std_balanced_accuracy_trade": 0.0,
        "mean_accuracy_validation": None,
        "mean_balanced_accuracy_validation": None,
        "trade_rows_total": int(len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_distribution": diagnostics["train_class_distribution"],
        "test_distribution": diagnostics["test_class_distribution"],
        "temporal_valid": True,
        "temporal_error": None,
        "coverage_start_row": None,
        "coverage_end_row": None,
        "coverage_target_bars": int(len(test_df)),
        "covered_bars": int(len(test_df)),
        "uncovered_bars": 0,
        "model_coverage_ratio": 1.0,
        "max_gap_bars": 0,
        "overlap_bars": 0,
        "first_gap_start_row": None,
        "first_gap_end_row": None,
        "ok_folds_for_coverage": 1,
        "skipped_folds_for_coverage": 0,
    }


def evaluate_walkforward_combo(
    cfg: Dict,
    candles: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_columns: List[str],
    policy: WalkForwardPolicy,
) -> Dict:
    payload = evaluate_walkforward(
        candles=candles,
        dataset=dataset,
        feature_columns=feature_columns,
        cfg=cfg,
        policy=policy,
        export_dir=None,
        validate_onnx=False,
    )
    aggregate = payload["aggregate"]
    temporal = summarize_temporal_coverage(payload["summary"], policy)
    eligible = bool(aggregate["eligible"]) and bool(temporal["temporal_valid"])
    raw_score = aggregate["score"]
    return {
        "available_folds": aggregate["available_folds"],
        "planned_folds": aggregate["planned_folds"],
        "valid_folds": aggregate["valid_folds"],
        "coverage": aggregate["coverage"],
        "eligible": eligible,
        "score": raw_score if eligible else None,
        "score_raw": raw_score,
        "mean_accuracy_trade": aggregate["mean_accuracy_trade"],
        "mean_balanced_accuracy_trade": aggregate["mean_balanced_accuracy_trade"],
        "std_balanced_accuracy_trade": aggregate["std_balanced_accuracy_trade"],
        "mean_accuracy_validation": aggregate["mean_accuracy_validation"],
        "mean_balanced_accuracy_validation": aggregate["mean_balanced_accuracy_validation"],
        "trade_rows_total": aggregate["trade_rows_total"],
        **temporal,
    }


def ensure_dataset_cache(
    state: SweepState,
    candidate: CandidateSpec,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    cache_key = (candidate.horizon, candidate.threshold)
    if cache_key in state.dataset_cache:
        return state.dataset_cache[cache_key]

    dataset_start = time.perf_counter()
    state.dataset_cache[cache_key] = dataset_for_label_combo(
        state.candles,
        state.base_cfg,
        candidate.horizon,
        candidate.threshold,
    )
    state.built_feature_sets += 1
    built_dataset, _feature_columns, _cfg = state.dataset_cache[cache_key]
    print(
        "[progress] "
        f"stage=feature_frame "
        f"built={state.built_feature_sets}/{state.total_feature_sets} "
        f"horizon={candidate.horizon} "
        f"threshold={candidate.threshold} "
        f"feature_rows={len(built_dataset)} "
        f"elapsed={format_duration(time.perf_counter() - state.sweep_start)} "
        f"last_stage={format_duration(time.perf_counter() - dataset_start)}",
        flush=True,
    )
    return state.dataset_cache[cache_key]


def evaluate_candidate(
    state: SweepState,
    candidate: CandidateSpec,
) -> EvaluatedCandidate:
    experiment_start = time.perf_counter()
    experiment_index = state.completed_experiments + 1
    if should_log_progress(experiment_index, state.total_experiments, state.progress_every):
        print(
            "[progress] "
            f"stage=experiment_start "
            f"index={experiment_index}/{state.total_experiments} "
            f"search_mode={state.search_mode} "
            f"family={candidate.model_type} "
            f"origin={candidate.search_origin} "
            f"generation={candidate.search_generation} "
            f"horizon={candidate.horizon} "
            f"threshold={candidate.threshold} "
            f"train_window={candidate.train_window} "
            f"trade_window={candidate.trade_window} "
            f"step_window={candidate.step_window} "
            f"min_class_count={candidate.min_class_count}",
            flush=True,
        )

    dataset, feature_columns, cfg = ensure_dataset_cache(state, candidate)
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["train"]["model_type"] = candidate.model_type
    variant_cfg["train"].update(candidate.cfg_updates)
    search_fields = {
        "search_mode": state.search_mode,
        "search_family": candidate.model_type,
        "search_origin": candidate.search_origin,
        "search_generation": candidate.search_generation,
    }

    if state.evaluation_mode == "holdout":
        base_record = build_experiment_record(
            evaluation_mode=state.evaluation_mode,
            objective=state.objective,
            model_type=candidate.model_type,
            horizon=candidate.horizon,
            threshold=candidate.threshold,
            train_window=None,
            validation_window=None,
            trade_window=None,
            step_window=None,
            min_class_count=None,
            max_folds=None,
            purge_bars=None,
            feature_rows=len(dataset),
            extra_fields={**candidate.record_fields, **search_fields},
        )
        try:
            metrics = evaluate_holdout(variant_cfg, state.candles, dataset, feature_columns)
            record = {**base_record, **metrics}
        except Exception as exc:
            record = {**base_record, "error": str(exc)}
    else:
        policy = WalkForwardPolicy(
            train_window=int(candidate.train_window),
            validation_window=int(candidate.validation_window),
            trade_window=int(candidate.trade_window),
            step_window=int(candidate.step_window),
            purge_bars=resolve_purge_bars(variant_cfg, state.args.purge_bars),
            min_folds=int(state.args.min_folds),
            max_folds=int(state.args.max_folds),
            min_class_count_per_train=int(candidate.min_class_count),
            objective=state.objective,
        )
        base_record = build_experiment_record(
            evaluation_mode=state.evaluation_mode,
            objective=state.objective,
            model_type=candidate.model_type,
            horizon=candidate.horizon,
            threshold=candidate.threshold,
            train_window=candidate.train_window,
            validation_window=candidate.validation_window,
            trade_window=candidate.trade_window,
            step_window=candidate.step_window,
            min_class_count=candidate.min_class_count,
            max_folds=policy.max_folds,
            purge_bars=policy.purge_bars,
            feature_rows=len(dataset),
            extra_fields={**candidate.record_fields, **search_fields},
        )

        structural_error = validate_single_model_policy(policy)
        if structural_error:
            record = {
                **base_record,
                "eligible": False,
                "score": None,
                "score_raw": None,
                "coverage": 0.0,
                "temporal_valid": False,
                "temporal_error": structural_error,
                "coverage_start_row": None,
                "coverage_end_row": None,
                "coverage_target_bars": None,
                "covered_bars": None,
                "uncovered_bars": None,
                "model_coverage_ratio": None,
                "max_gap_bars": None,
                "overlap_bars": None,
                "first_gap_start_row": None,
                "first_gap_end_row": None,
                "ok_folds_for_coverage": 0,
                "skipped_folds_for_coverage": 0,
                "error": structural_error,
            }
        else:
            try:
                metrics = evaluate_walkforward_combo(
                    variant_cfg,
                    state.candles,
                    dataset,
                    feature_columns,
                    policy,
                )
                record = {**base_record, **metrics}
                if not metrics["temporal_valid"]:
                    record["error"] = metrics["temporal_error"]
            except Exception as exc:
                record = {**base_record, "error": str(exc)}

    record["parameter_signature"] = candidate.signature
    state.records.append(record)
    state.completed_experiments += 1

    if should_log_progress(state.completed_experiments, state.total_experiments, state.progress_every):
        elapsed = time.perf_counter() - state.sweep_start
        avg = elapsed / state.completed_experiments
        eta = avg * (state.total_experiments - state.completed_experiments)
        print(
            "[progress] "
            f"stage=experiment "
            f"index={state.completed_experiments}/{state.total_experiments} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)} "
            f"last={format_duration(time.perf_counter() - experiment_start)} "
            f"{maybe_log_best(record)} "
            f"params={record['parameter_signature']}",
            flush=True,
        )

    return EvaluatedCandidate(candidate=candidate, record=record)


def resolve_search_budget(
    search_mode: str,
    total_candidates: int,
    args: argparse.Namespace,
) -> int:
    if total_candidates <= 0:
        return 0
    if search_mode == "grid":
        return total_candidates

    budget = int(math.ceil(total_candidates * float(args.sample_ratio)))
    budget = max(1, budget)
    if int(args.max_experiments) > 0:
        budget = min(budget, int(args.max_experiments))
    return min(total_candidates, budget)


def resolve_ga_seed_budget(
    search_mode: str,
    family_budget: int,
    population_size: int,
    random_fraction: float,
) -> int:
    if family_budget <= 0:
        return 0
    if search_mode == "random":
        return family_budget
    if search_mode == "hybrid":
        return min(family_budget, max(population_size, int(math.ceil(family_budget * random_fraction))))
    return min(family_budget, max(population_size, 2))


def run_family_grid_search(
    state: SweepState,
    spaces: Dict[str, SearchFamilySpace],
    model_values: List[str],
) -> None:
    for model_type in model_values:
        space = spaces[model_type]
        if space.total_candidates <= 0:
            continue
        print(
            "[progress] "
            f"stage=family_budget "
            f"family={model_type} "
            f"search_space={space.total_candidates} "
            f"planned={space.total_candidates}",
            flush=True,
        )
        for candidate in iter_family_candidates(space, int(state.args.max_folds), origin="grid", generation=0):
            evaluate_candidate(state, candidate)


def run_family_budgeted_search(
    state: SweepState,
    spaces: Dict[str, SearchFamilySpace],
    budgets: Dict[str, int],
    model_values: List[str],
) -> None:
    rng = random.Random(int(state.args.random_seed))
    global_seen_signatures: set[str] = set()

    for model_type in model_values:
        space = spaces[model_type]
        family_budget = int(budgets.get(model_type, 0))
        if space.total_candidates <= 0 or family_budget <= 0:
            continue

        print(
            "[progress] "
            f"stage=family_budget "
            f"family={model_type} "
            f"search_space={space.total_candidates} "
            f"planned={family_budget}",
            flush=True,
        )

        if state.search_mode == "random":
            random_candidates = sample_random_candidates(
                space,
                family_budget,
                int(state.args.max_folds),
                rng,
                global_seen_signatures,
                origin="random",
                generation=0,
            )
            for candidate in random_candidates:
                global_seen_signatures.add(candidate.signature)
                evaluate_candidate(state, candidate)
            continue

        population_size = min(
            max(2, int(state.args.ga_population_size)),
            family_budget,
            max(2, space.total_candidates),
        )
        seed_budget = resolve_ga_seed_budget(
            state.search_mode,
            family_budget,
            population_size,
            float(state.args.ga_random_frac),
        )
        evaluated_family: List[EvaluatedCandidate] = []

        seed_candidates = sample_random_candidates(
            space,
            seed_budget,
            int(state.args.max_folds),
            rng,
            global_seen_signatures,
            origin="hybrid_random" if state.search_mode == "hybrid" else "ga_seed",
            generation=0,
        )
        for candidate in seed_candidates:
            global_seen_signatures.add(candidate.signature)
            evaluated_family.append(evaluate_candidate(state, candidate))

        remaining_budget = family_budget - len(seed_candidates)
        if remaining_budget <= 0 or not evaluated_family:
            continue

        best_key = candidate_rank_key(max(evaluated_family, key=lambda item: candidate_rank_key(item.record)).record)
        stale_generations = 0
        generation = 1
        max_generations = int(state.args.ga_max_generations)

        while remaining_budget > 0:
            if max_generations > 0 and generation > max_generations:
                break
            if stale_generations >= int(state.args.ga_patience):
                break

            evaluated_family.sort(key=lambda item: candidate_rank_key(item.record), reverse=True)
            parent_pool = evaluated_family[:population_size]
            elite_count = max(1, int(math.ceil(len(parent_pool) * float(state.args.ga_elite_frac))))
            elite_pool = parent_pool[:elite_count]
            children_target = min(population_size, remaining_budget)
            children: List[CandidateSpec] = []
            local_signatures: set[str] = set()
            attempts = 0
            max_attempts = max(100, children_target * 60)

            while len(children) < children_target and attempts < max_attempts:
                attempts += 1
                parent_a = tournament_select(elite_pool, rng, int(state.args.ga_tournament_size))
                parent_b = tournament_select(parent_pool, rng, int(state.args.ga_tournament_size))
                child_genome = crossover_genomes(space, parent_a.candidate.genome, parent_b.candidate.genome, rng)
                child_genome = mutate_genome(space, child_genome, float(state.args.ga_mutation_rate), rng)
                child = candidate_from_genome(
                    space,
                    child_genome,
                    int(state.args.max_folds),
                    origin="ga_child",
                    generation=generation,
                )
                if child.signature in global_seen_signatures or child.signature in local_signatures:
                    continue
                local_signatures.add(child.signature)
                children.append(child)

            if len(children) < children_target:
                random_fill = sample_random_candidates(
                    space,
                    children_target - len(children),
                    int(state.args.max_folds),
                    rng,
                    global_seen_signatures.union(local_signatures),
                    origin="ga_fill",
                    generation=generation,
                )
                for candidate in random_fill:
                    if candidate.signature in local_signatures:
                        continue
                    local_signatures.add(candidate.signature)
                    children.append(candidate)

            if not children:
                break

            print(
                "[progress] "
                f"stage=ga_generation "
                f"family={model_type} "
                f"generation={generation} "
                f"children={len(children)} "
                f"remaining_budget={remaining_budget}",
                flush=True,
            )

            for candidate in children:
                global_seen_signatures.add(candidate.signature)
                evaluated_family.append(evaluate_candidate(state, candidate))
            remaining_budget -= len(children)

            current_best = candidate_rank_key(max(evaluated_family, key=lambda item: candidate_rank_key(item.record)).record)
            if current_best > best_key:
                best_key = current_best
                stale_generations = 0
            else:
                stale_generations += 1
            generation += 1

        if remaining_budget > 0:
            fallback_candidates = sample_random_candidates(
                space,
                remaining_budget,
                int(state.args.max_folds),
                rng,
                global_seen_signatures,
                origin="post_ga_random",
                generation=generation,
            )
            for candidate in fallback_candidates:
                global_seen_signatures.add(candidate.signature)
                evaluate_candidate(state, candidate)

def run_sweep(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    candles = load_candles_csv(Path(args.input))
    evaluation_mode = str(args.evaluation_mode).lower()
    objective = str(args.objective)
    ranking_metrics = parse_list(args.rank_metrics, str)
    progress_every = int(args.progress_every)
    search_mode = str(args.search_mode).lower()
    model_values = [normalize_model_type(model_type) for model_type in parse_list(args.models, str)]
    horizon_values = parse_list(args.horizons, int)
    threshold_values = parse_list(args.thresholds, float)
    spaces = build_family_search_spaces(base_cfg, model_values, args, evaluation_mode)
    total_feature_sets = len(horizon_values) * len(threshold_values)
    total_search_space = sum(space.total_candidates for space in spaces.values())
    planned_experiments = resolve_search_budget(search_mode, total_search_space, args)
    state = SweepState(
        args=args,
        base_cfg=base_cfg,
        candles=candles,
        evaluation_mode=evaluation_mode,
        objective=objective,
        search_mode=search_mode,
        progress_every=progress_every,
        total_feature_sets=total_feature_sets,
        total_experiments=planned_experiments,
    )

    print(
        "[progress] "
        f"stage=init "
        f"mode={evaluation_mode} "
        f"search_mode={search_mode} "
        f"dataset_rows={len(candles)} "
        f"feature_sets={total_feature_sets} "
        f"total_search_space={total_search_space} "
        f"planned_experiments={planned_experiments} "
        f"models={len(model_values)} "
        f"horizons={len(horizon_values)} "
        f"thresholds={len(threshold_values)} "
        f"sample_ratio={args.sample_ratio} "
        f"max_experiments={args.max_experiments} "
        f"progress_every={progress_every}",
        flush=True,
    )
    family_sizes = ", ".join(
        f"{model_type}:{spaces[model_type].total_candidates}"
        for model_type in model_values
    )
    print(f"[progress] stage=family_space sizes={family_sizes}", flush=True)

    if planned_experiments <= 0:
        report = pd.DataFrame()
    else:
        if search_mode == "grid":
            run_family_grid_search(state, spaces, model_values)
        else:
            budgets = allocate_family_budgets(spaces, planned_experiments)
            budget_summary = ", ".join(
                f"{model_type}:{budgets.get(model_type, 0)}"
                for model_type in model_values
            )
            print(f"[progress] stage=family_budget_summary budgets={budget_summary}", flush=True)
            run_family_budgeted_search(state, spaces, budgets, model_values)
        report = pd.DataFrame(state.records)

    if "score" in report.columns:
        report = rank_report(report, "score")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    export_rankings(report, output, ranking_metrics, int(args.top_k))

    print(f"[sweep] rows={len(report)} mode={evaluation_mode} output={output}")
    if not report.empty:
        print(report.head(int(args.top_k)).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Varre parametros de rotulo/modelo/janela para regime de mercado")
    parser.add_argument("--input", required=True, help="CSV com candles OHLCV")
    parser.add_argument("--config", type=Path, help="YAML base de configuracao (opcional)")
    parser.add_argument("--output", default="reports/sweep_report.csv", help="CSV de saida")
    parser.add_argument("--evaluation-mode", choices=["walkforward", "holdout"], default="walkforward")
    parser.add_argument(
        "--search-mode",
        choices=["grid", "random", "ga", "hybrid"],
        default="grid",
        help="Estrategia de busca: grade completa, amostragem aleatoria, genetica ou hibrida por familia",
    )
    parser.add_argument("--sample-ratio", type=float, default=0.10, help="Fracao do espaco valida a avaliar fora do modo grid")
    parser.add_argument("--max-experiments", type=int, default=0, help="Cap absoluto de experimentos para buscas random/ga/hybrid (0 desabilita)")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed da amostragem randomica e da busca genetica")
    parser.add_argument("--ga-population-size", type=int, default=12, help="Tamanho da populacao por familia na busca genetica")
    parser.add_argument("--ga-random-frac", type=float, default=0.70, help="Fracao do budget usada como seed aleatoria no modo hybrid")
    parser.add_argument("--ga-elite-frac", type=float, default=0.20, help="Fracao da populacao de elite usada como pool preferencial de pais")
    parser.add_argument("--ga-mutation-rate", type=float, default=0.15, help="Probabilidade de mutacao por gene")
    parser.add_argument("--ga-patience", type=int, default=3, help="Geracoes sem melhora antes de encerrar o refinamento genetico")
    parser.add_argument("--ga-tournament-size", type=int, default=3, help="Tamanho do torneio para selecao dos pais")
    parser.add_argument("--ga-max-generations", type=int, default=0, help="Limite de geracoes por familia na busca genetica (0 desabilita)")
    parser.add_argument("--models", default="logistic_regression,random_forest", help="Modelos separados por virgula")
    parser.add_argument("--horizons", default="4,8,12,24", help="Horizons separados por virgula")
    parser.add_argument("--thresholds", default="0.8,1.0,1.2,1.5,2.0", help="Thresholds ATR separados por virgula")
    parser.add_argument("--train-windows", default="5000,10000,15000,20000", help="Janelas de treino separadas por virgula")
    parser.add_argument("--validation-windows", default="0,1000,2500", help="Janelas de validacao separadas por virgula")
    parser.add_argument("--trade-windows", default="1000,2500", help="Janelas ativas de trade separadas por virgula")
    parser.add_argument("--step-windows", default="1000,2500", help="Passos entre folds separados por virgula")
    parser.add_argument("--purge-bars", default="auto", help="Barras de purge ou 'auto'")
    parser.add_argument("--logistic-cs", default="", help="Lista de valores de C para logistic_regression")
    parser.add_argument("--logistic-max-iters", default="", help="Lista de max_iter para logistic_regression")
    parser.add_argument("--rf-n-estimators", default="", help="Lista de n_estimators para random_forest")
    parser.add_argument("--rf-max-depths", default="", help="Lista de max_depth para random_forest")
    parser.add_argument("--rf-min-samples-leaf", default="", help="Lista de min_samples_leaf para random_forest")
    parser.add_argument("--lookback-bars", default="", help="Lista de lookback_bars para modelos sequenciais")
    parser.add_argument("--epochs", default="", help="Lista de epochs para modelos torch")
    parser.add_argument("--batch-sizes", default="", help="Lista de batch_size para modelos torch")
    parser.add_argument("--learning-rates", default="", help="Lista de learning_rate para modelos torch")
    parser.add_argument("--weight-decays", default="", help="Lista de weight_decay para modelos torch")
    parser.add_argument("--dropouts", default="", help="Lista de dropout para modelos torch")
    parser.add_argument("--mlp-hidden-sizes", default="", help="Lista de arquiteturas do deep_mlp, ex: 128-64,256-128")
    parser.add_argument("--lstm-hidden-sizes", default="", help="Lista de hidden_size para lstm")
    parser.add_argument("--lstm-num-layers", default="", help="Lista de num_layers para lstm")
    parser.add_argument("--cnn-channels", default="", help="Lista de channels para cnn")
    parser.add_argument("--cnn-kernel-sizes", default="", help="Lista de kernel_size para cnn")
    parser.add_argument("--transformer-d-models", default="", help="Lista de d_model para transformer")
    parser.add_argument("--transformer-nheads", default="", help="Lista de nhead para transformer")
    parser.add_argument("--transformer-num-layers", default="", help="Lista de num_layers para transformer")
    parser.add_argument(
        "--transformer-dim-feedforwards",
        default="",
        help="Lista de dim_feedforward para transformer",
    )
    parser.add_argument("--min-folds", type=int, default=3, help="Minimo de folds validos para experimento elegivel")
    parser.add_argument("--max-folds", type=int, default=0, help="Maximo de folds usados por experimento (0 desabilita)")
    parser.add_argument(
        "--min-class-count-per-train",
        type=int,
        default=1,
        help="Minimo unico de amostras por classe em cada treino de fold",
    )
    parser.add_argument(
        "--min-class-counts-per-train",
        default="",
        help="Lista de minimos por classe separada por virgula para otimizar no walk-forward",
    )
    parser.add_argument("--objective", default="balanced_accuracy_stable", help="Nome do objetivo agregado")
    parser.add_argument(
        "--rank-metrics",
        default="score,mean_balanced_accuracy_trade,mean_accuracy_trade",
        help="Metricas para exportar rankings top-N separados por virgula",
    )
    parser.add_argument("--progress-every", type=int, default=10, help="Loga progresso a cada N experimentos")
    parser.add_argument("--top-k", type=int, default=10, help="Quantidade de linhas a exibir no console")
    return parser


def main() -> None:
    run_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
