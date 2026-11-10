from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from market_regime_core import (
    FEATURE_SCHEMA_VERSION,
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_feature_frame,
    build_model_backend,
    chronological_split,
    dataset_diagnostics,
    ensure_dir,
    evaluate_model,
    export_market_runtime_contract,
    export_regime_onnx,
    load_candles_csv,
    load_config,
    save_json,
)
from walkforward import WalkForwardPolicy, evaluate_walkforward, resolve_purge_bars


def build_single_run_metadata(
    cfg: Dict,
    input_path: Path,
    onnx_contract: Dict,
    feature_columns: list[str],
    diagnostics: Dict,
    metrics: Dict,
    validation_payload: Dict,
    train_rows: int,
    test_rows: int,
) -> Dict:
    return {
        "project": "Market Regime ONNX",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "input_name": onnx_contract["input_name"],
        "output_names": onnx_contract["output_names"],
        "feature_columns": feature_columns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "dataset_diagnostics": diagnostics,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "class_distribution_test": metrics["class_distribution_test"],
        },
        "onnx_validation": validation_payload,
        "config": cfg,
    }


def train_single(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    artifacts_dir = Path(args.artifacts)
    ensure_dir(artifacts_dir)

    cfg = load_config(args.config)
    print("[train] stage=load_csv", flush=True)
    candles = load_candles_csv(input_path)
    print("[train] stage=build_features", flush=True)
    dataset, feature_columns = build_feature_frame(candles, cfg)
    print("[train] stage=split_holdout", flush=True)
    train_df, test_df = chronological_split(dataset, float(cfg["train"]["test_size"]))
    diagnostics = dataset_diagnostics(candles, dataset, train_df, test_df)

    print("[train] stage=fit_model", flush=True)
    model = build_model_backend(cfg)
    model.fit(train_df, feature_columns)

    print("[train] stage=evaluate", flush=True)
    test_history_df = dataset.iloc[:-len(test_df)].tail(model.history_rows_needed)
    metrics, confusion_df, predictions_df = evaluate_model(
        model,
        test_df,
        feature_columns,
        history_df=test_history_df,
    )

    print("[train] stage=export_onnx", flush=True)
    sample_df = None
    if args.validate_onnx:
        validate_rows = int(cfg["export"].get("validate_rows", 32))
        if validate_rows < 1:
            raise ValueError("export.validate_rows deve ser maior que zero")
        sample_df = test_df.iloc[:validate_rows]
    onnx_contract, validation_payload = export_regime_onnx(
        model=model,
        output_path=artifacts_dir / "model.onnx",
        opset=int(cfg["export"]["onnx_opset"]),
        feature_columns=feature_columns,
        sample_df=sample_df,
        history_df=test_history_df,
    )

    confusion_df.to_csv(artifacts_dir / "confusion_matrix.csv")
    predictions_df.reset_index().to_csv(artifacts_dir / "test_predictions.csv", index=False)
    save_json(artifacts_dir / "classification_report.json", metrics["classification_report"])

    metadata = build_single_run_metadata(
        cfg=cfg,
        input_path=input_path,
        onnx_contract=onnx_contract,
        feature_columns=feature_columns,
        diagnostics=diagnostics,
        metrics=metrics,
        validation_payload=validation_payload,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )
    save_json(artifacts_dir / "model_metadata.json", metadata)
    export_market_runtime_contract(
        destination=artifacts_dir / "runtime_contract.csv",
        cfg=cfg,
        feature_columns=feature_columns,
        onnx_contract=onnx_contract,
    )

    print(
        "[train] "
        f"mode=single "
        f"dataset_rows={len(dataset)} "
        f"train_rows={len(train_df)} "
        f"test_rows={len(test_df)} "
        f"model_type={cfg['train'].get('model_type', 'logistic_regression')} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"artifacts={artifacts_dir}"
    )

    for warning in diagnostics["warnings"]:
        print(f"[warning] {warning}")


def walkforward_policy_from_args(cfg: Dict, args: argparse.Namespace) -> WalkForwardPolicy:
    walk_cfg = cfg.get("walkforward", {})
    return WalkForwardPolicy(
        train_window=int(args.train_window or walk_cfg.get("train_window", 20000)),
        validation_window=int(args.validation_window if args.validation_window is not None else walk_cfg.get("validation_window", 0)),
        trade_window=int(args.trade_window or walk_cfg.get("trade_window", 2500)),
        step_window=int(args.step_window or walk_cfg.get("step_window", 2500)),
        purge_bars=resolve_purge_bars(cfg, args.purge_bars),
        min_folds=int(args.min_folds or walk_cfg.get("min_folds", 3)),
        max_folds=int(args.max_folds if args.max_folds is not None else walk_cfg.get("max_folds", 0)),
        min_class_count_per_train=int(
            args.min_class_count_per_train
            or walk_cfg.get("min_class_count_per_train", 1)
        ),
        objective=str(args.objective or walk_cfg.get("objective", "balanced_accuracy_stable")),
    )


def train_walkforward(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    artifacts_dir = Path(args.artifacts)
    ensure_dir(artifacts_dir)

    cfg = load_config(args.config)
    print("[train] stage=load_csv", flush=True)
    candles = load_candles_csv(input_path)
    print("[train] stage=build_features", flush=True)
    dataset, feature_columns = build_feature_frame(candles, cfg)
    policy = walkforward_policy_from_args(cfg, args)
    runtime_root = str(args.runtime_root or cfg.get("walkforward", {}).get("runtime_root", "ML_MarketRegime_ONNX\\walkforward"))
    print(
        "[train] "
        f"stage=walkforward_start "
        f"dataset_rows={len(dataset)} "
        f"train_window={policy.train_window} "
        f"trade_window={policy.trade_window} "
        f"step_window={policy.step_window} "
        f"min_folds={policy.min_folds} "
        f"max_folds={policy.max_folds}",
        flush=True,
    )

    payload = evaluate_walkforward(
        candles=candles,
        dataset=dataset,
        feature_columns=feature_columns,
        cfg=cfg,
        policy=policy,
        export_dir=artifacts_dir,
        validate_onnx=args.validate_onnx,
        runtime_root=runtime_root,
        symbol=str(args.symbol or cfg.get("walkforward", {}).get("symbol", "")),
        timeframe=str(args.timeframe or cfg.get("walkforward", {}).get("timeframe", "")),
        input_csv=str(input_path),
        enable_progress=True,
        progress_label="package",
        fold_progress_every=int(args.fold_progress_every),
        incremental_export=bool(args.incremental_export),
    )

    aggregate = payload["aggregate"]
    print(
        "[train] "
        f"mode=walkforward "
        f"dataset_rows={len(dataset)} "
        f"planned_folds={aggregate['planned_folds']} "
        f"available_folds={aggregate['available_folds']} "
        f"valid_folds={aggregate['valid_folds']} "
        f"reused_folds={aggregate['reused_folds']} "
        f"trained_folds={aggregate['trained_folds']} "
        f"live_tail_available={aggregate['live_tail_available']} "
        f"live_tail_reused={aggregate['live_tail_reused']} "
        f"eligible={aggregate['eligible']} "
        f"score={aggregate['score'] if aggregate['score'] is not None else 'nan'} "
        f"mean_balanced_accuracy_trade={aggregate['mean_balanced_accuracy_trade'] if aggregate['mean_balanced_accuracy_trade'] is not None else 'nan'} "
        f"artifacts={artifacts_dir}"
    )

    if aggregate["valid_folds"] < policy.min_folds:
        print(
            "[warning] "
            f"Walk-forward gerou menos folds validos que o minimo requerido ({aggregate['valid_folds']}/{policy.min_folds})."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export a market regime classifier to ONNX")
    parser.add_argument("--input", required=True, help="CSV com candles OHLCV")
    parser.add_argument("--artifacts", required=True, help="Diretorio de saida")
    parser.add_argument("--config", type=Path, help="Arquivo YAML de configuracao (opcional)")
    parser.add_argument("--mode", choices=["single", "walkforward"], default="single", help="Modo de treino/exportacao")
    parser.add_argument(
        "--validate-onnx",
        action="store_true",
        help="Exige paridade entre as probabilidades ONNX e as do modelo Python",
    )

    parser.add_argument("--train-window", type=int, default=None, help="Janela de treino walk-forward em barras")
    parser.add_argument("--validation-window", type=int, default=None, help="Janela opcional de validacao por fold")
    parser.add_argument("--trade-window", type=int, default=None, help="Janela ativa de trading por fold")
    parser.add_argument("--step-window", type=int, default=None, help="Passo entre folds")
    parser.add_argument("--purge-bars", default=None, help="Barras de purge entre treino/validacao/trade ou 'auto'")
    parser.add_argument("--min-folds", type=int, default=None, help="Minimo de folds validos")
    parser.add_argument("--max-folds", type=int, default=None, help="Maximo de folds usados no walk-forward (0 desabilita)")
    parser.add_argument(
        "--min-class-count-per-train",
        type=int,
        default=None,
        help="Minimo por classe no treino de cada fold",
    )
    parser.add_argument("--objective", default=None, help="Nome do objetivo agregado walk-forward")
    parser.add_argument("--runtime-root", default=None, help="Raiz relativa usada no manifesto para o runtime do EA")
    parser.add_argument("--symbol", default="", help="Simbolo a registrar no manifesto walk-forward")
    parser.add_argument("--timeframe", default="", help="Timeframe a registrar no manifesto walk-forward")
    parser.add_argument("--fold-progress-every", type=int, default=1, help="Loga progresso do walk-forward a cada N folds")
    parser.add_argument(
        "--incremental-export",
        action="store_true",
        help="Reaproveita folds ja exportados quando config/politica e janelas coincidirem",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "walkforward":
        train_walkforward(args)
        return

    train_single(args)


if __name__ == "__main__":
    main()
