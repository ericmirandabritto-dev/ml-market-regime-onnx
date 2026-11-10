from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from market_regime_core import ensure_dir, export_onnx_model, load_config, save_json
from position_management_model_core import (
    POSITION_MANAGEMENT_MODEL_SCHEMA_VERSION,
    build_position_dataset_diagnostics,
    chronological_trade_split,
    encode_action_targets,
    evaluate_position_management_model,
    fit_position_management_model,
    load_position_management_training_samples,
    resolve_action_names,
    resolve_feature_columns,
    validate_position_management_onnx_model,
    validate_train_action_classes,
)


RUNTIME_RELATIVE_DIR = Path("ML_MarketRegime_ONNX") / "position_management"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina e exporta um modelo ONNX para gestao de posicoes abertas")
    parser.add_argument("--manifest", default="", help="dataset_manifest.json gerado pelo pipeline de datasets")
    parser.add_argument("--training-samples", default="", help="Override opcional para o training_samples.*")
    parser.add_argument("--artifacts", required=True, help="Diretorio de saida")
    parser.add_argument("--config", required=True, help="Arquivo YAML de configuracao")
    parser.add_argument(
        "--validate-onnx",
        action="store_true",
        help="Compara a saida do ONNX com predict_proba do sklearn usando parte do conjunto de teste",
    )
    return parser


def path_to_windows_string(path: Path) -> str:
    return str(path).replace("/", "\\")


def build_runtime_manifest(artifacts_dir: Path) -> Dict:
    package_root = artifacts_dir / "runtime_package"
    runtime_dir = package_root / RUNTIME_RELATIVE_DIR
    relative_model_path = path_to_windows_string(RUNTIME_RELATIVE_DIR / "model.onnx")
    relative_metadata_path = path_to_windows_string(RUNTIME_RELATIVE_DIR / "model_metadata.json")
    relative_contract_path = path_to_windows_string(RUNTIME_RELATIVE_DIR / "runtime_contract.csv")

    return {
        "embedded_backtest": {
            "model_path": path_to_windows_string(artifacts_dir / "model.onnx"),
            "metadata_path": path_to_windows_string(artifacts_dir / "model_metadata.json"),
            "ea_resource_model_path": "ML_MarketRegime_ONNX\\artifacts_position_management\\model.onnx",
        },
        "external_live": {
            "package_dir": path_to_windows_string(package_root),
            "relative_model_path": relative_model_path,
            "relative_metadata_path": relative_metadata_path,
            "relative_runtime_contract_path": relative_contract_path,
            "target_root": "MQL5\\Files",
        },
        "external_tester": {
            "package_dir": path_to_windows_string(package_root),
            "relative_model_path": relative_model_path,
            "relative_metadata_path": relative_metadata_path,
            "relative_runtime_contract_path": relative_contract_path,
            "target_root": "Terminal\\Common\\Files",
        },
        "runtime_relative_dir": path_to_windows_string(RUNTIME_RELATIVE_DIR),
        "runtime_dir": path_to_windows_string(runtime_dir),
    }


def export_runtime_contract(
    destination: Path,
    feature_columns: list[str],
    action_names: list[str],
    fallback_action: str,
    exit_actions: list[str],
    use_validity_mask_on_inference: bool,
) -> None:
    rows = [
        ("key", "value"),
        ("project", "Position Management ONNX"),
        ("feature_count", str(len(feature_columns))),
        ("action_count", str(len(action_names))),
        ("fallback_action", fallback_action),
        ("use_validity_mask_on_inference", "1" if use_validity_mask_on_inference else "0"),
    ]

    for index, action in enumerate(action_names):
        rows.append((f"action_{index}", action))

    for index, action in enumerate(exit_actions):
        rows.append((f"exit_action_{index}", action))

    for index, feature_name in enumerate(feature_columns):
        rows.append((f"feature_{index}", feature_name))

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def export_runtime_package(
    artifacts_dir: Path,
    feature_columns: list[str],
    action_names: list[str],
    fallback_action: str,
    exit_actions: list[str],
    use_validity_mask_on_inference: bool,
) -> Dict:
    runtime_manifest = build_runtime_manifest(artifacts_dir)
    package_root = artifacts_dir / "runtime_package"
    runtime_dir = package_root / RUNTIME_RELATIVE_DIR

    ensure_dir(runtime_dir)
    shutil.copy2(artifacts_dir / "model.onnx", runtime_dir / "model.onnx")
    shutil.copy2(artifacts_dir / "model_metadata.json", runtime_dir / "model_metadata.json")
    export_runtime_contract(
        destination=artifacts_dir / "runtime_contract.csv",
        feature_columns=feature_columns,
        action_names=action_names,
        fallback_action=fallback_action,
        exit_actions=exit_actions,
        use_validity_mask_on_inference=use_validity_mask_on_inference,
    )
    shutil.copy2(artifacts_dir / "runtime_contract.csv", runtime_dir / "runtime_contract.csv")

    save_json(artifacts_dir / "runtime_package_manifest.json", runtime_manifest)
    save_json(runtime_dir / "runtime_package_manifest.json", runtime_manifest)
    return runtime_manifest


def build_model_metadata(
    cfg: Dict,
    manifest_path: Path,
    training_samples_path: Path,
    onnx_contract: Dict,
    feature_columns: list[str],
    action_names: list[str],
    diagnostics: Dict,
    metrics: Dict,
    validation_payload: Dict,
    runtime_manifest: Dict,
) -> Dict:
    action_to_id = {action: index for index, action in enumerate(action_names)}
    return {
        "project": "Position Management ONNX",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": str(manifest_path),
        "training_samples_path": str(training_samples_path),
        "input_name": onnx_contract["input_name"],
        "output_names": onnx_contract["output_names"],
        "feature_columns": feature_columns,
        "position_management_model_schema_version": POSITION_MANAGEMENT_MODEL_SCHEMA_VERSION,
        "target_column": "best_action",
        "action_to_id": action_to_id,
        "id_to_action": {str(index): action for action, index in action_to_id.items()},
        "actions_trained": action_names,
        "fallback_action": str(cfg.get("train", {}).get("fallback_action", "hold_base")),
        "use_validity_mask_on_inference": bool(cfg.get("train", {}).get("use_validity_mask_on_inference", True)),
        "runtime": {
            "default_action": str(cfg.get("train", {}).get("fallback_action", "hold_base")),
            "exit_actions": [str(value) for value in cfg.get("runtime", {}).get("exit_actions", ["exit_now"])],
        },
        "validity_columns": {action: f"{action}_is_valid" for action in action_names},
        "utility_columns": {action: f"{action}_utility" for action in action_names},
        "dataset_diagnostics": diagnostics,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "top2_accuracy": metrics["top2_accuracy"],
            "mean_predicted_utility": metrics["mean_predicted_utility"],
            "mean_best_utility": metrics["mean_best_utility"],
            "mean_utility_regret": metrics["mean_utility_regret"],
            "mean_predicted_minus_hold_base": metrics["mean_predicted_minus_hold_base"],
            "mean_predicted_minus_exit_now": metrics["mean_predicted_minus_exit_now"],
            "exit_decision_accuracy": metrics["exit_decision_accuracy"],
            "raw_invalid_rate": metrics["raw_invalid_rate"],
            "selected_invalid_rate": metrics["selected_invalid_rate"],
            "class_distribution_test": metrics["class_distribution_test"],
            "predicted_action_distribution": metrics["predicted_action_distribution"],
        },
        "onnx_validation": validation_payload,
        "runtime_package": runtime_manifest,
        "config": cfg,
    }


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    artifacts_dir = Path(args.artifacts)
    ensure_dir(artifacts_dir)

    cfg = load_config(config_path)
    manifest_path = Path(args.manifest or cfg.get("data", {}).get("manifest", "datasets/position_management/dataset_manifest.json"))
    training_samples_override = Path(args.training_samples) if args.training_samples else None

    print("[pm-train] stage=load_samples", flush=True)
    manifest, samples_df, training_samples_path = load_position_management_training_samples(
        manifest_path=manifest_path,
        training_samples_override=training_samples_override,
    )

    print("[pm-train] stage=resolve_schema", flush=True)
    feature_columns = resolve_feature_columns(cfg, manifest, samples_df)
    action_names = resolve_action_names(samples_df, manifest, cfg)
    encoded_df, action_to_id = encode_action_targets(samples_df, action_names)

    print("[pm-train] stage=split_holdout", flush=True)
    train_df, test_df, _ = chronological_trade_split(encoded_df, cfg)

    ok, issues = validate_train_action_classes(
        train_df=train_df,
        action_names=action_names,
        min_class_count=int(cfg.get("train", {}).get("min_class_count_per_train", 1)),
    )
    if not ok:
        raise ValueError(" ".join(issues))

    diagnostics = build_position_dataset_diagnostics(
        samples_df=encoded_df,
        train_df=train_df,
        test_df=test_df,
        action_names=action_names,
    )

    print("[pm-train] stage=fit_model", flush=True)
    model = fit_position_management_model(cfg=cfg, train_df=train_df, feature_columns=feature_columns)

    print("[pm-train] stage=evaluate", flush=True)
    metrics, confusion_df, predictions_df = evaluate_position_management_model(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
        action_names=action_names,
        cfg=cfg,
    )

    print("[pm-train] stage=export_onnx", flush=True)
    onnx_contract = export_onnx_model(
        model=model,
        feature_count=len(feature_columns),
        output_path=artifacts_dir / "model.onnx",
        opset=int(cfg["export"].get("onnx_opset", 17)),
    )

    validate_rows = int(cfg["export"].get("validate_rows", 256))
    validation_payload = {"status": "disabled"}
    if args.validate_onnx:
        print("[pm-train] stage=validate_onnx", flush=True)
        validation_payload = validate_position_management_onnx_model(
            model=model,
            onnx_path=artifacts_dir / "model.onnx",
            sample_df=test_df.iloc[:validate_rows].copy(),
            feature_columns=feature_columns,
            class_count=len(action_to_id),
        )

    confusion_df.to_csv(artifacts_dir / "confusion_matrix.csv")
    predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    save_json(artifacts_dir / "classification_report.json", metrics["classification_report"])
    runtime_manifest = build_runtime_manifest(artifacts_dir)

    metadata = build_model_metadata(
        cfg=cfg,
        manifest_path=manifest_path,
        training_samples_path=training_samples_path,
        onnx_contract=onnx_contract,
        feature_columns=feature_columns,
        action_names=action_names,
        diagnostics=diagnostics,
        metrics=metrics,
        validation_payload=validation_payload,
        runtime_manifest=runtime_manifest,
    )
    save_json(artifacts_dir / "model_metadata.json", metadata)
    runtime_manifest = export_runtime_package(
        artifacts_dir=artifacts_dir,
        feature_columns=feature_columns,
        action_names=action_names,
        fallback_action=str(cfg.get("train", {}).get("fallback_action", "hold_base")),
        exit_actions=[str(value) for value in cfg.get("runtime", {}).get("exit_actions", ["exit_now"])],
        use_validity_mask_on_inference=bool(cfg.get("train", {}).get("use_validity_mask_on_inference", True)),
    )

    print(
        "[pm-train] "
        f"rows_total={len(encoded_df)} "
        f"rows_train={len(train_df)} "
        f"rows_test={len(test_df)} "
        f"trades_train={diagnostics['trades_train']} "
        f"trades_test={diagnostics['trades_test']} "
        f"actions={','.join(action_names)} "
        f"model_type={cfg['train'].get('model_type', 'logistic_regression')} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"mean_predicted_utility={metrics['mean_predicted_utility'] if metrics['mean_predicted_utility'] is not None else 'nan'} "
        f"mean_utility_regret={metrics['mean_utility_regret'] if metrics['mean_utility_regret'] is not None else 'nan'} "
        f"runtime_package={runtime_manifest['external_live']['package_dir']} "
        f"artifacts={artifacts_dir}",
        flush=True,
    )

    for warning in diagnostics["warnings"]:
        print(f"[warning] {warning}", flush=True)


if __name__ == "__main__":
    main()
