from __future__ import annotations

import copy
import hashlib
import itertools
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import yaml

from market_regime_core import (
    build_feature_frame,
    load_candles_csv,
    load_config,
    normalize_model_type,
    save_json,
)
from walkforward import WalkForwardPolicy, evaluate_walkforward, resolve_purge_bars


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
GENERIC_WFO_SRC = WORKSPACE_ROOT / "MT5_WalkForward_Window_Optimizer" / "src"
if str(GENERIC_WFO_SRC) not in sys.path:
    sys.path.insert(0, str(GENERIC_WFO_SRC))

from mt5_walkforward_core import parse_ini_file, run_walkforward_search, write_ini_file  # noqa: E402


DEFAULT_SYNC_FILES = [
    "EA_MarketRegime_ONNX.mq5",
    "ML_MarketRegime_ONNX\\artifacts\\model.onnx",
    "ML_MarketRegime_ONNX\\artifacts_position_management\\model.onnx",
]

DEFAULT_SHORT_KEYS = {
    "train.model_type": "model",
    "label.horizon_bars": "h",
    "label.trend_threshold_atr": "thr",
    "walkforward.train_window": "train",
    "walkforward.validation_window": "valid",
    "walkforward.trade_window": "trade",
    "walkforward.step_window": "step",
    "walkforward.min_class_count_per_train": "mincls",
    "train.C": "c",
    "train.max_iter": "iter",
    "train.n_estimators": "trees",
    "train.max_depth": "depth",
    "train.min_samples_leaf": "leaf",
    "train.lookback_bars": "lookback",
    "train.epochs": "epochs",
    "train.batch_size": "batch",
    "train.learning_rate": "lr",
    "train.weight_decay": "wd",
    "train.dropout": "drop",
    "train.mlp_hidden_sizes": "mlp",
    "train.lstm_hidden_size": "lstmh",
    "train.lstm_num_layers": "lstml",
    "train.cnn_channels": "cnnc",
    "train.cnn_kernel_size": "cnnk",
    "train.transformer_d_model": "tdm",
    "train.transformer_nhead": "th",
    "train.transformer_num_layers": "tl",
    "train.transformer_dim_feedforward": "tff",
}

VALUE_ALIASES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "deep_mlp": "mlp",
    "transformer": "trf",
}


@dataclass
class ModelCandidate:
    candidate_id: str
    path_id: str
    source: str
    cfg: Dict[str, Any] | None
    package_dir: Path
    runtime_root: str
    manifest_relative_path: str
    parameter_values: Dict[str, Any]
    metadata: Dict[str, Any]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuracao invalida em {path}")
    return payload


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    segments = [segment.strip() for segment in dotted_key.split(".") if segment.strip()]
    if not segments:
        raise ValueError(f"Chave de grid invalida: {dotted_key}")
    for segment in segments[:-1]:
        next_value = cursor.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[segment] = next_value
        cursor = next_value
    cursor[segments[-1]] = value


def get_nested_value(payload: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = payload
    for segment in [segment.strip() for segment in dotted_key.split(".") if segment.strip()]:
        if not isinstance(cursor, dict) or segment not in cursor:
            return default
        cursor = cursor[segment]
    return cursor


def flatten_mapping(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_mapping(value, prefix=dotted_key))
        else:
            result[dotted_key] = value
    return result


def slugify_token(value: Any) -> str:
    text = VALUE_ALIASES.get(str(value), str(value))
    replacements = {
        " ": "",
        "\\": "",
        "/": "",
        ",": "-",
        "[": "",
        "]": "",
        "(": "",
        ")": "",
        "{": "",
        "}": "",
        "'": "",
        '"': "",
        ":": "",
        ";": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(".", "p")
    text = text.replace("True", "1").replace("False", "0")
    return text.lower()


def build_candidate_id(parameter_values: Dict[str, Any]) -> str:
    signature_parts: List[str] = []
    for dotted_key, value in parameter_values.items():
        short_key = DEFAULT_SHORT_KEYS.get(dotted_key, dotted_key.split(".")[-1])
        signature_parts.append(f"{short_key}-{slugify_token(value)}")
    signature = "__".join(signature_parts)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]

    model_token = slugify_token(parameter_values.get("train.model_type", "model"))
    horizon_token = slugify_token(parameter_values.get("label.horizon_bars", "na"))
    threshold_token = slugify_token(parameter_values.get("label.trend_threshold_atr", "na"))
    train_token = slugify_token(parameter_values.get("walkforward.train_window", "na"))
    valid_token = slugify_token(parameter_values.get("walkforward.validation_window", "na"))
    trade_token = slugify_token(parameter_values.get("walkforward.trade_window", "na"))
    step_token = slugify_token(parameter_values.get("walkforward.step_window", "na"))
    return (
        f"m-{model_token}_h-{horizon_token}_thr-{threshold_token}"
        f"_tr-{train_token}_v-{valid_token}_td-{trade_token}_s-{step_token}_{digest}"
    )


def coerce_grid_values(raw_value: Any) -> List[Any]:
    if isinstance(raw_value, list):
        return list(raw_value)
    if isinstance(raw_value, tuple):
        return list(raw_value)
    return [raw_value]


def build_path_id(candidate_id: str) -> str:
    digest = hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"c_{digest}"


def build_walkforward_policy(cfg: Dict[str, Any]) -> WalkForwardPolicy:
    walk_cfg = cfg.get("walkforward", {})
    return WalkForwardPolicy(
        train_window=int(walk_cfg.get("train_window", 20000)),
        validation_window=int(walk_cfg.get("validation_window", 0)),
        trade_window=int(walk_cfg.get("trade_window", 2500)),
        step_window=int(walk_cfg.get("step_window", 2500)),
        purge_bars=resolve_purge_bars(cfg, walk_cfg.get("purge_bars", "auto")),
        min_folds=int(walk_cfg.get("min_folds", 3)),
        max_folds=int(walk_cfg.get("max_folds", 0)),
        min_class_count_per_train=int(walk_cfg.get("min_class_count_per_train", 1)),
        objective=str(walk_cfg.get("objective", "balanced_accuracy_stable")),
    )


def build_runtime_root(deploy_root: str, candidate_id: str) -> str:
    return str(Path(deploy_root) / candidate_id).replace("/", "\\")


def build_grid_candidates(
    orchestrator_cfg: Dict[str, Any],
    package_artifacts_root: Path,
) -> List[ModelCandidate]:
    model_cfg = orchestrator_cfg["model_candidates"]
    base_cfg_path = resolve_path(PROJECT_ROOT, str(model_cfg["base_config"]))
    base_cfg = load_config(base_cfg_path)
    grid = model_cfg.get("grid", {})
    if not isinstance(grid, dict) or not grid:
        raise ValueError("model_candidates.grid precisa ter pelo menos uma chave")

    grid_keys = list(grid.keys())
    grid_values = [coerce_grid_values(grid[key]) for key in grid_keys]
    deploy_root = str(model_cfg.get("deploy_root", "ML_MarketRegime_ONNX\\ea_optimizer")).replace("/", "\\")

    candidates: List[ModelCandidate] = []
    for combo_values in itertools.product(*grid_values):
        parameter_values = {grid_keys[index]: combo_values[index] for index in range(len(grid_keys))}
        candidate_cfg = copy.deepcopy(base_cfg)
        for dotted_key, value in parameter_values.items():
            set_nested_value(candidate_cfg, dotted_key, value)

        model_type = normalize_model_type(get_nested_value(candidate_cfg, "train.model_type", "logistic_regression"))
        set_nested_value(candidate_cfg, "train.model_type", model_type)
        parameter_values["train.model_type"] = model_type

        candidate_id = build_candidate_id(parameter_values)
        path_id = build_path_id(candidate_id)
        runtime_root = build_runtime_root(deploy_root, candidate_id)
        package_dir = package_artifacts_root / path_id
        manifest_relative_path = f"{runtime_root}\\manifest.csv"
        candidates.append(
            ModelCandidate(
                candidate_id=candidate_id,
                path_id=path_id,
                source="grid",
                cfg=candidate_cfg,
                package_dir=package_dir,
                runtime_root=runtime_root,
                manifest_relative_path=manifest_relative_path,
                parameter_values=parameter_values,
                metadata={"base_config": str(base_cfg_path)},
            )
        )
    return candidates


def build_existing_package_candidates(
    orchestrator_cfg: Dict[str, Any],
) -> List[ModelCandidate]:
    model_cfg = orchestrator_cfg["model_candidates"]
    deploy_root = str(model_cfg.get("deploy_root", "ML_MarketRegime_ONNX\\ea_optimizer")).replace("/", "\\")
    packages = model_cfg.get("packages", [])
    if not isinstance(packages, list) or not packages:
        raise ValueError("model_candidates.packages precisa ter pelo menos um item")

    candidates: List[ModelCandidate] = []
    for entry in packages:
        if not isinstance(entry, dict):
            raise ValueError("Cada item de model_candidates.packages deve ser um objeto")
        candidate_id = str(entry.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("candidate_id obrigatorio em model_candidates.packages")
        path_id = str(entry.get("path_id", "")).strip() or build_path_id(candidate_id)
        package_dir = resolve_path(PROJECT_ROOT, str(entry["package_dir"]))
        runtime_root = build_runtime_root(deploy_root, candidate_id)
        manifest_relative_path = str(entry.get("manifest_relative_path", f"{runtime_root}\\manifest.csv")).replace("/", "\\")
        candidates.append(
            ModelCandidate(
                candidate_id=candidate_id,
                path_id=path_id,
                source="existing_package",
                cfg=None,
                package_dir=package_dir,
                runtime_root=runtime_root,
                manifest_relative_path=manifest_relative_path,
                parameter_values=dict(entry.get("parameter_values", {})),
                metadata={"package_dir": str(package_dir)},
            )
        )
    return candidates


def build_candidate_list_candidates(
    orchestrator_cfg: Dict[str, Any],
    package_artifacts_root: Path,
) -> List[ModelCandidate]:
    model_cfg = orchestrator_cfg["model_candidates"]
    base_cfg_path = resolve_path(PROJECT_ROOT, str(model_cfg["base_config"]))
    base_cfg = load_config(base_cfg_path)
    deploy_root = str(model_cfg.get("deploy_root", "ML_MarketRegime_ONNX\\ea_optimizer")).replace("/", "\\")
    raw_candidates = model_cfg.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("model_candidates.candidates precisa ter pelo menos um item")

    candidates: List[ModelCandidate] = []
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            raise ValueError("Cada item de model_candidates.candidates deve ser um objeto")

        entry_copy = copy.deepcopy(entry)
        candidate_id_override = str(entry_copy.pop("candidate_id", "")).strip()
        metadata = {}
        if "notes" in entry_copy:
            metadata["notes"] = entry_copy.pop("notes")

        parameter_values = flatten_mapping(entry_copy)
        if not parameter_values:
            raise ValueError("Cada candidato explicito precisa definir ao menos um parametro")

        candidate_cfg = copy.deepcopy(base_cfg)
        for dotted_key, value in parameter_values.items():
            set_nested_value(candidate_cfg, dotted_key, value)

        model_type = normalize_model_type(get_nested_value(candidate_cfg, "train.model_type", "logistic_regression"))
        set_nested_value(candidate_cfg, "train.model_type", model_type)
        parameter_values["train.model_type"] = model_type

        candidate_id = candidate_id_override or build_candidate_id(parameter_values)
        path_id = build_path_id(candidate_id)
        runtime_root = build_runtime_root(deploy_root, candidate_id)
        package_dir = package_artifacts_root / path_id
        manifest_relative_path = f"{runtime_root}\\manifest.csv"
        candidates.append(
            ModelCandidate(
                candidate_id=candidate_id,
                path_id=path_id,
                source="candidate_list",
                cfg=candidate_cfg,
                package_dir=package_dir,
                runtime_root=runtime_root,
                manifest_relative_path=manifest_relative_path,
                parameter_values=parameter_values,
                metadata=metadata,
            )
        )
    return candidates


def build_model_candidates(orchestrator_cfg: Dict[str, Any], package_artifacts_root: Path) -> List[ModelCandidate]:
    mode = str(orchestrator_cfg["model_candidates"].get("mode", "grid")).strip().lower()
    if mode == "grid":
        return build_grid_candidates(orchestrator_cfg, package_artifacts_root)
    if mode == "existing_packages":
        return build_existing_package_candidates(orchestrator_cfg)
    if mode in {"candidate_list", "explicit_list"}:
        return build_candidate_list_candidates(orchestrator_cfg, package_artifacts_root)
    raise ValueError(f"model_candidates.mode invalido: {mode}")


def feature_dataset_cache_key(cfg: Dict[str, Any]) -> str:
    payload = {
        "feature": cfg.get("feature", {}),
        "label": cfg.get("label", {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def train_package_candidate(
    candidate: ModelCandidate,
    orchestrator_cfg: Dict[str, Any],
    dataset_cache: Dict[str, tuple[pd.DataFrame, List[str]]],
    candles: pd.DataFrame,
    input_csv: Path,
) -> Dict[str, Any]:
    if candidate.cfg is None:
        if not candidate.package_dir.exists():
            raise FileNotFoundError(f"Package inexistente para candidate_id={candidate.candidate_id}: {candidate.package_dir}")
        return {
            "candidate_id": candidate.candidate_id,
            "status": "reused_existing_package",
            "eligible": True,
            "package_dir": str(candidate.package_dir),
            "runtime_root": candidate.runtime_root,
            "manifest_relative_path": candidate.manifest_relative_path,
        }

    cache_key = feature_dataset_cache_key(candidate.cfg)
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = build_feature_frame(candles, candidate.cfg)
    dataset, feature_columns = dataset_cache[cache_key]

    if candidate.package_dir.exists():
        shutil.rmtree(candidate.package_dir)
    ensure_dir(candidate.package_dir)

    model_cfg = orchestrator_cfg["model_candidates"]
    symbol = str(model_cfg.get("symbol", ""))
    timeframe = str(model_cfg.get("timeframe", ""))
    payload = evaluate_walkforward(
        candles=candles,
        dataset=dataset,
        feature_columns=feature_columns,
        cfg=candidate.cfg,
        policy=build_walkforward_policy(candidate.cfg),
        export_dir=candidate.package_dir,
        validate_onnx=bool(model_cfg.get("validate_onnx", False)),
        runtime_root=candidate.runtime_root,
        symbol=symbol,
        timeframe=timeframe,
        input_csv=str(input_csv),
        enable_progress=True,
        progress_label=candidate.candidate_id,
        fold_progress_every=int(model_cfg.get("fold_progress_every", 10)),
        incremental_export=bool(model_cfg.get("incremental_export", False)),
    )

    aggregate = payload["aggregate"]
    summary = {
        "candidate_id": candidate.candidate_id,
        "status": "package_ready",
        "eligible": bool(aggregate.get("eligible", False)),
        "package_dir": str(candidate.package_dir),
        "runtime_root": candidate.runtime_root,
        "manifest_relative_path": candidate.manifest_relative_path,
        "planned_folds": int(aggregate.get("planned_folds", 0)),
        "valid_folds": int(aggregate.get("valid_folds", 0)),
        "score": aggregate.get("score"),
        "mean_balanced_accuracy_trade": aggregate.get("mean_balanced_accuracy_trade"),
    }
    save_json(candidate.package_dir / "ea_optimizer_package_summary.json", summary)
    return summary


def deploy_package_to_tester(candidate: ModelCandidate, mt5_data_root: Path) -> Path:
    target_dir = mt5_data_root / "MQL5" / "Files" / Path(candidate.runtime_root)
    common_target_dir = mt5_data_root.parent / "Common" / "Files" / Path(candidate.runtime_root)
    target_dirs = [target_dir, common_target_dir]

    for destination in target_dirs:
        if destination.exists():
            shutil.rmtree(destination)
        ensure_dir(destination.parent)
        shutil.copytree(candidate.package_dir, destination)

    return target_dir


def parse_tester_inputs_set_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff") or b"\x00" in raw[:200]:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def render_candidate_template(
    base_template_path: Path,
    manifest_relative_path: str,
    destination: Path,
    tester_inputs_override: Dict[str, str] | None = None,
) -> Path:
    template_data = parse_ini_file(base_template_path)
    tester_inputs = template_data.setdefault("TesterInputs", {})
    if tester_inputs_override:
        tester_inputs.update(tester_inputs_override)
    tester_inputs["InpManifestFile"] = manifest_relative_path
    ensure_dir(destination.parent)
    write_ini_file(template_data, destination)
    return destination


def build_sync_config(orchestrator_cfg: Dict[str, Any], compile_log: Path, mt5_data_root: Path) -> Dict[str, Any]:
    sync_cfg = copy.deepcopy(orchestrator_cfg.get("sync", {}))
    sync_cfg.setdefault("enabled", True)
    sync_cfg.setdefault("source_root", str(WORKSPACE_ROOT))
    sync_cfg.setdefault("target_root", str(mt5_data_root / "MQL5" / "Experts" / "Eric"))
    sync_cfg.setdefault("compile", True)
    sync_cfg.setdefault("compile_target", "EA_MarketRegime_ONNX.mq5")
    sync_cfg.setdefault("files", list(DEFAULT_SYNC_FILES))
    sync_cfg["compile_log"] = str(compile_log)
    return sync_cfg


def build_candidate_wfo_config(
    orchestrator_cfg: Dict[str, Any],
    candidate: ModelCandidate,
    candidate_root: Path,
    template_path: Path,
    mt5_data_root: Path,
) -> Dict[str, Any]:
    compile_log = candidate_root / "compile.log"
    return {
        "mt5": copy.deepcopy(orchestrator_cfg["mt5"]),
        "sync": build_sync_config(orchestrator_cfg, compile_log=compile_log, mt5_data_root=mt5_data_root),
        "template": {
            "tester_ini": str(template_path),
            "report_name_prefix": f"{orchestrator_cfg['template']['report_name_prefix']}_{candidate.candidate_id}",
        },
        "search": copy.deepcopy(orchestrator_cfg["search"]),
        "execution": copy.deepcopy(orchestrator_cfg["execution"]),
        "ranking": copy.deepcopy(orchestrator_cfg.get("ranking", {})),
        "outputs": {
            "artifact_root": str(candidate_root / "r"),
        },
    }


def load_best_wfo_row(summary_csv: Path) -> Dict[str, Any] | None:
    if not summary_csv.exists():
        return None
    frame = pd.read_csv(summary_csv)
    if frame.empty:
        return None
    return frame.iloc[0].to_dict()


def build_global_summary_row(
    candidate: ModelCandidate,
    package_summary: Dict[str, Any],
    best_wfo_row: Dict[str, Any] | None,
    candidate_root: Path,
) -> Dict[str, Any]:
    row = {
        "candidate_id": candidate.candidate_id,
        "candidate_source": candidate.source,
        "candidate_path_id": candidate.path_id,
        "package_status": package_summary.get("status"),
        "package_eligible": package_summary.get("eligible"),
        "package_dir": str(candidate.package_dir),
        "runtime_root": candidate.runtime_root,
        "manifest_relative_path": candidate.manifest_relative_path,
        "candidate_artifact_root": str(candidate_root),
    }

    for dotted_key, value in candidate.parameter_values.items():
        row[f"cfg_{dotted_key}"] = value

    if best_wfo_row is None:
        row["mt5_status"] = "not_run"
        return row

    row["mt5_status"] = "ok"
    for key, value in best_wfo_row.items():
        row[f"mt5_{key}"] = value
    return row


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def run_end_to_end_mt5_walkforward(config_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    orchestrator_cfg = load_yaml(config_path)
    artifacts_root = ensure_dir(resolve_path(PROJECT_ROOT, str(orchestrator_cfg["outputs"]["artifact_root"])))
    package_artifacts_root = ensure_dir(
        resolve_path(
            PROJECT_ROOT,
            str(orchestrator_cfg["model_candidates"].get("artifact_root", str(artifacts_root / "pkg"))),
        )
    )
    candidates = build_model_candidates(orchestrator_cfg, package_artifacts_root=package_artifacts_root)
    if not candidates:
        raise ValueError("Nenhum candidato de modelo foi gerado")

    model_cfg = orchestrator_cfg["model_candidates"]
    input_csv = resolve_path(PROJECT_ROOT, str(model_cfg["input_csv"])) if model_cfg.get("input_csv") else None
    candles = load_candles_csv(input_csv) if input_csv is not None else None
    dataset_cache: Dict[str, tuple[pd.DataFrame, List[str]]] = {}
    mt5_data_root = resolve_path(PROJECT_ROOT, str(orchestrator_cfg["mt5"]["data_root"]))
    base_template_path = resolve_path(PROJECT_ROOT, str(orchestrator_cfg["template"]["tester_ini"]))
    tester_inputs_override = None
    tester_set_file = orchestrator_cfg.get("template", {}).get("tester_set_file")
    if tester_set_file:
        tester_inputs_override = parse_tester_inputs_set_file(resolve_path(PROJECT_ROOT, str(tester_set_file)))

    global_rows: List[Dict[str, Any]] = []
    package_rows: List[Dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        candidate_root = ensure_dir(artifacts_root / candidate.path_id)
        print(
            "[ea-mt5-wfo] "
            f"candidate={index}/{len(candidates)} "
            f"id={candidate.candidate_id} "
            f"source={candidate.source}",
            flush=True,
        )

        if candidate.source in {"grid", "candidate_list"}:
            if candles is None or input_csv is None:
                raise ValueError("input_csv obrigatorio para candidatos treinaveis do ML_MarketRegime_ONNX")
            package_summary = train_package_candidate(
                candidate=candidate,
                orchestrator_cfg=orchestrator_cfg,
                dataset_cache=dataset_cache,
                candles=candles,
                input_csv=input_csv,
            )
        else:
            package_summary = train_package_candidate(
                candidate=candidate,
                orchestrator_cfg=orchestrator_cfg,
                dataset_cache=dataset_cache,
                candles=pd.DataFrame(),
                input_csv=Path(""),
            )

        package_rows.append(package_summary)

        if bool(orchestrator_cfg.get("comparison", {}).get("require_candidate_package_eligible", True)) and not bool(package_summary.get("eligible", False)):
            global_rows.append(build_global_summary_row(candidate, package_summary, best_wfo_row=None, candidate_root=candidate_root))
            continue

        deployed_dir = deploy_package_to_tester(candidate, mt5_data_root=mt5_data_root)
        template_path = render_candidate_template(
            base_template_path=base_template_path,
            manifest_relative_path=candidate.manifest_relative_path,
            destination=candidate_root / "t.ini",
            tester_inputs_override=tester_inputs_override,
        )

        candidate_config = build_candidate_wfo_config(
            orchestrator_cfg=orchestrator_cfg,
            candidate=candidate,
            candidate_root=candidate_root,
            template_path=template_path,
            mt5_data_root=mt5_data_root,
        )
        candidate_config_path = candidate_root / "wfo.yaml"
        save_yaml(candidate_config_path, candidate_config)

        best_wfo_row = None
        if not dry_run:
            outputs = run_walkforward_search(candidate_config_path)
            best_wfo_row = load_best_wfo_row(outputs["summary_csv"])
            save_json(
                candidate_root / "candidate_run_manifest.json",
                {
                    "candidate_id": candidate.candidate_id,
                    "package_summary": package_summary,
                    "package_deployed_dir": str(deployed_dir),
                    "mt5_outputs": {key: str(value) for key, value in outputs.items()},
                    "best_wfo_row": best_wfo_row,
                },
            )
        else:
            save_json(
                candidate_root / "candidate_run_manifest.json",
                {
                    "candidate_id": candidate.candidate_id,
                    "package_summary": package_summary,
                    "package_deployed_dir": str(deployed_dir),
                    "dry_run": True,
                },
            )

        global_rows.append(build_global_summary_row(candidate, package_summary, best_wfo_row=best_wfo_row, candidate_root=candidate_root))

    summary_df = pd.DataFrame(global_rows)
    package_df = pd.DataFrame(package_rows)
    sort_metric = str(orchestrator_cfg.get("comparison", {}).get("sort_metric", "mt5_composite_score"))
    if not summary_df.empty and sort_metric in summary_df.columns:
        summary_df = summary_df.sort_values(by=[sort_metric], ascending=False, na_position="last")

    summary_csv = artifacts_root / "model_candidate_summary.csv"
    package_csv = artifacts_root / "package_candidate_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    package_df.to_csv(package_csv, index=False)

    best_json = artifacts_root / "best_model_candidate.json"
    if not summary_df.empty:
        save_json(best_json, json.loads(summary_df.iloc[0].to_json()))

    manifest = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dry_run": bool(dry_run),
        "config_path": str(config_path.resolve()),
        "artifacts_root": str(artifacts_root),
        "candidate_count": int(len(candidates)),
        "paths": {
            "model_candidate_summary": str(summary_csv),
            "package_candidate_summary": str(package_csv),
            "best_model_candidate": str(best_json),
        },
    }
    manifest_path = artifacts_root / "research_manifest.json"
    save_json(manifest_path, manifest)

    return {
        "artifacts_root": artifacts_root,
        "summary_csv": summary_csv,
        "package_csv": package_csv,
        "best_json": best_json,
        "manifest_json": manifest_path,
    }
