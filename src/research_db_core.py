from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pandas as pd
import yaml


DEFAULT_CONFIG = {
    "database": {
        "path": "db/research.sqlite",
    },
    "scan": {
        "reports_root": "reports",
        "walkforward_package_globs": [
            "artifacts*",
            "a/ea_pkg*",
        ],
        "ea_run_globs": [
            "a/ea_wfo*",
        ],
        "include_oos_predictions": False,
        "include_json_documents": True,
    },
}

SWEEP_SUFFIXES = [
    (".top_mean_balanced_accuracy_trade.csv", "top_mean_balanced_accuracy_trade"),
    (".top_mean_accuracy_trade.csv", "top_mean_accuracy_trade"),
    (".top_score.csv", "top_score"),
    (".best_summary.csv", "best_summary"),
    (".csv", "main"),
]

WALKFORWARD_JSON_FILENAMES = {
    "walkforward_report.json": "walkforward_report",
    "manifest.json": "walkforward_runtime_manifest",
    "research_manifest.json": "walkforward_research_manifest",
}

EA_CANDIDATE_JSON_FILENAMES = {
    "best_candidate.json": "ea_candidate_best",
    "research_manifest.json": "ea_candidate_research_manifest",
    "candidate_run_manifest.json": "ea_candidate_run_manifest",
}


@dataclass
class SyncStats:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    rows_inserted: int = 0

    def add(self, other: "SyncStats") -> None:
        self.files_seen += other.files_seen
        self.files_ingested += other.files_ingested
        self.files_skipped += other.files_skipped
        self.rows_inserted += other.rows_inserted


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuracao invalida em {path}")
    return payload


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_sync_config(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    payload = load_yaml(config_path)
    return merge_dicts(DEFAULT_CONFIG, payload)


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect_database(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS artifact_files (
            source_path TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            root_path TEXT,
            variant TEXT,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            row_count INTEGER,
            ingested_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS json_documents (
            source_path TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            root_path TEXT,
            payload_json TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sweep_rows (
            source_path TEXT NOT NULL,
            sweep_name TEXT NOT NULL,
            report_variant TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            evaluation_mode TEXT,
            objective TEXT,
            model_type TEXT,
            horizon_bars REAL,
            trend_threshold_atr REAL,
            train_window REAL,
            validation_window REAL,
            trade_window REAL,
            step_window REAL,
            min_class_count_per_train REAL,
            max_folds REAL,
            score REAL,
            mean_accuracy_trade REAL,
            mean_balanced_accuracy_trade REAL,
            mean_accuracy_validation REAL,
            mean_balanced_accuracy_validation REAL,
            eligible INTEGER,
            temporal_valid INTEGER,
            parameter_signature TEXT,
            error TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS walkforward_packages (
            root_path TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            package_name TEXT NOT NULL,
            objective TEXT,
            available_folds INTEGER,
            planned_folds INTEGER,
            valid_folds INTEGER,
            coverage REAL,
            eligible INTEGER,
            train_window INTEGER,
            validation_window INTEGER,
            trade_window INTEGER,
            step_window INTEGER,
            purge_bars INTEGER,
            min_folds INTEGER,
            max_folds INTEGER,
            min_class_count_per_train INTEGER,
            score REAL,
            mean_accuracy_trade REAL,
            mean_balanced_accuracy_trade REAL,
            std_balanced_accuracy_trade REAL,
            mean_accuracy_validation REAL,
            mean_balanced_accuracy_validation REAL,
            trade_rows_total INTEGER,
            reused_folds INTEGER,
            trained_folds INTEGER,
            skipped_folds INTEGER,
            live_tail_available INTEGER,
            live_tail_reused INTEGER,
            live_tail_trained_until TEXT,
            payload_json TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS walkforward_fold_rows (
            source_path TEXT NOT NULL,
            root_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            fold_id TEXT NOT NULL,
            status TEXT,
            skip_reason TEXT,
            train_start TEXT,
            train_end TEXT,
            trained_until TEXT,
            validation_start TEXT,
            validation_end TEXT,
            trade_start TEXT,
            trade_end TEXT,
            active_from TEXT,
            active_to TEXT,
            train_rows INTEGER,
            validation_rows INTEGER,
            trade_rows INTEGER,
            validation_accuracy REAL,
            validation_balanced_accuracy REAL,
            trade_accuracy REAL,
            trade_balanced_accuracy REAL,
            train_class_distribution_json TEXT,
            validation_class_distribution_json TEXT,
            trade_class_distribution_json TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS walkforward_manifest_rows (
            source_path TEXT NOT NULL,
            root_path TEXT NOT NULL,
            manifest_kind TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            fold_id TEXT NOT NULL,
            active_from TEXT,
            active_to TEXT,
            trained_until TEXT,
            model_path TEXT,
            metadata_path TEXT,
            train_rows INTEGER,
            validation_rows INTEGER,
            trade_rows INTEGER,
            symbol TEXT,
            timeframe TEXT,
            feature_schema_version INTEGER,
            status TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS walkforward_oos_predictions (
            source_path TEXT NOT NULL,
            root_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            prediction_time TEXT,
            fold_id TEXT,
            active_from TEXT,
            active_to TEXT,
            trained_until TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            regime_id INTEGER,
            regime_name TEXT,
            predicted_regime_id INTEGER,
            predicted_regime_name TEXT,
            prob_bear REAL,
            prob_range REAL,
            prob_bull REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_wfo_runs (
            root_path TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            generated_at_utc TEXT,
            dry_run INTEGER,
            config_path TEXT,
            artifacts_root TEXT,
            candidate_count INTEGER,
            payload_json TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ea_model_candidate_rows (
            source_path TEXT NOT NULL,
            root_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            candidate_id TEXT,
            candidate_source TEXT,
            candidate_path_id TEXT,
            package_status TEXT,
            package_eligible INTEGER,
            package_dir TEXT,
            runtime_root TEXT,
            manifest_relative_path TEXT,
            candidate_artifact_root TEXT,
            mt5_status TEXT,
            mt5_candidate_id TEXT,
            mt5_window_mode TEXT,
            mt5_train_days REAL,
            mt5_forward_days REAL,
            mt5_step_days REAL,
            mt5_segments REAL,
            mt5_eligible INTEGER,
            mt5_composite_score REAL,
            cfg_values_json TEXT,
            mt5_values_json TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_package_candidate_rows (
            source_path TEXT NOT NULL,
            root_path TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            candidate_id TEXT,
            status TEXT,
            eligible INTEGER,
            package_dir TEXT,
            runtime_root TEXT,
            manifest_relative_path TEXT,
            planned_folds INTEGER,
            valid_folds INTEGER,
            score REAL,
            mean_balanced_accuracy_trade REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_candidate_window_rows (
            source_path TEXT NOT NULL,
            run_root TEXT NOT NULL,
            candidate_root TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            candidate_id TEXT,
            window_mode TEXT,
            train_days REAL,
            forward_days REAL,
            step_days REAL,
            segments REAL,
            planned_segments REAL,
            error_segments REAL,
            segments_with_trades REAL,
            segments_meeting_min_trades REAL,
            eligible INTEGER,
            pct_profitable_segments REAL,
            mean_forward_profit REAL,
            median_forward_profit REAL,
            mean_forward_result REAL,
            mean_forward_profit_factor REAL,
            mean_forward_trades REAL,
            mean_forward_equity_dd_pct REAL,
            mean_wfe_profit REAL,
            mean_wfe_result REAL,
            total_forward_profit REAL,
            composite_score REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_segment_rows (
            source_path TEXT NOT NULL,
            run_root TEXT NOT NULL,
            candidate_root TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            candidate_id TEXT,
            window_mode TEXT,
            train_days REAL,
            forward_days REAL,
            step_days REAL,
            segment_index REAL,
            train_start TEXT,
            train_end TEXT,
            forward_start TEXT,
            forward_end TEXT,
            status TEXT,
            error TEXT,
            selected_pass REAL,
            back_result REAL,
            back_profit REAL,
            back_trades REAL,
            forward_result REAL,
            forward_profit REAL,
            forward_trades REAL,
            walkforward_efficiency_profit REAL,
            walkforward_efficiency_result REAL,
            parameter_values_json TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_pass_rows (
            source_path TEXT NOT NULL,
            run_root TEXT NOT NULL,
            candidate_root TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            candidate_id TEXT,
            window_mode TEXT,
            train_days REAL,
            forward_days REAL,
            step_days REAL,
            segment_index REAL,
            pass_id REAL,
            is_selected_pass INTEGER,
            selection_rule TEXT,
            back_result REAL,
            back_profit REAL,
            back_trades REAL,
            forward_result REAL,
            forward_profit REAL,
            forward_trades REAL,
            walkforward_efficiency_profit REAL,
            walkforward_efficiency_result REAL,
            parameter_values_json TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE TABLE IF NOT EXISTS ea_parameter_profile_rows (
            source_path TEXT NOT NULL,
            run_root TEXT NOT NULL,
            candidate_root TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            passes REAL,
            segments REAL,
            selected_count REAL,
            pct_profitable REAL,
            mean_forward_result REAL,
            mean_forward_profit REAL,
            mean_back_result REAL,
            mean_back_profit REAL,
            mean_wfe_result REAL,
            mean_wfe_profit REAL,
            profile_score REAL,
            parameter_values_json TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (source_path, row_index)
        );

        CREATE INDEX IF NOT EXISTS idx_sweep_rows_signature ON sweep_rows(parameter_signature);
        CREATE INDEX IF NOT EXISTS idx_walkforward_packages_score ON walkforward_packages(score DESC);
        CREATE INDEX IF NOT EXISTS idx_walkforward_fold_rows_root ON walkforward_fold_rows(root_path, fold_id);
        CREATE INDEX IF NOT EXISTS idx_walkforward_manifest_rows_root ON walkforward_manifest_rows(root_path, fold_id);
        CREATE INDEX IF NOT EXISTS idx_walkforward_oos_root_time ON walkforward_oos_predictions(root_path, prediction_time);
        CREATE INDEX IF NOT EXISTS idx_ea_model_candidates_score ON ea_model_candidate_rows(mt5_composite_score DESC);
        CREATE INDEX IF NOT EXISTS idx_ea_candidate_windows_score ON ea_candidate_window_rows(composite_score DESC);
        CREATE INDEX IF NOT EXISTS idx_ea_segment_rows_candidate ON ea_segment_rows(candidate_root, segment_index);
        CREATE INDEX IF NOT EXISTS idx_ea_pass_rows_candidate ON ea_pass_rows(candidate_root, segment_index, pass_id);

        CREATE VIEW IF NOT EXISTS vw_best_sweep_rows AS
        SELECT *
        FROM sweep_rows
        WHERE report_variant = 'main' AND COALESCE(eligible, 0) = 1
        ORDER BY score DESC;

        CREATE VIEW IF NOT EXISTS vw_best_walkforward_packages AS
        SELECT *
        FROM walkforward_packages
        WHERE COALESCE(eligible, 0) = 1
        ORDER BY score DESC;

        CREATE VIEW IF NOT EXISTS vw_best_ea_candidates AS
        SELECT *
        FROM ea_model_candidate_rows
        WHERE mt5_status = 'ok' AND COALESCE(mt5_eligible, 0) = 1
        ORDER BY mt5_composite_score DESC;
        """
    )
    connection.commit()


def normalize_path(path: Path) -> str:
    return str(path.resolve())


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): clean_scalar(value) for key, value in record.items()}


def bool_to_int(value: Any) -> int | None:
    cleaned = clean_scalar(value)
    if cleaned is None or cleaned == "":
        return None
    if isinstance(cleaned, bool):
        return 1 if cleaned else 0
    text = str(cleaned).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    return None


def to_float(value: Any) -> float | None:
    cleaned = clean_scalar(value)
    if cleaned is None or cleaned == "":
        return None
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def row_json(record: Dict[str, Any]) -> str:
    return json.dumps(clean_record(record), ensure_ascii=True, sort_keys=True)


def extract_prefixed(record: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith(prefix):
            payload[key] = clean_scalar(value)
    return payload


def scan_pattern_paths(project_root: Path, patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.exists():
                found.append(path.resolve())
    unique_paths = sorted({normalize_path(path): path for path in found}.values(), key=lambda item: str(item).lower())
    return unique_paths


def detect_sweep_variant(path: Path) -> tuple[str, str]:
    name = path.name
    for suffix, variant in SWEEP_SUFFIXES:
        if name.endswith(suffix):
            base_name = name[: -len(suffix)] if suffix != ".csv" else path.stem
            return base_name, variant
    return path.stem, "unknown"


def file_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def should_skip_file(connection: sqlite3.Connection, path: Path, category: str, force: bool = False) -> bool:
    if force:
        return False
    size_bytes, mtime_ns = file_state(path)
    row = connection.execute(
        "SELECT size_bytes, mtime_ns FROM artifact_files WHERE source_path = ? AND category = ?",
        (normalize_path(path), category),
    ).fetchone()
    return row is not None and int(row[0]) == size_bytes and int(row[1]) == mtime_ns


def register_file(
    connection: sqlite3.Connection,
    path: Path,
    category: str,
    root_path: str | None,
    variant: str | None,
    row_count: int | None,
) -> None:
    size_bytes, mtime_ns = file_state(path)
    connection.execute(
        """
        INSERT INTO artifact_files (
            source_path, category, root_path, variant, size_bytes, mtime_ns, row_count, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            category = excluded.category,
            root_path = excluded.root_path,
            variant = excluded.variant,
            size_bytes = excluded.size_bytes,
            mtime_ns = excluded.mtime_ns,
            row_count = excluded.row_count,
            ingested_at_utc = excluded.ingested_at_utc
        """,
        (
            normalize_path(path),
            category,
            root_path,
            variant,
            size_bytes,
            mtime_ns,
            row_count,
            utc_now_iso(),
        ),
    )


def upsert_json_document(
    connection: sqlite3.Connection,
    path: Path,
    category: str,
    root_path: str | None,
    payload: Dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO json_documents (source_path, category, root_path, payload_json, ingested_at_utc)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            category = excluded.category,
            root_path = excluded.root_path,
            payload_json = excluded.payload_json,
            ingested_at_utc = excluded.ingested_at_utc
        """,
        (
            normalize_path(path),
            category,
            root_path,
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            utc_now_iso(),
        ),
    )


def delete_existing_rows(connection: sqlite3.Connection, table_name: str, source_path: Path) -> None:
    connection.execute(f"DELETE FROM {table_name} WHERE source_path = ?", (normalize_path(source_path),))


def chunked(rows: Sequence[tuple], size: int = 5000) -> Iterator[Sequence[tuple]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def ingest_sweep_report(connection: sqlite3.Connection, path: Path, force: bool = False) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "sweep_report"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    sweep_name, variant = detect_sweep_variant(path)
    frame = pd.read_csv(path)
    delete_existing_rows(connection, "sweep_rows", path)

    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                sweep_name,
                variant,
                row_index,
                cleaned.get("evaluation_mode"),
                cleaned.get("objective"),
                cleaned.get("model_type"),
                to_float(cleaned.get("horizon_bars")),
                to_float(cleaned.get("trend_threshold_atr")),
                to_float(cleaned.get("train_window")),
                to_float(cleaned.get("validation_window")),
                to_float(cleaned.get("trade_window")),
                to_float(cleaned.get("step_window")),
                to_float(cleaned.get("min_class_count_per_train")),
                to_float(cleaned.get("max_folds")),
                to_float(cleaned.get("score")),
                to_float(cleaned.get("mean_accuracy_trade")),
                to_float(cleaned.get("mean_balanced_accuracy_trade")),
                to_float(cleaned.get("mean_accuracy_validation")),
                to_float(cleaned.get("mean_balanced_accuracy_validation")),
                bool_to_int(cleaned.get("eligible")),
                bool_to_int(cleaned.get("temporal_valid")),
                cleaned.get("parameter_signature"),
                cleaned.get("error"),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO sweep_rows (
            source_path, sweep_name, report_variant, row_index, evaluation_mode, objective, model_type,
            horizon_bars, trend_threshold_atr, train_window, validation_window, trade_window, step_window,
            min_class_count_per_train, max_folds, score, mean_accuracy_trade, mean_balanced_accuracy_trade,
            mean_accuracy_validation, mean_balanced_accuracy_validation, eligible, temporal_valid,
            parameter_signature, error, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=None, variant=variant, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_walkforward_package_json(
    connection: sqlite3.Connection,
    root_path: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = WALKFORWARD_JSON_FILENAMES.get(path.name, "walkforward_json")
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    payload = load_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
    upsert_json_document(connection, path, category=category, root_path=normalize_path(root_path), payload=payload)

    if path.name == "walkforward_report.json":
        connection.execute(
            "DELETE FROM walkforward_packages WHERE root_path = ?",
            (normalize_path(root_path),),
        )
        connection.execute(
            """
            INSERT INTO walkforward_packages (
                root_path, source_path, package_name, objective, available_folds, planned_folds, valid_folds,
                coverage, eligible, train_window, validation_window, trade_window, step_window, purge_bars,
                min_folds, max_folds, min_class_count_per_train, score, mean_accuracy_trade,
                mean_balanced_accuracy_trade, std_balanced_accuracy_trade, mean_accuracy_validation,
                mean_balanced_accuracy_validation, trade_rows_total, reused_folds, trained_folds,
                skipped_folds, live_tail_available, live_tail_reused, live_tail_trained_until,
                payload_json, ingested_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_path(root_path),
                normalize_path(path),
                root_path.name,
                payload.get("objective"),
                to_int(payload.get("available_folds")),
                to_int(payload.get("planned_folds")),
                to_int(payload.get("valid_folds")),
                to_float(payload.get("coverage")),
                bool_to_int(payload.get("eligible")),
                to_int(payload.get("train_window")),
                to_int(payload.get("validation_window")),
                to_int(payload.get("trade_window")),
                to_int(payload.get("step_window")),
                to_int(payload.get("purge_bars")),
                to_int(payload.get("min_folds")),
                to_int(payload.get("max_folds")),
                to_int(payload.get("min_class_count_per_train")),
                to_float(payload.get("score")),
                to_float(payload.get("mean_accuracy_trade")),
                to_float(payload.get("mean_balanced_accuracy_trade")),
                to_float(payload.get("std_balanced_accuracy_trade")),
                to_float(payload.get("mean_accuracy_validation")),
                to_float(payload.get("mean_balanced_accuracy_validation")),
                to_int(payload.get("trade_rows_total")),
                to_int(payload.get("reused_folds")),
                to_int(payload.get("trained_folds")),
                to_int(payload.get("skipped_folds")),
                bool_to_int(payload.get("live_tail_available")),
                bool_to_int(payload.get("live_tail_reused")),
                payload.get("live_tail_trained_until"),
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                utc_now_iso(),
            ),
        )

    register_file(connection, path, category=category, root_path=normalize_path(root_path), variant=None, row_count=None)
    stats.files_ingested = 1
    stats.rows_inserted = 1
    return stats


def ingest_walkforward_fold_summary(
    connection: sqlite3.Connection,
    root_path: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "walkforward_fold_summary"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "walkforward_fold_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(root_path),
                row_index,
                cleaned.get("fold_id"),
                cleaned.get("status"),
                cleaned.get("skip_reason"),
                cleaned.get("train_start"),
                cleaned.get("train_end"),
                cleaned.get("trained_until"),
                cleaned.get("validation_start"),
                cleaned.get("validation_end"),
                cleaned.get("trade_start"),
                cleaned.get("trade_end"),
                cleaned.get("active_from"),
                cleaned.get("active_to"),
                to_int(cleaned.get("train_rows")),
                to_int(cleaned.get("validation_rows")),
                to_int(cleaned.get("trade_rows")),
                to_float(cleaned.get("validation_accuracy")),
                to_float(cleaned.get("validation_balanced_accuracy")),
                to_float(cleaned.get("trade_accuracy")),
                to_float(cleaned.get("trade_balanced_accuracy")),
                cleaned.get("train_class_distribution"),
                cleaned.get("validation_class_distribution"),
                cleaned.get("trade_class_distribution"),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO walkforward_fold_rows (
            source_path, root_path, row_index, fold_id, status, skip_reason, train_start, train_end, trained_until,
            validation_start, validation_end, trade_start, trade_end, active_from, active_to, train_rows,
            validation_rows, trade_rows, validation_accuracy, validation_balanced_accuracy, trade_accuracy,
            trade_balanced_accuracy, train_class_distribution_json, validation_class_distribution_json,
            trade_class_distribution_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(root_path), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_walkforward_manifest_csv(
    connection: sqlite3.Connection,
    root_path: Path,
    path: Path,
    manifest_kind: str,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = f"walkforward_{manifest_kind}_manifest"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "walkforward_manifest_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(root_path),
                manifest_kind,
                row_index,
                cleaned.get("fold_id"),
                cleaned.get("active_from"),
                cleaned.get("active_to"),
                cleaned.get("trained_until"),
                cleaned.get("model_path"),
                cleaned.get("metadata_path"),
                to_int(cleaned.get("train_rows")),
                to_int(cleaned.get("validation_rows")),
                to_int(cleaned.get("trade_rows")),
                cleaned.get("symbol"),
                cleaned.get("timeframe"),
                to_int(cleaned.get("feature_schema_version")),
                cleaned.get("status"),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO walkforward_manifest_rows (
            source_path, root_path, manifest_kind, row_index, fold_id, active_from, active_to, trained_until,
            model_path, metadata_path, train_rows, validation_rows, trade_rows, symbol, timeframe,
            feature_schema_version, status, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(root_path), variant=manifest_kind, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_walkforward_oos_predictions(
    connection: sqlite3.Connection,
    root_path: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "walkforward_oos_predictions"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "walkforward_oos_predictions", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(root_path),
                row_index,
                cleaned.get("time"),
                cleaned.get("fold_id"),
                cleaned.get("active_from"),
                cleaned.get("active_to"),
                cleaned.get("trained_until"),
                to_float(cleaned.get("open")),
                to_float(cleaned.get("high")),
                to_float(cleaned.get("low")),
                to_float(cleaned.get("close")),
                to_float(cleaned.get("volume")),
                to_int(cleaned.get("regime_id")),
                cleaned.get("regime_name"),
                to_int(cleaned.get("predicted_regime_id")),
                cleaned.get("predicted_regime_name"),
                to_float(cleaned.get("prob_bear")),
                to_float(cleaned.get("prob_range")),
                to_float(cleaned.get("prob_bull")),
                row_json(cleaned),
            )
        )

    for chunk in chunked(rows):
        connection.executemany(
            """
            INSERT INTO walkforward_oos_predictions (
                source_path, root_path, row_index, prediction_time, fold_id, active_from, active_to, trained_until,
                open, high, low, close, volume, regime_id, regime_name, predicted_regime_id, predicted_regime_name,
                prob_bear, prob_range, prob_bull, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk,
        )

    register_file(connection, path, category=category, root_path=normalize_path(root_path), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def discover_walkforward_package_roots(project_root: Path, patterns: Sequence[str]) -> List[Path]:
    roots = scan_pattern_paths(project_root, patterns)
    packages: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root.is_file():
            continue
        for json_path in root.rglob("walkforward_report.json"):
            package_root = json_path.parent.resolve()
            normalized = normalize_path(package_root)
            if normalized in seen:
                continue
            seen.add(normalized)
            packages.append(package_root)
    return sorted(packages, key=lambda item: str(item).lower())


def ingest_walkforward_package(
    connection: sqlite3.Connection,
    root_path: Path,
    include_oos_predictions: bool,
    include_json_documents: bool,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats()

    if include_json_documents:
        for filename in WALKFORWARD_JSON_FILENAMES:
            path = root_path / filename
            if path.exists():
                stats.add(ingest_walkforward_package_json(connection, root_path, path, force=force))

    summary_path = root_path / "summary.csv"
    if summary_path.exists():
        stats.add(ingest_walkforward_fold_summary(connection, root_path, summary_path, force=force))

    manifest_path = root_path / "manifest.csv"
    if manifest_path.exists():
        stats.add(ingest_walkforward_manifest_csv(connection, root_path, manifest_path, manifest_kind="runtime", force=force))

    research_manifest_path = root_path / "research_manifest.csv"
    if research_manifest_path.exists():
        stats.add(
            ingest_walkforward_manifest_csv(
                connection,
                root_path,
                research_manifest_path,
                manifest_kind="research",
                force=force,
            )
        )

    oos_path = root_path / "oos_predictions.csv"
    if include_oos_predictions and oos_path.exists():
        stats.add(ingest_walkforward_oos_predictions(connection, root_path, oos_path, force=force))

    return stats


def ingest_ea_run_json(
    connection: sqlite3.Connection,
    root_path: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "ea_run_manifest"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    payload = json.loads(path.read_text(encoding="utf-8"))
    upsert_json_document(connection, path, category=category, root_path=normalize_path(root_path), payload=payload)

    connection.execute("DELETE FROM ea_wfo_runs WHERE root_path = ?", (normalize_path(root_path),))
    connection.execute(
        """
        INSERT INTO ea_wfo_runs (
            root_path, source_path, generated_at_utc, dry_run, config_path, artifacts_root, candidate_count,
            payload_json, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalize_path(root_path),
            normalize_path(path),
            payload.get("generated_at_utc"),
            bool_to_int(payload.get("dry_run")),
            payload.get("config_path"),
            payload.get("artifacts_root"),
            to_int(payload.get("candidate_count")),
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            utc_now_iso(),
        ),
    )

    register_file(connection, path, category=category, root_path=normalize_path(root_path), variant=None, row_count=1)
    stats.files_ingested = 1
    stats.rows_inserted = 1
    return stats


def ingest_ea_model_candidate_summary(
    connection: sqlite3.Connection,
    run_root: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "ea_model_candidate_summary"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "ea_model_candidate_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(run_root),
                row_index,
                cleaned.get("candidate_id"),
                cleaned.get("candidate_source"),
                cleaned.get("candidate_path_id"),
                cleaned.get("package_status"),
                bool_to_int(cleaned.get("package_eligible")),
                cleaned.get("package_dir"),
                cleaned.get("runtime_root"),
                cleaned.get("manifest_relative_path"),
                cleaned.get("candidate_artifact_root"),
                cleaned.get("mt5_status"),
                cleaned.get("mt5_candidate_id"),
                cleaned.get("mt5_window_mode"),
                to_float(cleaned.get("mt5_train_days")),
                to_float(cleaned.get("mt5_forward_days")),
                to_float(cleaned.get("mt5_step_days")),
                to_float(cleaned.get("mt5_segments")),
                bool_to_int(cleaned.get("mt5_eligible")),
                to_float(cleaned.get("mt5_composite_score")),
                json.dumps(extract_prefixed(cleaned, "cfg_"), ensure_ascii=True, sort_keys=True),
                json.dumps(extract_prefixed(cleaned, "mt5_"), ensure_ascii=True, sort_keys=True),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO ea_model_candidate_rows (
            source_path, root_path, row_index, candidate_id, candidate_source, candidate_path_id, package_status,
            package_eligible, package_dir, runtime_root, manifest_relative_path, candidate_artifact_root, mt5_status,
            mt5_candidate_id, mt5_window_mode, mt5_train_days, mt5_forward_days, mt5_step_days, mt5_segments,
            mt5_eligible, mt5_composite_score, cfg_values_json, mt5_values_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(run_root), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_ea_package_candidate_summary(
    connection: sqlite3.Connection,
    run_root: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "ea_package_candidate_summary"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "ea_package_candidate_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(run_root),
                row_index,
                cleaned.get("candidate_id"),
                cleaned.get("status"),
                bool_to_int(cleaned.get("eligible")),
                cleaned.get("package_dir"),
                cleaned.get("runtime_root"),
                cleaned.get("manifest_relative_path"),
                to_int(cleaned.get("planned_folds")),
                to_int(cleaned.get("valid_folds")),
                to_float(cleaned.get("score")),
                to_float(cleaned.get("mean_balanced_accuracy_trade")),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO ea_package_candidate_rows (
            source_path, root_path, row_index, candidate_id, status, eligible, package_dir, runtime_root,
            manifest_relative_path, planned_folds, valid_folds, score, mean_balanced_accuracy_trade, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(run_root), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_ea_candidate_window_summary(
    connection: sqlite3.Connection,
    run_root: Path,
    candidate_root: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "ea_candidate_window_summary"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "ea_candidate_window_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(run_root),
                normalize_path(candidate_root),
                row_index,
                cleaned.get("candidate_id"),
                cleaned.get("window_mode"),
                to_float(cleaned.get("train_days")),
                to_float(cleaned.get("forward_days")),
                to_float(cleaned.get("step_days")),
                to_float(cleaned.get("segments")),
                to_float(cleaned.get("planned_segments")),
                to_float(cleaned.get("error_segments")),
                to_float(cleaned.get("segments_with_trades")),
                to_float(cleaned.get("segments_meeting_min_trades")),
                bool_to_int(cleaned.get("eligible")),
                to_float(cleaned.get("pct_profitable_segments")),
                to_float(cleaned.get("mean_forward_profit")),
                to_float(cleaned.get("median_forward_profit")),
                to_float(cleaned.get("mean_forward_result")),
                to_float(cleaned.get("mean_forward_profit_factor")),
                to_float(cleaned.get("mean_forward_trades")),
                to_float(cleaned.get("mean_forward_equity_dd_pct")),
                to_float(cleaned.get("mean_wfe_profit")),
                to_float(cleaned.get("mean_wfe_result")),
                to_float(cleaned.get("total_forward_profit")),
                to_float(cleaned.get("composite_score")),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO ea_candidate_window_rows (
            source_path, run_root, candidate_root, row_index, candidate_id, window_mode, train_days, forward_days,
            step_days, segments, planned_segments, error_segments, segments_with_trades, segments_meeting_min_trades,
            eligible, pct_profitable_segments, mean_forward_profit, median_forward_profit, mean_forward_result,
            mean_forward_profit_factor, mean_forward_trades, mean_forward_equity_dd_pct, mean_wfe_profit,
            mean_wfe_result, total_forward_profit, composite_score, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(candidate_root), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_ea_segment_results(
    connection: sqlite3.Connection,
    run_root: Path,
    candidate_root: Path,
    path: Path,
    table_name: str,
    category: str,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, table_name, path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        parameter_values = json.dumps(extract_prefixed(cleaned, "param_"), ensure_ascii=True, sort_keys=True)

        if table_name == "ea_segment_rows":
            rows.append(
                (
                    normalize_path(path),
                    normalize_path(run_root),
                    normalize_path(candidate_root),
                    row_index,
                    cleaned.get("candidate_id"),
                    cleaned.get("window_mode"),
                    to_float(cleaned.get("train_days")),
                    to_float(cleaned.get("forward_days")),
                    to_float(cleaned.get("step_days")),
                    to_float(cleaned.get("segment_index")),
                    cleaned.get("train_start"),
                    cleaned.get("train_end"),
                    cleaned.get("forward_start"),
                    cleaned.get("forward_end"),
                    cleaned.get("status"),
                    cleaned.get("error"),
                    to_float(cleaned.get("selected_pass")),
                    to_float(cleaned.get("back_result")),
                    to_float(cleaned.get("back_profit")),
                    to_float(cleaned.get("back_trades")),
                    to_float(cleaned.get("forward_result")),
                    to_float(cleaned.get("forward_profit")),
                    to_float(cleaned.get("forward_trades")),
                    to_float(cleaned.get("walkforward_efficiency_profit")),
                    to_float(cleaned.get("walkforward_efficiency_result")),
                    parameter_values,
                    row_json(cleaned),
                )
            )
        else:
            rows.append(
                (
                    normalize_path(path),
                    normalize_path(run_root),
                    normalize_path(candidate_root),
                    row_index,
                    cleaned.get("candidate_id"),
                    cleaned.get("window_mode"),
                    to_float(cleaned.get("train_days")),
                    to_float(cleaned.get("forward_days")),
                    to_float(cleaned.get("step_days")),
                    to_float(cleaned.get("segment_index")),
                    to_float(cleaned.get("pass_id")),
                    bool_to_int(cleaned.get("is_selected_pass")),
                    cleaned.get("selection_rule"),
                    to_float(cleaned.get("back_result")),
                    to_float(cleaned.get("back_profit")),
                    to_float(cleaned.get("back_trades")),
                    to_float(cleaned.get("forward_result")),
                    to_float(cleaned.get("forward_profit")),
                    to_float(cleaned.get("forward_trades")),
                    to_float(cleaned.get("walkforward_efficiency_profit")),
                    to_float(cleaned.get("walkforward_efficiency_result")),
                    parameter_values,
                    row_json(cleaned),
                )
            )

    if table_name == "ea_segment_rows":
        connection.executemany(
            """
            INSERT INTO ea_segment_rows (
                source_path, run_root, candidate_root, row_index, candidate_id, window_mode, train_days,
                forward_days, step_days, segment_index, train_start, train_end, forward_start, forward_end,
                status, error, selected_pass, back_result, back_profit, back_trades, forward_result,
                forward_profit, forward_trades, walkforward_efficiency_profit, walkforward_efficiency_result,
                parameter_values_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    else:
        connection.executemany(
            """
            INSERT INTO ea_pass_rows (
                source_path, run_root, candidate_root, row_index, candidate_id, window_mode, train_days,
                forward_days, step_days, segment_index, pass_id, is_selected_pass, selection_rule, back_result,
                back_profit, back_trades, forward_result, forward_profit, forward_trades,
                walkforward_efficiency_profit, walkforward_efficiency_result, parameter_values_json, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    register_file(connection, path, category=category, root_path=normalize_path(candidate_root), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def ingest_ea_parameter_profiles(
    connection: sqlite3.Connection,
    run_root: Path,
    candidate_root: Path,
    path: Path,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats(files_seen=1)
    category = "ea_parameter_profile_summary"
    if should_skip_file(connection, path, category=category, force=force):
        stats.files_skipped = 1
        return stats

    frame = pd.read_csv(path)
    delete_existing_rows(connection, "ea_parameter_profile_rows", path)
    rows: List[tuple] = []
    for row_index, record in enumerate(frame.to_dict(orient="records"), start=1):
        cleaned = clean_record(record)
        rows.append(
            (
                normalize_path(path),
                normalize_path(run_root),
                normalize_path(candidate_root),
                row_index,
                to_float(cleaned.get("passes")),
                to_float(cleaned.get("segments")),
                to_float(cleaned.get("selected_count")),
                to_float(cleaned.get("pct_profitable")),
                to_float(cleaned.get("mean_forward_result")),
                to_float(cleaned.get("mean_forward_profit")),
                to_float(cleaned.get("mean_back_result")),
                to_float(cleaned.get("mean_back_profit")),
                to_float(cleaned.get("mean_wfe_result")),
                to_float(cleaned.get("mean_wfe_profit")),
                to_float(cleaned.get("profile_score")),
                json.dumps(extract_prefixed(cleaned, "param_"), ensure_ascii=True, sort_keys=True),
                row_json(cleaned),
            )
        )

    connection.executemany(
        """
        INSERT INTO ea_parameter_profile_rows (
            source_path, run_root, candidate_root, row_index, passes, segments, selected_count, pct_profitable,
            mean_forward_result, mean_forward_profit, mean_back_result, mean_back_profit, mean_wfe_result,
            mean_wfe_profit, profile_score, parameter_values_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    register_file(connection, path, category=category, root_path=normalize_path(candidate_root), variant=None, row_count=len(rows))
    stats.files_ingested = 1
    stats.rows_inserted = len(rows)
    return stats


def discover_ea_run_roots(project_root: Path, patterns: Sequence[str]) -> List[Path]:
    roots = scan_pattern_paths(project_root, patterns)
    runs: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        if (root / "model_candidate_summary.csv").exists() and (root / "research_manifest.json").exists():
            runs.append(root.resolve())
    return sorted(runs, key=lambda item: str(item).lower())


def ingest_ea_run(
    connection: sqlite3.Connection,
    run_root: Path,
    include_json_documents: bool,
    force: bool = False,
) -> SyncStats:
    stats = SyncStats()

    manifest_path = run_root / "research_manifest.json"
    if manifest_path.exists():
        stats.add(ingest_ea_run_json(connection, run_root, manifest_path, force=force))

    model_candidate_summary = run_root / "model_candidate_summary.csv"
    if model_candidate_summary.exists():
        stats.add(ingest_ea_model_candidate_summary(connection, run_root, model_candidate_summary, force=force))

    package_candidate_summary = run_root / "package_candidate_summary.csv"
    if package_candidate_summary.exists():
        stats.add(ingest_ea_package_candidate_summary(connection, run_root, package_candidate_summary, force=force))

    for candidate_root in sorted([path for path in run_root.iterdir() if path.is_dir()], key=lambda item: str(item).lower()):
        candidate_results_root = candidate_root / "r"
        if not candidate_results_root.exists():
            continue

        candidate_summary_path = candidate_results_root / "candidate_summary.csv"
        if candidate_summary_path.exists():
            stats.add(
                ingest_ea_candidate_window_summary(
                    connection,
                    run_root,
                    candidate_root,
                    candidate_summary_path,
                    force=force,
                )
            )

        segment_results_path = candidate_results_root / "segment_results.csv"
        if segment_results_path.exists():
            stats.add(
                ingest_ea_segment_results(
                    connection,
                    run_root,
                    candidate_root,
                    segment_results_path,
                    table_name="ea_segment_rows",
                    category="ea_segment_results",
                    force=force,
                )
            )

        pass_results_path = candidate_results_root / "pass_results.csv"
        if pass_results_path.exists():
            stats.add(
                ingest_ea_segment_results(
                    connection,
                    run_root,
                    candidate_root,
                    pass_results_path,
                    table_name="ea_pass_rows",
                    category="ea_pass_results",
                    force=force,
                )
            )

        parameter_profile_path = candidate_results_root / "parameter_profile_summary.csv"
        if parameter_profile_path.exists():
            stats.add(
                ingest_ea_parameter_profiles(
                    connection,
                    run_root,
                    candidate_root,
                    parameter_profile_path,
                    force=force,
                )
            )

        if include_json_documents:
            for filename, category in EA_CANDIDATE_JSON_FILENAMES.items():
                candidate_json_path = candidate_root / filename
                if candidate_json_path.exists():
                    if should_skip_file(connection, candidate_json_path, category=category, force=force):
                        stats.files_seen += 1
                        stats.files_skipped += 1
                        continue
                    payload = json.loads(candidate_json_path.read_text(encoding="utf-8"))
                    upsert_json_document(
                        connection,
                        candidate_json_path,
                        category=category,
                        root_path=normalize_path(candidate_root),
                        payload=payload,
                    )
                    register_file(
                        connection,
                        candidate_json_path,
                        category=category,
                        root_path=normalize_path(candidate_root),
                        variant=None,
                        row_count=1,
                    )
                    stats.files_seen += 1
                    stats.files_ingested += 1
                    stats.rows_inserted += 1

    return stats


def sync_research_database(
    project_root: Path,
    db_path: Path,
    config: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    scan_cfg = config["scan"]
    connection = connect_database(db_path)
    try:
        stats = SyncStats()

        reports_root = resolve_path(project_root, str(scan_cfg.get("reports_root", "reports")))
        if reports_root.exists():
            for report_path in sorted(reports_root.glob("*.csv"), key=lambda item: str(item).lower()):
                stats.add(ingest_sweep_report(connection, report_path, force=force))
                connection.commit()

        walkforward_patterns = [str(value) for value in scan_cfg.get("walkforward_package_globs", [])]
        for package_root in discover_walkforward_package_roots(project_root, walkforward_patterns):
            stats.add(
                ingest_walkforward_package(
                    connection,
                    package_root,
                    include_oos_predictions=bool(scan_cfg.get("include_oos_predictions", True)),
                    include_json_documents=bool(scan_cfg.get("include_json_documents", True)),
                    force=force,
                )
            )
            connection.commit()

        ea_run_patterns = [str(value) for value in scan_cfg.get("ea_run_globs", [])]
        for run_root in discover_ea_run_roots(project_root, ea_run_patterns):
            stats.add(
                ingest_ea_run(
                    connection,
                    run_root,
                    include_json_documents=bool(scan_cfg.get("include_json_documents", True)),
                    force=force,
                )
            )
            connection.commit()

        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "database_path": str(db_path),
            "files_seen": stats.files_seen,
            "files_ingested": stats.files_ingested,
            "files_skipped": stats.files_skipped,
            "rows_inserted": stats.rows_inserted,
        }
    finally:
        connection.close()
