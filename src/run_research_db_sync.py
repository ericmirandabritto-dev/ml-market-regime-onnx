from __future__ import annotations

import argparse
from pathlib import Path

from research_db_core import load_sync_config, resolve_path, sync_research_database


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_cli_path(project_root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.parent.exists():
        return cwd_candidate
    return resolve_path(project_root, raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sincroniza os artefatos do ML_MarketRegime_ONNX com um banco SQLite local"
    )
    parser.add_argument(
        "--config",
        default="config.research_db.example.yaml",
        help="Arquivo YAML com caminho do banco e raizes de scan",
    )
    parser.add_argument(
        "--db",
        default="",
        help="Sobrescreve database.path do YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reingere os arquivos mesmo quando size/mtime nao mudaram",
    )
    parser.add_argument(
        "--include-oos-predictions",
        action="store_true",
        help="Inclui oos_predictions.csv dos pacotes walkforward na sincronizacao",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_cli_path(PROJECT_ROOT, args.config)
    config = load_sync_config(config_path if config_path.exists() else None)

    if args.db:
        config.setdefault("database", {})
        config["database"]["path"] = args.db
    if args.include_oos_predictions:
        config.setdefault("scan", {})
        config["scan"]["include_oos_predictions"] = True

    db_path = resolve_cli_path(PROJECT_ROOT, str(config["database"]["path"]))
    result = sync_research_database(
        project_root=PROJECT_ROOT,
        db_path=db_path,
        config=config,
        force=bool(args.force),
    )
    print(
        "[research-db] "
        f"db={result['database_path']} "
        f"files_seen={result['files_seen']} "
        f"files_ingested={result['files_ingested']} "
        f"files_skipped={result['files_skipped']} "
        f"rows_inserted={result['rows_inserted']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
