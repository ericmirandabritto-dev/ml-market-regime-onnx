from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from market_regime_core import load_candles_csv, load_config, save_json
from position_management_core import build_position_management_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera datasets bronze/silver/gold para gestao de posicoes abertas")
    parser.add_argument("--candles", required=True, help="CSV com candles OHLCV")
    parser.add_argument("--report", required=True, help="Relatorio XLSX do Strategy Tester")
    parser.add_argument("--config", required=True, help="Arquivo YAML com features, snapshots e simulacao")
    parser.add_argument("--output-root", default="", help="Diretorio raiz dos datasets gerados")
    parser.add_argument("--output-format", choices=["auto", "csv", "parquet"], default="", help="Formato de saida")
    parser.add_argument("--symbol", default="", help="Override do simbolo")
    parser.add_argument("--timeframe", default="", help="Override do timeframe")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    candles_path = Path(args.candles)
    report_path = Path(args.report)

    cfg = load_config(config_path)
    dataset_cfg = cfg.get("dataset", {})

    symbol = str(args.symbol or dataset_cfg.get("symbol", "US100"))
    timeframe = str(args.timeframe or dataset_cfg.get("timeframe", "M5"))
    output_root = Path(args.output_root or dataset_cfg.get("output_root", "datasets/position_management"))
    output_format = str(args.output_format or dataset_cfg.get("output_format", "auto"))

    print("[dataset] stage=load_candles", flush=True)
    candles = load_candles_csv(candles_path)

    print("[dataset] stage=build_all", flush=True)
    manifest = build_position_management_dataset(
        candles=candles,
        report_path=report_path,
        output_root=output_root,
        symbol=symbol,
        timeframe=timeframe,
        cfg=cfg,
        output_format=output_format,
    )
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["config_path"] = str(config_path)
    manifest["candles_path"] = str(candles_path)

    save_json(output_root / "dataset_manifest.json", manifest)

    counts = manifest["counts"]
    print(
        "[dataset] "
        f"symbol={symbol} "
        f"timeframe={timeframe} "
        f"candles={counts['candles']} "
        f"trades={counts['trade_legs']} "
        f"snapshots={counts['snapshots']} "
        f"action_outcomes={counts['action_outcomes']} "
        f"training_samples={counts['training_samples']} "
        f"output_root={output_root} "
        f"format={manifest['output_format']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
