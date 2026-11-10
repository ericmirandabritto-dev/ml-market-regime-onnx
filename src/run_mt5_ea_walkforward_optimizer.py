from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orquestra treinamento de candidatos do ML_MarketRegime_ONNX e otimiza o EA no MT5 Strategy Tester"
    )
    parser.add_argument("--config", required=True, help="Arquivo YAML de configuracao")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gera pacotes, publica no tester e monta configs sem rodar o Strategy Tester",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        from mt5_ea_walkforward_optimizer_core import run_end_to_end_mt5_walkforward
    except ModuleNotFoundError as exc:
        if exc.name != "mt5_walkforward_core":
            raise
        parser.error("O Strategy Tester requer o projeto irmao MT5_WalkForward_Window_Optimizer.")
    outputs = run_end_to_end_mt5_walkforward(Path(args.config), dry_run=bool(args.dry_run))
    print(
        "[ea-mt5-wfo] "
        f"summary={outputs['summary_csv']} "
        f"packages={outputs['package_csv']} "
        f"best={outputs['best_json']} "
        f"manifest={outputs['manifest_json']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
