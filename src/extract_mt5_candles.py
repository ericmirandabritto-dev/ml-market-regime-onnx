from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_CSV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def _load_metatrader5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(
            "Biblioteca MetaTrader5 nao instalada. Rode: python -m pip install MetaTrader5"
        ) from exc

    return mt5


def _resolve_timeframe(mt5, timeframe: str) -> int:
    value = timeframe.strip().upper()
    if value.startswith("PERIOD_"):
        value = value.replace("PERIOD_", "", 1)
    if value.startswith("TIMEFRAME_"):
        value = value.replace("TIMEFRAME_", "", 1)

    attr_name = f"TIMEFRAME_{value}"
    if not hasattr(mt5, attr_name):
        supported = "M1, M2, M3, M4, M5, M6, M10, M12, M15, M20, M30, H1, H2, H3, H4, H6, H8, H12, D1, W1, MN1"
        raise ValueError(f"Timeframe invalido: {timeframe}. Use um destes: {supported}")

    return getattr(mt5, attr_name)


def export_mt5_candles_to_csv(
    symbol: str = "US100",
    timeframe: str = "M5",
    bars: int = 50000,
    output: str | Path = "data/raw/us100_m5_candles.csv",
    terminal_path: Optional[str] = None,
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    skip_current_bar: bool = True,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    append: bool = True,
) -> Path:
    """Exporta candles do MetaTrader 5 para CSV no formato de treino.

    O CSV gerado contem: time, open, high, low, close, volume.
    A coluna volume usa tick_volume do MT5.
    """

    if bars <= 0:
        raise ValueError("bars deve ser maior que zero")

    mt5 = _load_metatrader5()

    initialize_kwargs = {}
    if terminal_path:
        initialize_kwargs["path"] = terminal_path
    if login is not None:
        initialize_kwargs["login"] = int(login)
    if password:
        initialize_kwargs["password"] = password
    if server:
        initialize_kwargs["server"] = server

    if not mt5.initialize(**initialize_kwargs):
        error_code, error_message = mt5.last_error()
        raise RuntimeError(f"Falha ao inicializar MetaTrader5: {error_code} {error_message}")

    try:
        if not mt5.symbol_select(symbol, True):
            error_code, error_message = mt5.last_error()
            raise RuntimeError(f"Falha ao selecionar simbolo {symbol}: {error_code} {error_message}")

        timeframe_value = _resolve_timeframe(mt5, timeframe)

        if date_from or date_to:
            start = _parse_datetime(date_from, "date_from")
            end = _parse_datetime(date_to, "date_to") if date_to else datetime.now()
            if start >= end:
                raise ValueError("date_from deve ser anterior a date_to")

            rates = mt5.copy_rates_range(symbol, timeframe_value, start, end)
        else:
            start_pos = 1 if skip_current_bar else 0
            rates = mt5.copy_rates_from_pos(symbol, timeframe_value, start_pos, int(bars))

        if rates is None or len(rates) == 0:
            error_code, error_message = mt5.last_error()
            hint = (
                "Verifique se o simbolo esta correto no Market Watch, se o terminal esta logado "
                "e se ha historico carregado para esse timeframe."
            )
            raise RuntimeError(f"Nenhum candle retornado para {symbol} {timeframe}: {error_code} {error_message}. {hint}")

        df = pd.DataFrame(rates)
        required = ["time", "open", "high", "low", "close", "tick_volume"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise RuntimeError(f"Retorno do MT5 sem colunas esperadas: {missing}")

        candles = _rates_to_candles_frame(df[required].copy())

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if append:
            existing = load_existing_candles(output_path)
            before_rows = len(existing)
            candles = merge_candles(existing, candles)
            added_rows = len(candles) - before_rows
            duplicates_removed = before_rows + len(df) - len(candles)
        else:
            candles = normalize_candles_frame(candles)
            added_rows = len(candles)
            duplicates_removed = len(df) - len(candles)

        write_candles_csv(candles, output_path)

        print(
            "[mt5-export] "
            f"fetched={len(df)} "
            f"saved={len(candles)} "
            f"added={added_rows} "
            f"duplicates_removed={max(0, duplicates_removed)} "
            f"start={candles['time'].min() if len(candles) else 'NA'} "
            f"end={candles['time'].max() if len(candles) else 'NA'}"
        )

        return output_path
    finally:
        mt5.shutdown()


def _rates_to_candles_frame(df: pd.DataFrame) -> pd.DataFrame:
    candles = df.copy()
    candles["time"] = pd.to_datetime(candles["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    candles = candles.rename(columns={"tick_volume": "volume"})
    candles = candles[REQUIRED_CSV_COLUMNS]
    return normalize_candles_frame(candles)


def load_existing_candles(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Um arquivo em branco pode restar de uma execucao antiga interrompida.
        # Nesse caso, a proxima extracao deve reconstruir a base normalmente.
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS)
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV existente sem colunas obrigatorias: {missing}")

    return normalize_candles_frame(df[REQUIRED_CSV_COLUMNS])


def write_candles_csv(candles: pd.DataFrame, path: Path) -> None:
    """Grava o CSV de forma atomica para nao deixar um arquivo truncado."""
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            candles.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def normalize_candles_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS)

    candles = df.copy()
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce")
    candles = candles.dropna(subset=["time"])

    for column in ["open", "high", "low", "close", "volume"]:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")

    candles = candles.dropna(subset=["open", "high", "low", "close", "volume"])
    candles["time"] = candles["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    candles = candles[REQUIRED_CSV_COLUMNS]
    candles = candles.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return candles.reset_index(drop=True)


def merge_candles(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return normalize_candles_frame(incoming)
    if incoming.empty:
        return normalize_candles_frame(existing)

    merged = pd.concat([existing, incoming], ignore_index=True)
    return normalize_candles_frame(merged)


def _parse_datetime(value: Optional[str], field_name: str) -> datetime:
    if not value:
        raise ValueError(f"{field_name} e obrigatorio quando usar extracao por periodo")

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} invalido: {value}. Use formato YYYY-MM-DD ou YYYY-MM-DD HH:MM:SS")

    return parsed.to_pydatetime()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exporta candles do MetaTrader 5 para CSV de treino")
    parser.add_argument("--symbol", default="US100", help="Simbolo no MT5")
    parser.add_argument("--timeframe", default="M5", help="Timeframe: M1, M5, M15, H1, D1 etc.")
    parser.add_argument("--bars", type=int, default=50000, help="Quantidade de candles")
    parser.add_argument("--output", default="data/raw/us100_m5_candles.csv", help="CSV de saida")
    parser.add_argument("--terminal-path", default=None, help="Caminho opcional do terminal64.exe")
    parser.add_argument("--login", type=int, default=None, help="Login opcional da conta")
    parser.add_argument("--password", default=None, help="Senha opcional da conta")
    parser.add_argument("--server", default=None, help="Servidor opcional da corretora")
    parser.add_argument("--include-current-bar", action="store_true", help="Inclui candle atual ainda em formacao")
    parser.add_argument("--date-from", default=None, help="Inicio opcional do periodo: YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="Fim opcional do periodo: YYYY-MM-DD")
    parser.add_argument("--append", dest="append", action="store_true", default=True, help="Acumula no CSV existente")
    parser.add_argument("--overwrite", dest="append", action="store_false", help="Sobrescreve o CSV de saida")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = export_mt5_candles_to_csv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        output=args.output,
        terminal_path=args.terminal_path,
        login=args.login,
        password=args.password,
        server=args.server,
        skip_current_bar=not args.include_current_bar,
        date_from=args.date_from,
        date_to=args.date_to,
        append=args.append,
    )
    print(f"[mt5-export] output={output_path}")


if __name__ == "__main__":
    main()
