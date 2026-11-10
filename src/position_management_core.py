from __future__ import annotations

import mimetypes
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from market_regime_core import ensure_dir

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

try:
    import fastparquet  # noqa: F401
except ImportError:
    fastparquet = None


ORDERS_COLUMNS = [
    "open_time",
    "order_id",
    "symbol",
    "type",
    "volume",
    "volume_unused",
    "requested_price",
    "sl_init",
    "tp_init",
    "fill_time",
    "fill_time_unused",
    "state",
    "comment",
    "unused_1",
    "unused_2",
]

DEALS_COLUMNS = [
    "time",
    "deal_id",
    "symbol",
    "type",
    "direction",
    "volume",
    "price",
    "order_id",
    "commission",
    "swap",
    "profit",
    "balance",
    "comment",
    "unused_1",
    "unused_2",
]


def parquet_available() -> bool:
    return pyarrow is not None or fastparquet is not None


def resolve_output_format(requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "parquet", "csv"}:
        raise ValueError("dataset.output_format deve ser auto, parquet ou csv")

    if normalized == "auto":
        return "parquet" if parquet_available() else "csv"

    if normalized == "parquet" and not parquet_available():
        raise RuntimeError(
            "Saida parquet solicitada, mas nem pyarrow nem fastparquet estao disponiveis. "
            "Instale pyarrow ou use dataset.output_format = csv."
        )

    return normalized


def save_table(df: pd.DataFrame, path_without_extension: Path, output_format: str) -> Path:
    ensure_dir(path_without_extension.parent)
    if output_format == "parquet":
        destination = path_without_extension.with_suffix(".parquet")
        df.to_parquet(destination, index=False)
        return destination

    destination = path_without_extension.with_suffix(".csv")
    df.to_csv(destination, index=False)
    return destination


def _safe_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_mt5_time(value) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y.%m.%d %H:%M:%S")


def _parse_volume_pair(value: str) -> Tuple[float, float]:
    raw = _safe_str(value).strip()
    if raw == "":
        return float("nan"), float("nan")
    if "/" not in raw:
        parsed = float(raw)
        return parsed, parsed

    left_raw, right_raw = raw.split("/", 1)
    return float(left_raw.strip()), float(right_raw.strip())


def _signed_move(side: str, from_price: float, to_price: float) -> float:
    if side == "buy":
        return float(to_price - from_price)
    if side == "sell":
        return float(from_price - to_price)
    raise ValueError(f"lado invalido: {side}")


def _tighten_stop(side: str, current_stop: float | None, candidate_stop: float | None) -> float | None:
    if candidate_stop is None or pd.isna(candidate_stop):
        return current_stop
    if current_stop is None or pd.isna(current_stop):
        return float(candidate_stop)
    if side == "buy":
        return float(max(current_stop, candidate_stop))
    return float(min(current_stop, candidate_stop))


def _is_stop_invalid_at_close(side: str, stop_price: float | None, current_close: float) -> bool:
    if stop_price is None or pd.isna(stop_price):
        return False
    if side == "buy":
        return bool(stop_price >= current_close)
    return bool(stop_price <= current_close)


def _prepare_mimetypes_for_openpyxl() -> None:
    if getattr(_prepare_mimetypes_for_openpyxl, "_done", False):
        return

    if sys.platform == "win32" and not mimetypes.inited:
        original_knownfiles = mimetypes.knownfiles
        try:
            # Avoid scanning the full Windows registry during openpyxl import.
            mimetypes.knownfiles = []
            mimetypes.init([])
        finally:
            mimetypes.knownfiles = original_knownfiles

    _prepare_mimetypes_for_openpyxl._done = True


def _load_report_workbook(report_path: Path):
    _prepare_mimetypes_for_openpyxl()
    from openpyxl import load_workbook

    return load_workbook(report_path, read_only=True, data_only=True)


def _extract_sheet_rows(report_path: Path) -> List[List[object]]:
    workbook = _load_report_workbook(report_path)
    try:
        sheet = workbook[workbook.sheetnames[0]]
        return [list(row[:15]) for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def parse_tester_report(report_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = _extract_sheet_rows(report_path)
    orders_header_idx = None
    deals_header_idx = None

    for idx, row in enumerate(rows):
        if len(row) >= 4 and row[1] == "Ordem" and row[2] == "Ativo" and row[3] == "Tipo":
            orders_header_idx = idx
        if len(row) >= 4 and row[1] == "Oferta" and row[2] == "Ativo" and row[3] == "Tipo":
            deals_header_idx = idx
            break

    if orders_header_idx is None or deals_header_idx is None:
        raise ValueError("Nao foi possivel localizar as secoes Ordens/Transacoes no relatorio do MT5.")

    orders_rows = [row for row in rows[orders_header_idx + 1 : deals_header_idx - 1] if row[0] is not None]
    deals_rows = [row for row in rows[deals_header_idx + 1 :] if row[0] is not None and row[3] != "balance"]

    orders_df = pd.DataFrame(orders_rows, columns=ORDERS_COLUMNS)
    deals_df = pd.DataFrame(deals_rows, columns=DEALS_COLUMNS)

    order_requested, order_filled = zip(*[_parse_volume_pair(value) for value in orders_df["volume"]], strict=True)
    orders_df["volume_requested"] = order_requested
    orders_df["volume_filled"] = order_filled
    orders_df["open_time"] = orders_df["open_time"].map(_parse_mt5_time)
    orders_df["fill_time"] = orders_df["fill_time"].map(_parse_mt5_time)
    orders_df["order_id"] = orders_df["order_id"].astype(int)
    orders_df["requested_price"] = orders_df["requested_price"].map(_safe_float)
    orders_df["sl_init"] = orders_df["sl_init"].map(_safe_float)
    orders_df["tp_init"] = orders_df["tp_init"].map(_safe_float)
    orders_df["type"] = orders_df["type"].map(lambda value: _safe_str(value).strip().lower())
    orders_df["comment"] = orders_df["comment"].map(_safe_str)
    orders_df["state"] = orders_df["state"].map(_safe_str)

    deals_df["time"] = deals_df["time"].map(_parse_mt5_time)
    deals_df["deal_id"] = deals_df["deal_id"].astype(int)
    deals_df["order_id"] = deals_df["order_id"].astype(int)
    deals_df["type"] = deals_df["type"].map(lambda value: _safe_str(value).strip().lower())
    deals_df["direction"] = deals_df["direction"].map(lambda value: _safe_str(value).strip().lower())
    deals_df["volume"] = deals_df["volume"].map(_safe_float)
    deals_df["price"] = deals_df["price"].map(_safe_float)
    deals_df["commission"] = deals_df["commission"].map(_safe_float).fillna(0.0)
    deals_df["swap"] = deals_df["swap"].map(_safe_float).fillna(0.0)
    deals_df["profit"] = deals_df["profit"].map(_safe_float).fillna(0.0)
    deals_df["balance"] = deals_df["balance"].map(_safe_float)
    deals_df["comment"] = deals_df["comment"].map(_safe_str)

    orders_df = orders_df[
        [
            "order_id",
            "open_time",
            "fill_time",
            "symbol",
            "type",
            "volume_requested",
            "volume_filled",
            "requested_price",
            "sl_init",
            "tp_init",
            "state",
            "comment",
        ]
    ].copy()

    deals_df = deals_df[
        [
            "deal_id",
            "time",
            "order_id",
            "symbol",
            "type",
            "direction",
            "volume",
            "price",
            "commission",
            "swap",
            "profit",
            "balance",
            "comment",
        ]
    ].copy()

    return orders_df, deals_df


def build_bars_table(candles: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    bars = candles.reset_index().rename(columns={"time": "bar_time"})
    if "index" in bars.columns and "bar_time" not in bars.columns:
        bars = bars.rename(columns={"index": "bar_time"})
    if "bar_time" not in bars.columns:
        bars = bars.rename(columns={bars.columns[0]: "bar_time"})

    bars["bar_id"] = np.arange(len(bars), dtype=np.int64)
    bars["symbol"] = symbol
    bars["timeframe"] = timeframe
    bars["gap_min"] = bars["bar_time"].diff().dt.total_seconds().div(60.0)
    bars["gap_min"] = bars["gap_min"].fillna(5.0)
    bars["is_gap_bar"] = (bars["gap_min"] != 5.0).astype(int)
    bars["hour"] = bars["bar_time"].dt.hour
    bars["weekday"] = bars["bar_time"].dt.weekday
    bars["month"] = bars["bar_time"].dt.month
    return bars[
        [
            "bar_id",
            "symbol",
            "timeframe",
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "gap_min",
            "is_gap_bar",
            "hour",
            "weekday",
            "month",
        ]
    ].copy()


def build_bar_feature_table(candles: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    feature_cfg = cfg["feature"]

    returns_lags = [int(value) for value in feature_cfg["returns_lags"]]
    atr_window = int(feature_cfg["atr_window"])
    vol_window = int(feature_cfg["vol_window"])
    ema_fast = int(feature_cfg["ema_fast"])
    ema_slow = int(feature_cfg["ema_slow"])

    if ema_fast >= ema_slow:
        raise ValueError("feature.ema_fast deve ser menor que feature.ema_slow")

    df = candles.copy()
    close_safe = df["close"].replace(0.0, np.nan)

    for lag in returns_lags:
        df[f"ret_{lag}"] = df["close"].pct_change(lag)

    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["range_pct"] = (df["high"] - df["low"]) / close_safe
    df["body_pct"] = body / close_safe
    df["upper_wick_pct"] = upper_wick / close_safe
    df["lower_wick_pct"] = lower_wick / close_safe

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_abs"] = tr.rolling(atr_window).mean()
    df["atr_norm"] = df["atr_abs"] / close_safe

    ret_1 = df["close"].pct_change(1)
    df["realized_vol"] = ret_1.rolling(vol_window).std()

    vol_mean = df["volume"].rolling(vol_window).mean()
    vol_std = df["volume"].rolling(vol_window).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    ema_fast_series = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["ema_fast_gap"] = df["close"] / ema_fast_series - 1.0
    df["ema_slow_gap"] = df["close"] / ema_slow_series - 1.0
    df["ema_spread"] = ema_fast_series / ema_slow_series - 1.0

    feature_columns = (
        [f"ret_{lag}" for lag in returns_lags]
        + [
            "range_pct",
            "body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "atr_norm",
            "atr_abs",
            "realized_vol",
            "vol_z",
            "ema_fast_gap",
            "ema_slow_gap",
            "ema_spread",
        ]
    )

    feature_df = df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    feature_df.index.name = "bar_time"
    return feature_df, feature_columns


def _exit_reason_from_comment(comment: str) -> str:
    comment = _safe_str(comment).strip().lower()
    if comment.startswith("tp "):
        return "tp"
    if comment.startswith("sl "):
        return "sl"
    if comment == "":
        return "other"
    return comment.split(" ", 1)[0]


def reconstruct_trade_legs(
    orders_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    bar_features_df: pd.DataFrame,
) -> pd.DataFrame:
    order_lookup = orders_df.set_index("order_id").to_dict(orient="index")
    time_to_bar_id = dict(zip(bars_df["bar_time"], bars_df["bar_id"], strict=True))
    bar_time_lookup = dict(zip(bars_df["bar_id"], bars_df["bar_time"], strict=True))
    feature_lookup = bar_features_df.reset_index().rename(columns={"bar_time": "time"})
    feature_lookup = feature_lookup.set_index("time").to_dict(orient="index")

    open_buy_queue: deque[Dict] = deque()
    open_sell_queue: deque[Dict] = deque()
    trade_rows: List[Dict] = []

    for deal in deals_df.sort_values(["time", "deal_id"]).to_dict(orient="records"):
        if deal["direction"] == "in" and deal["type"] == "buy":
            open_buy_queue.append(deal)
            continue
        if deal["direction"] == "in" and deal["type"] == "sell":
            open_sell_queue.append(deal)
            continue

        if deal["direction"] == "out" and deal["type"] == "sell":
            if not open_buy_queue:
                raise ValueError("Encontrado sell/out sem buy/in correspondente no relatorio do tester.")
            entry_deal = open_buy_queue.popleft()
            side = "buy"
        elif deal["direction"] == "out" and deal["type"] == "buy":
            if not open_sell_queue:
                raise ValueError("Encontrado buy/out sem sell/in correspondente no relatorio do tester.")
            entry_deal = open_sell_queue.popleft()
            side = "sell"
        else:
            raise ValueError(f"Evento inesperado no relatorio: {deal}")

        entry_order = order_lookup.get(int(entry_deal["order_id"]))
        exit_order = order_lookup.get(int(deal["order_id"]))
        if entry_order is None or exit_order is None:
            raise ValueError("Falha ao casar ordem e deal pelo order_id.")

        entry_time = pd.Timestamp(entry_deal["time"])
        exit_time = pd.Timestamp(deal["time"])
        entry_bar_id = time_to_bar_id.get(entry_time)
        if entry_bar_id is None:
            raise ValueError(f"Entrada sem candle correspondente: {entry_time}")

        exit_floor_time = exit_time.floor("5min")
        exit_bar_id = time_to_bar_id.get(exit_floor_time)
        if exit_bar_id is None:
            raise ValueError(f"Saida sem candle floor correspondente: {exit_time}")

        entry_feature_row = feature_lookup.get(entry_time, {})
        entry_atr_abs = float(entry_feature_row.get("atr_abs", np.nan))

        entry_price = float(entry_deal["price"])
        sl_init = float(entry_order["sl_init"]) if not pd.isna(entry_order["sl_init"]) else float("nan")
        tp_init = float(entry_order["tp_init"]) if not pd.isna(entry_order["tp_init"]) else float("nan")

        risk_price = abs(entry_price - sl_init) if not pd.isna(sl_init) else float("nan")
        risk_source = "initial_stop"
        if pd.isna(risk_price) or risk_price <= 0.0:
            risk_price = entry_atr_abs
            risk_source = "entry_atr"

        reward_price = abs(tp_init - entry_price) if not pd.isna(tp_init) else float("nan")
        reward_to_risk = (
            float(reward_price / risk_price) if not pd.isna(reward_price) and risk_price and risk_price > 0 else float("nan")
        )

        trade_rows.append(
            {
                "trade_id": len(trade_rows) + 1,
                "symbol": entry_deal["symbol"],
                "side": side,
                "entry_order_id": int(entry_deal["order_id"]),
                "exit_order_id": int(deal["order_id"]),
                "entry_deal_id": int(entry_deal["deal_id"]),
                "exit_deal_id": int(deal["deal_id"]),
                "entry_time": entry_time,
                "entry_bar_id": int(entry_bar_id),
                "entry_bar_time": bar_time_lookup[entry_bar_id],
                "entry_price": entry_price,
                "entry_volume": float(entry_deal["volume"]),
                "entry_balance": float(entry_deal["balance"]),
                "sl_init": sl_init,
                "tp_init": tp_init,
                "risk_price": float(risk_price),
                "risk_source": risk_source,
                "reward_price": reward_price,
                "reward_to_risk": reward_to_risk,
                "entry_atr_abs": entry_atr_abs,
                "exit_time": exit_time,
                "exit_floor_time": exit_floor_time,
                "exit_bar_id": int(exit_bar_id),
                "exit_bar_time": bar_time_lookup[exit_bar_id],
                "exit_offset_sec": int((exit_time - exit_floor_time).total_seconds()),
                "exit_price": float(deal["price"]),
                "exit_reason": _exit_reason_from_comment(deal["comment"]),
                "profit_cash": float(deal["profit"]),
                "swap_cash": float(deal["swap"]),
                "commission_cash": float(deal["commission"]),
                "net_profit_cash": float(deal["profit"] + deal["swap"] + deal["commission"]),
                "close_balance": float(deal["balance"]),
                "bars_held_actual": int(exit_bar_id - entry_bar_id + 1),
                "actual_result_price": _signed_move(side, entry_price, float(deal["price"])),
                "actual_result_r": (
                    float(_signed_move(side, entry_price, float(deal["price"])) / risk_price)
                    if risk_price and risk_price > 0
                    else float("nan")
                ),
                "entry_comment": entry_order["comment"],
                "exit_comment": deal["comment"],
                "overlap_opposite_flag": 0,
                "opposite_overlap_count": 0,
            }
        )

    trade_legs_df = pd.DataFrame(trade_rows)

    overlap_flags = np.zeros(len(trade_legs_df), dtype=int)
    overlap_counts = np.zeros(len(trade_legs_df), dtype=int)
    trades = trade_legs_df.to_dict(orient="records")
    for left_idx in range(len(trades)):
        left = trades[left_idx]
        for right_idx in range(left_idx + 1, len(trades)):
            right = trades[right_idx]
            if right["entry_time"] > left["exit_time"]:
                break
            if left["side"] == right["side"]:
                continue
            overlap = max(left["entry_time"], right["entry_time"]) < min(left["exit_time"], right["exit_time"])
            if overlap:
                overlap_flags[left_idx] = 1
                overlap_flags[right_idx] = 1
                overlap_counts[left_idx] += 1
                overlap_counts[right_idx] += 1

    trade_legs_df["overlap_opposite_flag"] = overlap_flags
    trade_legs_df["opposite_overlap_count"] = overlap_counts
    return trade_legs_df


def build_position_snapshots(
    trade_legs_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    feature_table_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    snapshot_cfg = cfg.get("snapshots", {})
    include_entry_bar = bool(snapshot_cfg.get("include_entry_bar", True))
    include_exit_bar = bool(snapshot_cfg.get("include_exit_bar", False))
    store_only_actionable = bool(snapshot_cfg.get("store_only_actionable", True))

    bars = bars_df.set_index("bar_id")
    features = feature_table_df.reset_index().rename(columns={"bar_time": "bar_time"})
    features = features.merge(bars_df[["bar_id", "bar_time"]], on="bar_time", how="inner").set_index("bar_id")

    feature_columns = [column for column in feature_table_df.columns]
    recommended_feature_columns = (
        [
            "side_is_buy",
            "bars_since_entry",
            "mark_pnl_close_r",
            "mfe_to_date_r",
            "mae_to_date_r",
            "dist_to_sl_init_r",
            "dist_to_tp_init_r",
            "reward_to_risk",
            "opposite_leg_open",
            "hour",
            "weekday",
            "is_gap_bar",
        ]
        + feature_columns
    )

    trades = trade_legs_df.to_dict(orient="records")
    snapshot_rows: List[Dict] = []

    for trade in trades:
        entry_bar_id = int(trade["entry_bar_id"])
        exit_bar_id = int(trade["exit_bar_id"])
        start_bar_id = entry_bar_id if include_entry_bar else entry_bar_id + 1
        end_bar_id = exit_bar_id if include_exit_bar else exit_bar_id - 1
        if end_bar_id < start_bar_id:
            continue

        trade_bar_ids = np.arange(start_bar_id, end_bar_id + 1, dtype=np.int64)
        trade_bars = bars.loc[trade_bar_ids]
        if trade_bars.empty:
            continue

        if trade["side"] == "buy":
            mfe_series = (trade_bars["high"].cummax() - float(trade["entry_price"])).astype(float)
            mae_series = (float(trade["entry_price"]) - trade_bars["low"].cummin()).astype(float)
            mark_pnl_series = (trade_bars["close"] - float(trade["entry_price"])).astype(float)
            favorable_bar_series = (trade_bars["high"] - float(trade["entry_price"])).astype(float)
            adverse_bar_series = (float(trade["entry_price"]) - trade_bars["low"]).astype(float)
            dist_sl_series = (
                (trade_bars["close"] - float(trade["sl_init"])).astype(float)
                if not pd.isna(trade["sl_init"])
                else pd.Series(np.nan, index=trade_bars.index)
            )
            dist_tp_series = (
                (float(trade["tp_init"]) - trade_bars["close"]).astype(float)
                if not pd.isna(trade["tp_init"])
                else pd.Series(np.nan, index=trade_bars.index)
            )
        else:
            mfe_series = (float(trade["entry_price"]) - trade_bars["low"].cummin()).astype(float)
            mae_series = (trade_bars["high"].cummax() - float(trade["entry_price"])).astype(float)
            mark_pnl_series = (float(trade["entry_price"]) - trade_bars["close"]).astype(float)
            favorable_bar_series = (float(trade["entry_price"]) - trade_bars["low"]).astype(float)
            adverse_bar_series = (trade_bars["high"] - float(trade["entry_price"])).astype(float)
            dist_sl_series = (
                (float(trade["sl_init"]) - trade_bars["close"]).astype(float)
                if not pd.isna(trade["sl_init"])
                else pd.Series(np.nan, index=trade_bars.index)
            )
            dist_tp_series = (
                (trade_bars["close"] - float(trade["tp_init"])).astype(float)
                if not pd.isna(trade["tp_init"])
                else pd.Series(np.nan, index=trade_bars.index)
            )

        for bar_id in trade_bar_ids:
            bar_row = trade_bars.loc[bar_id]
            if bar_id not in features.index:
                continue
            feature_row = features.loc[bar_id]

            opposite_leg_open = 0
            for other in trades:
                if other["trade_id"] == trade["trade_id"] or other["side"] == trade["side"]:
                    continue
                if int(other["entry_bar_id"]) <= int(bar_id) < int(other["exit_bar_id"]):
                    opposite_leg_open = 1
                    break

            risk_price = float(trade["risk_price"])
            if not risk_price or pd.isna(risk_price) or risk_price <= 0:
                continue

            snapshot_row = {
                "snapshot_id": len(snapshot_rows) + 1,
                "trade_id": int(trade["trade_id"]),
                "symbol": trade["symbol"],
                "side": trade["side"],
                "side_is_buy": int(trade["side"] == "buy"),
                "bar_id": int(bar_id),
                "bar_time": bar_row["bar_time"],
                "entry_time": trade["entry_time"],
                "entry_bar_id": int(trade["entry_bar_id"]),
                "entry_price": float(trade["entry_price"]),
                "sl_init": float(trade["sl_init"]) if not pd.isna(trade["sl_init"]) else float("nan"),
                "tp_init": float(trade["tp_init"]) if not pd.isna(trade["tp_init"]) else float("nan"),
                "risk_price": risk_price,
                "reward_price": float(trade["reward_price"]) if not pd.isna(trade["reward_price"]) else float("nan"),
                "reward_to_risk": float(trade["reward_to_risk"]) if not pd.isna(trade["reward_to_risk"]) else float("nan"),
                "entry_balance": float(trade["entry_balance"]),
                "actual_exit_time": trade["exit_time"],
                "actual_exit_bar_id": int(trade["exit_bar_id"]),
                "actual_exit_price": float(trade["exit_price"]),
                "actual_exit_reason": trade["exit_reason"],
                "bars_since_entry": int(bar_id - trade["entry_bar_id"]),
                "current_open": float(bar_row["open"]),
                "current_high": float(bar_row["high"]),
                "current_low": float(bar_row["low"]),
                "current_close": float(bar_row["close"]),
                "current_volume": float(bar_row["volume"]),
                "current_gap_min": float(bar_row["gap_min"]),
                "is_gap_bar": int(bar_row["is_gap_bar"]),
                "hour": int(bar_row["hour"]),
                "weekday": int(bar_row["weekday"]),
                "month": int(bar_row["month"]),
                "mark_pnl_close_price": float(mark_pnl_series.loc[bar_id]),
                "mark_pnl_close_r": float(mark_pnl_series.loc[bar_id] / risk_price),
                "bar_favorable_move_price": float(favorable_bar_series.loc[bar_id]),
                "bar_adverse_move_price": float(adverse_bar_series.loc[bar_id]),
                "mfe_to_date_price": float(mfe_series.loc[bar_id]),
                "mae_to_date_price": float(mae_series.loc[bar_id]),
                "mfe_to_date_r": float(mfe_series.loc[bar_id] / risk_price),
                "mae_to_date_r": float(mae_series.loc[bar_id] / risk_price),
                "dist_to_sl_init_price": float(dist_sl_series.loc[bar_id]) if not pd.isna(dist_sl_series.loc[bar_id]) else float("nan"),
                "dist_to_tp_init_price": float(dist_tp_series.loc[bar_id]) if not pd.isna(dist_tp_series.loc[bar_id]) else float("nan"),
                "dist_to_sl_init_r": (
                    float(dist_sl_series.loc[bar_id] / risk_price) if not pd.isna(dist_sl_series.loc[bar_id]) else float("nan")
                ),
                "dist_to_tp_init_r": (
                    float(dist_tp_series.loc[bar_id] / risk_price) if not pd.isna(dist_tp_series.loc[bar_id]) else float("nan")
                ),
                "opposite_leg_open": opposite_leg_open,
                "overlap_opposite_flag": int(trade["overlap_opposite_flag"]),
            }

            for feature_name in feature_columns:
                value = feature_row.get(feature_name, np.nan)
                snapshot_row[feature_name] = float(value) if not pd.isna(value) else float("nan")

            snapshot_rows.append(snapshot_row)

    snapshots_df = pd.DataFrame(snapshot_rows)
    if snapshots_df.empty:
        raise ValueError("Nenhum snapshot de posicao foi gerado.")

    if store_only_actionable:
        required_columns = recommended_feature_columns + ["risk_price", "current_close"]
        snapshots_df = snapshots_df.dropna(subset=required_columns).reset_index(drop=True)
        snapshots_df["snapshot_id"] = np.arange(1, len(snapshots_df) + 1, dtype=np.int64)

    return snapshots_df, recommended_feature_columns


def _sanitize_action_name(action_name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", action_name.lower()).strip("_")


def _simulate_single_action(
    snapshot: Dict,
    action: Dict,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    time_arr: np.ndarray,
    bar_index_max: int,
    atr_abs_arr: np.ndarray,
    cfg: Dict,
) -> Dict:
    simulation_cfg = cfg.get("simulation", {})
    utility_cfg = simulation_cfg.get("utility", {})
    horizon_bars = int(simulation_cfg.get("horizon_bars", 48))
    allow_extension_beyond_actual_exit = bool(simulation_cfg.get("allow_extension_beyond_actual_exit", False))
    intrabar_priority = str(simulation_cfg.get("intrabar_priority", "stop_first")).strip().lower()
    use_next_bar_open_for_exit_now = bool(simulation_cfg.get("use_next_bar_open_for_exit_now", True))

    pnl_r_weight = float(utility_cfg.get("pnl_r_weight", 1.0))
    mae_r_weight = float(utility_cfg.get("mae_r_weight", 0.35))
    time_weight = float(utility_cfg.get("time_weight", 0.02))

    start_bar_id = int(snapshot["bar_id"]) + 1
    current_bar_id = int(snapshot["bar_id"])
    current_close = float(snapshot["current_close"])
    current_atr_abs = float(snapshot.get("atr_abs", np.nan))
    side = str(snapshot["side"])
    risk_price = float(snapshot["risk_price"])
    actual_exit_bar_id = int(snapshot["actual_exit_bar_id"])
    actual_exit_price = float(snapshot["actual_exit_price"])

    horizon_cap_bar_id = min(current_bar_id + horizon_bars, bar_index_max)
    if not allow_extension_beyond_actual_exit:
        horizon_cap_bar_id = min(horizon_cap_bar_id, actual_exit_bar_id)

    if start_bar_id > bar_index_max:
        exit_price = current_close
        final_pnl_price = _signed_move(side, float(snapshot["entry_price"]), exit_price)
        final_pnl_r = final_pnl_price / risk_price
        return {
            "action_name": action["name"],
            "action_kind": action["kind"],
            "is_valid_action": 1,
            "exit_bar_id": current_bar_id,
            "exit_time": snapshot["bar_time"],
            "exit_reason_sim": "no_future_bars",
            "exit_price_sim": exit_price,
            "final_pnl_price": final_pnl_price,
            "final_pnl_r": final_pnl_r,
            "incremental_pnl_r": final_pnl_r - float(snapshot["mark_pnl_close_r"]),
            "future_favorable_from_now_price": 0.0,
            "future_adverse_from_now_price": 0.0,
            "future_favorable_from_now_r": 0.0,
            "future_adverse_from_now_r": 0.0,
            "bars_forward": 0,
            "utility_score": pnl_r_weight * final_pnl_r,
        }

    stop_price = float(snapshot["sl_init"]) if not pd.isna(snapshot["sl_init"]) else None
    tp_price = float(snapshot["tp_init"]) if not pd.isna(snapshot["tp_init"]) else None

    action_kind = str(action["kind"]).strip().lower()
    if action_kind == "move_sl_to_entry":
        stop_price = _tighten_stop(side, stop_price, float(snapshot["entry_price"]))
    elif action_kind == "tighten_sl_atr":
        atr_mult = float(action.get("atr_mult", 1.0))
        if pd.isna(current_atr_abs) or current_atr_abs <= 0.0:
            stop_price = None
        else:
            desired_stop = current_close - atr_mult * current_atr_abs if side == "buy" else current_close + atr_mult * current_atr_abs
            stop_price = _tighten_stop(side, stop_price, desired_stop)

    if action_kind == "exit_now":
        fill_bar_id = min(start_bar_id, bar_index_max)
        exit_price = float(open_arr[fill_bar_id]) if use_next_bar_open_for_exit_now else current_close
        delta_from_now = _signed_move(side, current_close, exit_price)
        final_pnl_price = _signed_move(side, float(snapshot["entry_price"]), exit_price)
        final_pnl_r = final_pnl_price / risk_price
        future_favorable = max(0.0, delta_from_now)
        future_adverse = max(0.0, -delta_from_now)
        bars_forward = max(1, fill_bar_id - current_bar_id)
        utility_score = (
            pnl_r_weight * final_pnl_r
            - mae_r_weight * (future_adverse / risk_price)
            - time_weight * (bars_forward / max(1, horizon_bars))
        )
        return {
            "action_name": action["name"],
            "action_kind": action_kind,
            "is_valid_action": 1,
            "exit_bar_id": fill_bar_id,
            "exit_time": time_arr[fill_bar_id],
            "exit_reason_sim": "exit_now",
            "exit_price_sim": exit_price,
            "final_pnl_price": final_pnl_price,
            "final_pnl_r": final_pnl_r,
            "incremental_pnl_r": final_pnl_r - float(snapshot["mark_pnl_close_r"]),
            "future_favorable_from_now_price": future_favorable,
            "future_adverse_from_now_price": future_adverse,
            "future_favorable_from_now_r": future_favorable / risk_price,
            "future_adverse_from_now_r": future_adverse / risk_price,
            "bars_forward": bars_forward,
            "utility_score": utility_score,
        }

    if _is_stop_invalid_at_close(side, stop_price, current_close):
        return {
            "action_name": action["name"],
            "action_kind": action_kind,
            "is_valid_action": 0,
            "exit_bar_id": current_bar_id,
            "exit_time": snapshot["bar_time"],
            "exit_reason_sim": "invalid_action",
            "exit_price_sim": current_close,
            "final_pnl_price": float(snapshot["mark_pnl_close_price"]),
            "final_pnl_r": float(snapshot["mark_pnl_close_r"]),
            "incremental_pnl_r": 0.0,
            "future_favorable_from_now_price": 0.0,
            "future_adverse_from_now_price": 0.0,
            "future_favorable_from_now_r": 0.0,
            "future_adverse_from_now_r": 0.0,
            "bars_forward": 0,
            "utility_score": -1e9,
        }

    prev_close = current_close
    prev_atr_abs = current_atr_abs
    future_favorable_from_now_price = 0.0
    future_adverse_from_now_price = 0.0

    exit_price = float("nan")
    exit_bar_id = horizon_cap_bar_id
    exit_time = time_arr[horizon_cap_bar_id]
    exit_reason = "horizon"

    for bar_id in range(start_bar_id, horizon_cap_bar_id + 1):
        if action_kind == "trail_atr":
            atr_mult = float(action.get("atr_mult", 1.0))
            if not pd.isna(prev_atr_abs) and prev_atr_abs > 0.0:
                desired_stop = prev_close - atr_mult * prev_atr_abs if side == "buy" else prev_close + atr_mult * prev_atr_abs
                stop_price = _tighten_stop(side, stop_price, desired_stop)
                if _is_stop_invalid_at_close(side, stop_price, prev_close):
                    stop_price = None

        bar_open = float(open_arr[bar_id])
        bar_high = float(high_arr[bar_id])
        bar_low = float(low_arr[bar_id])
        bar_close = float(close_arr[bar_id])

        if side == "buy":
            future_favorable_from_now_price = max(future_favorable_from_now_price, bar_high - current_close)
            future_adverse_from_now_price = max(future_adverse_from_now_price, current_close - bar_low)
        else:
            future_favorable_from_now_price = max(future_favorable_from_now_price, current_close - bar_low)
            future_adverse_from_now_price = max(future_adverse_from_now_price, bar_high - current_close)

        gap_stop = stop_price is not None and ((side == "buy" and bar_open <= stop_price) or (side == "sell" and bar_open >= stop_price))
        gap_tp = tp_price is not None and ((side == "buy" and bar_open >= tp_price) or (side == "sell" and bar_open <= tp_price))
        if gap_stop or gap_tp:
            exit_bar_id = bar_id
            exit_time = time_arr[bar_id]
            if gap_stop and gap_tp:
                choose_stop = intrabar_priority == "stop_first"
                exit_reason = "stop_gap" if choose_stop else "tp_gap"
            elif gap_stop:
                exit_reason = "stop_gap"
            else:
                exit_reason = "tp_gap"
            exit_price = bar_open
            break

        intrabar_stop = stop_price is not None and ((side == "buy" and bar_low <= stop_price) or (side == "sell" and bar_high >= stop_price))
        intrabar_tp = tp_price is not None and ((side == "buy" and bar_high >= tp_price) or (side == "sell" and bar_low <= tp_price))

        if intrabar_stop or intrabar_tp:
            exit_bar_id = bar_id
            exit_time = time_arr[bar_id]
            if intrabar_stop and intrabar_tp:
                choose_stop = intrabar_priority == "stop_first"
                if choose_stop:
                    exit_reason = "stop"
                    exit_price = float(stop_price)
                else:
                    exit_reason = "tp"
                    exit_price = float(tp_price)
            elif intrabar_stop:
                exit_reason = "stop"
                exit_price = float(stop_price)
            else:
                exit_reason = "tp"
                exit_price = float(tp_price)
            break

        if not allow_extension_beyond_actual_exit and bar_id == actual_exit_bar_id:
            exit_bar_id = bar_id
            exit_time = time_arr[bar_id]
            exit_reason = "actual_exit_cap"
            exit_price = actual_exit_price
            break

        if bar_id == horizon_cap_bar_id:
            exit_bar_id = bar_id
            exit_time = time_arr[bar_id]
            exit_reason = "horizon"
            exit_price = bar_close
            break

        prev_close = bar_close
        bar_atr_abs = float(atr_abs_arr[bar_id])
        if not pd.isna(bar_atr_abs) and bar_atr_abs > 0.0:
            prev_atr_abs = bar_atr_abs

    if pd.isna(exit_price):
        exit_price = current_close

    final_pnl_price = _signed_move(side, float(snapshot["entry_price"]), exit_price)
    final_pnl_r = final_pnl_price / risk_price
    bars_forward = max(0, exit_bar_id - current_bar_id)
    utility_score = (
        pnl_r_weight * final_pnl_r
        - mae_r_weight * (future_adverse_from_now_price / risk_price)
        - time_weight * (bars_forward / max(1, horizon_bars))
    )

    return {
        "action_name": action["name"],
        "action_kind": action_kind,
        "is_valid_action": 1,
        "exit_bar_id": int(exit_bar_id),
        "exit_time": exit_time,
        "exit_reason_sim": exit_reason,
        "exit_price_sim": float(exit_price),
        "final_pnl_price": float(final_pnl_price),
        "final_pnl_r": float(final_pnl_r),
        "incremental_pnl_r": float(final_pnl_r - float(snapshot["mark_pnl_close_r"])),
        "future_favorable_from_now_price": float(future_favorable_from_now_price),
        "future_adverse_from_now_price": float(future_adverse_from_now_price),
        "future_favorable_from_now_r": float(future_favorable_from_now_price / risk_price),
        "future_adverse_from_now_r": float(future_adverse_from_now_price / risk_price),
        "bars_forward": int(bars_forward),
        "utility_score": float(utility_score),
    }


def build_action_outcomes(
    snapshots_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    bar_features_export_df: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    actions = cfg.get("simulation", {}).get("actions", [])
    if not actions:
        raise ValueError("simulation.actions precisa ter ao menos uma acao configurada.")

    bar_index_max = int(bars_df["bar_id"].iloc[-1])
    open_arr = bars_df["open"].to_numpy()
    high_arr = bars_df["high"].to_numpy()
    low_arr = bars_df["low"].to_numpy()
    close_arr = bars_df["close"].to_numpy()
    time_arr = bars_df["bar_time"].to_numpy()
    atr_abs_arr = np.full(len(bars_df), np.nan, dtype=float)
    if "atr_abs" in bar_features_export_df.columns:
        atr_rows = bar_features_export_df[["bar_id", "atr_abs"]].dropna(subset=["bar_id"])
        atr_abs_arr[atr_rows["bar_id"].astype(int).to_numpy()] = atr_rows["atr_abs"].astype(float).to_numpy()

    outcome_rows: List[Dict] = []
    snapshot_records = snapshots_df.to_dict(orient="records")
    total_snapshots = len(snapshot_records)
    for snapshot_index, snapshot in enumerate(snapshot_records, start=1):
        if snapshot_index % 5000 == 0:
            print(
                f"[dataset] stage=simulate_actions snapshot={snapshot_index}/{total_snapshots}",
                flush=True,
            )
        for priority, action in enumerate(actions):
            result = _simulate_single_action(
                snapshot,
                action,
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                time_arr,
                bar_index_max,
                atr_abs_arr,
                cfg,
            )
            result.update(
                {
                    "snapshot_id": int(snapshot["snapshot_id"]),
                    "trade_id": int(snapshot["trade_id"]),
                    "bar_id": int(snapshot["bar_id"]),
                    "bar_time": snapshot["bar_time"],
                    "action_priority": int(priority),
                }
            )
            outcome_rows.append(result)

    action_outcomes_df = pd.DataFrame(outcome_rows)
    if action_outcomes_df.empty:
        raise ValueError("Nenhum outcome de acao foi gerado.")
    return action_outcomes_df


def build_training_labels(action_outcomes_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    labels_cfg = cfg.get("labels", {})
    exit_action_names = {str(name) for name in labels_cfg.get("binary_exit_actions", ["exit_now"])}

    outcomes = action_outcomes_df.copy()
    outcomes["utility_sort"] = outcomes["utility_score"] - outcomes["action_priority"] * 1e-9
    best_idx = outcomes.groupby("snapshot_id")["utility_sort"].idxmax()
    best_rows = outcomes.loc[best_idx].copy()

    utility_pivot = (
        outcomes.pivot(index="snapshot_id", columns="action_name", values="utility_score")
        .rename(columns=lambda name: f"{_sanitize_action_name(name)}_utility")
        .reset_index()
    )
    validity_pivot = (
        outcomes.pivot(index="snapshot_id", columns="action_name", values="is_valid_action")
        .rename(columns=lambda name: f"{_sanitize_action_name(name)}_is_valid")
        .reset_index()
    )

    labels_df = best_rows[
        [
            "snapshot_id",
            "action_name",
            "action_kind",
            "utility_score",
            "final_pnl_r",
            "incremental_pnl_r",
            "future_adverse_from_now_r",
            "bars_forward",
        ]
    ].rename(
        columns={
            "action_name": "best_action",
            "action_kind": "best_action_kind",
            "utility_score": "best_action_utility_score",
            "final_pnl_r": "best_action_final_pnl_r",
            "incremental_pnl_r": "best_action_incremental_pnl_r",
            "future_adverse_from_now_r": "best_action_future_adverse_r",
            "bars_forward": "best_action_bars_forward",
        }
    )

    labels_df["label_should_exit"] = labels_df["best_action"].isin(exit_action_names).astype(int)
    labels_df["label_should_adjust_stop"] = (~labels_df["best_action"].isin(exit_action_names | {"hold_base"})).astype(int)
    labels_df = labels_df.merge(utility_pivot, on="snapshot_id", how="left")
    labels_df = labels_df.merge(validity_pivot, on="snapshot_id", how="left")

    exit_now_column = f"{_sanitize_action_name('exit_now')}_utility"
    hold_base_column = f"{_sanitize_action_name('hold_base')}_utility"
    if exit_now_column in labels_df.columns:
        labels_df["best_minus_exit_now_utility"] = labels_df["best_action_utility_score"] - labels_df[exit_now_column]
    else:
        labels_df["best_minus_exit_now_utility"] = np.nan

    if hold_base_column in labels_df.columns:
        labels_df["best_minus_hold_base_utility"] = labels_df["best_action_utility_score"] - labels_df[hold_base_column]
    else:
        labels_df["best_minus_hold_base_utility"] = np.nan

    return labels_df


def build_training_samples(
    snapshots_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    recommended_feature_columns: List[str],
) -> pd.DataFrame:
    snapshot_counts = snapshots_df.groupby("trade_id").size().rename("trade_snapshot_count").reset_index()
    training_samples_df = snapshots_df.merge(labels_df, on="snapshot_id", how="inner")
    training_samples_df = training_samples_df.merge(snapshot_counts, on="trade_id", how="left")
    training_samples_df["sample_weight"] = 1.0 / training_samples_df["trade_snapshot_count"].clip(lower=1)

    ordered_columns = (
        ["snapshot_id", "trade_id", "bar_id", "bar_time"]
        + recommended_feature_columns
        + [
            "best_action",
            "label_should_exit",
            "label_should_adjust_stop",
            "best_action_utility_score",
            "best_minus_exit_now_utility",
            "best_minus_hold_base_utility",
            "sample_weight",
        ]
    )
    present_columns = [column for column in ordered_columns if column in training_samples_df.columns]
    remaining_columns = [column for column in training_samples_df.columns if column not in present_columns]
    return training_samples_df[present_columns + remaining_columns].copy()


def build_position_management_dataset(
    candles: pd.DataFrame,
    report_path: Path,
    output_root: Path,
    symbol: str,
    timeframe: str,
    cfg: Dict,
    output_format: str,
) -> Dict:
    resolved_format = resolve_output_format(output_format)

    bronze_dir = output_root / "bronze"
    silver_dir = output_root / "silver"
    gold_dir = output_root / "gold"
    ensure_dir(bronze_dir)
    ensure_dir(silver_dir)
    ensure_dir(gold_dir)

    orders_df, deals_df = parse_tester_report(report_path)
    bars_df = build_bars_table(candles, symbol=symbol, timeframe=timeframe)
    bar_features_df, feature_columns = build_bar_feature_table(candles, cfg)
    bar_features_export_df = (
        bar_features_df.reset_index()
        .rename(columns={"bar_time": "bar_time"})
        .merge(bars_df[["bar_id", "bar_time"]], on="bar_time", how="inner")
    )
    trade_legs_df = reconstruct_trade_legs(
        orders_df=orders_df,
        deals_df=deals_df,
        bars_df=bars_df,
        bar_features_df=bar_features_df,
    )
    snapshots_df, recommended_feature_columns = build_position_snapshots(
        trade_legs_df=trade_legs_df,
        bars_df=bars_df,
        feature_table_df=bar_features_df,
        cfg=cfg,
    )
    action_outcomes_df = build_action_outcomes(
        snapshots_df=snapshots_df,
        bars_df=bars_df,
        bar_features_export_df=bar_features_export_df,
        cfg=cfg,
    )
    labels_df = build_training_labels(action_outcomes_df=action_outcomes_df, cfg=cfg)
    training_samples_df = build_training_samples(
        snapshots_df=snapshots_df,
        labels_df=labels_df,
        recommended_feature_columns=recommended_feature_columns,
    )

    paths = {
        "candles": save_table(candles.reset_index(), bronze_dir / "candles", resolved_format),
        "orders": save_table(orders_df, bronze_dir / "orders", resolved_format),
        "deals": save_table(deals_df, bronze_dir / "deals", resolved_format),
        "bars": save_table(bars_df, silver_dir / "bars", resolved_format),
        "bar_features": save_table(bar_features_export_df, silver_dir / "bar_features", resolved_format),
        "trade_legs": save_table(trade_legs_df, silver_dir / "trade_legs", resolved_format),
        "position_snapshots": save_table(snapshots_df, gold_dir / "position_snapshots", resolved_format),
        "action_outcomes": save_table(action_outcomes_df, gold_dir / "action_outcomes", resolved_format),
        "training_labels": save_table(labels_df, gold_dir / "training_labels", resolved_format),
        "training_samples": save_table(training_samples_df, gold_dir / "training_samples", resolved_format),
    }

    best_action_distribution = (
        labels_df["best_action"].value_counts(dropna=False).sort_index().to_dict()
        if not labels_df.empty
        else {}
    )
    label_should_exit_distribution = (
        labels_df["label_should_exit"].value_counts(dropna=False).sort_index().to_dict()
        if not labels_df.empty
        else {}
    )

    overlap_count = int(trade_legs_df["overlap_opposite_flag"].sum())
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "report_path": str(report_path),
        "output_root": str(output_root),
        "output_format": resolved_format,
        "feature_columns": feature_columns,
        "recommended_training_columns": recommended_feature_columns,
        "simulation_actions": cfg.get("simulation", {}).get("actions", []),
        "counts": {
            "candles": int(len(candles)),
            "orders": int(len(orders_df)),
            "deals": int(len(deals_df)),
            "trade_legs": int(len(trade_legs_df)),
            "snapshots": int(len(snapshots_df)),
            "action_outcomes": int(len(action_outcomes_df)),
            "training_labels": int(len(labels_df)),
            "training_samples": int(len(training_samples_df)),
        },
        "time_range": {
            "candles_start": str(candles.index.min()) if len(candles) else None,
            "candles_end": str(candles.index.max()) if len(candles) else None,
            "trades_start": str(trade_legs_df["entry_time"].min()) if len(trade_legs_df) else None,
            "trades_end": str(trade_legs_df["exit_time"].max()) if len(trade_legs_df) else None,
        },
        "quality": {
            "overlap_opposite_trade_count": overlap_count,
            "best_action_distribution": best_action_distribution,
            "label_should_exit_distribution": label_should_exit_distribution,
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return summary
