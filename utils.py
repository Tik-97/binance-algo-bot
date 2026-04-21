import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import ALL_COLUMNS, DEFAULT_INPUT_CANDIDATES


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("trade_reverse_analysis")


def autodetect_input_file(base_dir: Path) -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        "Input file not found. Expected one of: "
        + ", ".join(DEFAULT_INPUT_CANDIDATES)
    )


def extract_trade_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], dict):
            nested = payload["data"]
            if "list" in nested and isinstance(nested["list"], list):
                return [item for item in nested["list"] if isinstance(item, dict)]
        if "list" in payload and isinstance(payload["list"], list):
            return [item for item in payload["list"] if isinstance(item, dict)]
        for value in payload.values():
            extracted = extract_trade_records(value)
            if extracted:
                return extracted
    return []


def load_trades(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    logger.info("Loading trade history from %s", input_path)

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        records = extract_trade_records(payload)
        if not records:
            raise ValueError(f"No trade records found in JSON: {input_path}")
        df = pd.DataFrame(records)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    logger.info("Loaded %s raw rows", len(df))
    return df


def ensure_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    normalized = df.copy()
    for column in ALL_COLUMNS:
        if column not in normalized.columns:
            logger.warning("Missing expected column '%s'. Filling with nulls.", column)
            normalized[column] = pd.NA
    return normalized


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_mixed_datetime_to_utc(series: pd.Series) -> pd.Series:
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_mask = numeric.notna()

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="ms",
            errors="coerce",
            utc=True,
        )

    string_mask = ~numeric_mask & raw.notna()
    if string_mask.any():
        parsed.loc[string_mask] = pd.to_datetime(
            raw.loc[string_mask],
            errors="coerce",
            utc=True,
        )

    return parsed


def safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        if denominator in (0, 0.0) or pd.isna(denominator):
            return float("nan")
        return float(numerator) / float(denominator)
    except Exception:
        return float("nan")


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def fmt_hours(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f} h"
