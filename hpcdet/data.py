"""Loading and validating HPC trace files."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Counters present in the provided clean trace file (column order is irrelevant: we select by name).
DEFAULT_COLUMNS = ("cache-references", "cycles", "LLC-loads")


def load_traces(path: str, columns: Optional[Sequence[str]] = None,
                require_all_rows_valid: bool = False) -> pd.DataFrame:
    """Read a trace CSV and return only the requested HPC columns as float64.

    Extra columns (e.g. a ``label`` column in a validation file) are ignored here and
    must be read separately by the caller.  Rows with non-finite or non-positive counter
    values cannot be log-transformed; they are reported and (by default) kept as NaN so
    that ``predict`` can flag them as anomalous instead of crashing.
    """
    columns = list(columns or DEFAULT_COLUMNS)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing HPC columns {missing}; found {list(df.columns)}")
    X = df[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    bad = ~np.isfinite(X.values).all(axis=1) | (X.values <= 0).any(axis=1)
    if bad.any():
        msg = f"{path}: {int(bad.sum())} rows have non-finite / non-positive counters"
        if require_all_rows_valid:
            raise ValueError(msg)
        log.warning(msg)
    return X
