"""Shared JSON contract: exact identifiers, finite numbers, run provenance."""
import math
from dataclasses import asdict, is_dataclass
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,set)):
        return [clean(v) for v in value]
    if is_dataclass(value):
        return clean(asdict(value))
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
