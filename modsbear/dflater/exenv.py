#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exenv.py
#   Author: xyy15926
#   Created: 2024-11-10 19:33:40
#   Updated: 2024-12-03 20:09:52
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable
from collections import ChainMap

import logging
import numpy as np
import pandas as pd
# from IPython.core.debugger import set_trace

from modsbear.locale.calender import ChineseHolidaysCalendar

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def _sortby(x: pd.Series, y: pd.Series, ascending: bool = 1):
    return x[y.sort_values(ascending=ascending).index]


def _flat1_max(x: pd.Series):
    """Count the length of max-continuous 1's in x.
    """
    x = np.sign(np.asarray(x, dtype=np.int8))
    if len(x) == 0:
        return 0
    if len(x) == 1:
        return x[0]
    x = np.concatenate([[0,], x[1:] - x[:-1], [1 if x[-1] == 0 else -1,]])
    xx = x[x != 0]
    edges = np.concatenate([[0,], np.arange(len(x))[x != 0]])
    flat1s = (edges[1:] - edges[:-1])[xx < 0]
    return 0 if len(flat1s) == 0 else flat1s.max()


def _getn(x: pd.Series, y: int):
    if y is None or y >= len(x):
        return None
    if isinstance(x, pd.Series):
        return x.iloc[y]
    else:
        return x[y]


def _mon_itvl(x: pd.Series | pd.Timestamp, y: pd.Series | pd.Timestamp):
    """Months of the intervals.
    The arguments will be converted the Period[M] directly without consider
    the days.

    Example:
    >>> mon_itvl("2023-02-01", "2023-01-31") == 1
    """
    # x = np.array(x, dtype="datetime64[M]")
    # y = np.array(y, dtype="datetime64[M]")
    # return np.asarray(x - y, dtype=int)
    if x is None or y is None:
        return np.nan

    if isinstance(x, pd.Series):
        x = pd.to_datetime(x).dt.to_period("M")
    else:
        x = pd.to_datetime(x).to_period("M")
    if isinstance(y, pd.Series):
        y = pd.to_datetime(y).dt.to_period("M")
    else:
        y = pd.to_datetime(y).to_period("M")

    return (x - y).apply(lambda x: getattr(x, "n", np.nan))


def _day_itvl(x: pd.Series | pd.Timestamp, y: pd.Series | pd.Timestamp):
    """Months of the intervals.
    The arguments will be converted the Period[M] directly without consider
    the days.

    Example:
    >>> mon_itvl("2023-02-01", "2023-01-31") == 1
    """
    # x = np.array(x, dtype="datetime64[M]")
    # y = np.array(y, dtype="datetime64[M]")
    # return np.asarray(x - y, dtype=int)
    if x is None or y is None:
        return np.nan

    if isinstance(x, pd.Series):
        x = pd.to_datetime(x).dt.to_period("D")
    else:
        x = pd.to_datetime(x).to_period("D")
    if isinstance(y, pd.Series):
        y = pd.to_datetime(y).dt.to_period("D")
    else:
        y = pd.to_datetime(y).to_period("D")

    return (x - y).apply(lambda x: getattr(x, "n", np.nan))


def _ser_map(x: pd.Series, y: dict | Callable, z: Any = None):
    """Map with default value for NaN.
    """
    if z is None:
        return x.map(y)
    else:
        return x.map(y).fillna(z)


def _argmaxs(x: pd.Series, y: pd.Series):
    """Get all the corresponding values for the maximums.
    """
    if len(x) == 0:
        return pd.Series(dtype=int)
    max_ = np.nanmax(x)
    return y[x == max_]


def _argmins(x: pd.Series, y: pd.Series):
    """Get all the corresponding values for the minimums.
    """
    if len(x) == 0:
        return pd.Series(dtype=int)
    min_ = np.nanmin(x)
    return y[x == min_]


def _cb_max(x: pd.Series | int, y: pd.Series | int):
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return pd.concat([x, y], axis=1).max(axis=1)
    elif isinstance(x, pd.Series):
        return x.apply(lambda ele: ele if ele > y else y)
    elif isinstance(y, pd.Series):
        return y.apply(lambda ele: ele if ele > x else x)
    else:
        return np.nanmax([x, y])


def _cb_min(x: pd.Series | int, y: pd.Series | int):
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return pd.concat([x, y], axis=1).min(axis=1)
    elif isinstance(x, pd.Series):
        return x.apply(lambda ele: ele if ele < y else y)
    elif isinstance(y, pd.Series):
        return y.apply(lambda ele: ele if ele < x else x)
    else:
        return np.nanmin([x, y])


def _drop_duplicates(x: pd.Series | list):
    if isinstance(x, list):
        x = pd.concat(x, axis=1)
    return x.drop_duplicates()


def _not_busiday(x: pd.Series):
    chc = ChineseHolidaysCalendar().holidays()
    return x.isin(chc) | x.dt.weekday.isin([0, 6])


def _is_busiday(x: pd.Series):
    return ~_not_busiday(x)


def _sep_map(ser: pd.Series[str],
             mapref: Mapping,
             sep_from: str = ",",
             sep_to: str = None):
    def str_sep_map(val: str):
        vals = set()
        for ele in val.strip().split(sep_from):
            med = mapref.get(ele)
            if pd.notna(med):
                vals.add(med)
        if sep_to is None:
            return tuple(vals)
        else:
            return sep_to.join([str(ele) for ele in vals])
    return ser.apply(str_sep_map)


def _any_contains(ser: pd.Series[tuple], target: Sequence):
    return ser.apply(lambda x: np.any([ele in x for ele in target]))


# %%
EXGINE_ENV = {
    "today"     : pd.Timestamp.today(),
    "map"       : _ser_map,
    "cb_fst"    : pd.Series.combine_first,
    "cb_max"    : _cb_max,
    "cb_min"    : _cb_min,
    "mon_itvl"  : _mon_itvl,
    "day_itvl"  : _day_itvl,
    "drop_duplicates": _drop_duplicates,
    "isin"      : lambda x, y: x.isin(y),
    "count"     : len,
    "unique"    : np.unique,
    "sum"       : lambda x: x.sum(),
    "max"       : lambda x: x.max(),
    "min"       : lambda x: x.min(),
    "avg"       : lambda x: x.mean(),
    "nnfilter"  : lambda x: [i for i in x if i is not None],
    "nncount"   : lambda x: len([i for i in x if i is not None]),
    "flat1_max" : _flat1_max,
    "sortby"    : _sortby,
    "argmax"    : lambda x: None if len(x) == 0 else np.argmax(x),
    "argmin"    : lambda x: None if len(x) == 0 else np.argmin(x),
    "argmaxs"   : _argmaxs,
    "argmins"   : _argmins,
    "getn"      : _getn,
    "head"      : lambda x, y: x.iloc[:y],
    "tail"      : lambda x, y: x.iloc[-y:],
    "sadd"      : lambda x, y: x + y,
    "ssub"      : lambda x, y: x - y,
    "smul"      : lambda x, y: x * y,
    "sdiv"      : lambda x, y: np.nan if isinstance(y, int) and y == 0 else x / y,
    "hist"      : lambda x, y: np.histogram(x, y)[0],
    "coef_var"  : lambda x: 0 if len(x) == 0 else np.std(x) / np.mean(x),
    "contains"  : lambda x, y: x.apply(lambda val: y in val),
    "isnull"    : pd.isna,
    "notnull"   : pd.notna,
    "is_busiday": _is_busiday,
    "not_busiday": _not_busiday,
    "sep_map"   : _sep_map,
    "any_contains": _any_contains,
}
