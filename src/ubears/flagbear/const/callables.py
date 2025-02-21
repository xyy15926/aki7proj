#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: callables.py
#   Author: xyy15926
#   Created: 2025-01-14 21:41:50
#   Updated: 2025-02-21 17:51:16
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any
from functools import wraps
from collections.abc import Mapping, Sequence, Callable
import numpy as np
import pandas as pd
# from IPython.core.debugger import set_trace

from ubears.modsbear.locale.calender import is_chn_busday, not_chn_busday


today = np.datetime64("today")
drop_duplicates = np.unique
max = np.nanmax
min = np.nanmin
count = len
is_busiday = is_chn_busday
not_busiday = not_chn_busday
isnull = np.isnan
nan = np.nan


# %% -------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def nnfilter(seq: Sequence):
    return [x for x in seq if x is not None]


def getn(x: Sequence, y: int):
    x = np.asarray(x)
    if y is None or y >= len(x):
        return None
    else:
        return x[y]


# %%
def argmax(x: Sequence):
    if len(x) == 0 or np.all(np.isnan(x)):
        return None
    else:
        return np.nanargmax(x)


def argmin(x: Sequence):
    if len(x) == 0 or np.all(np.isnan(x)):
        return None
    else:
        return np.nanargmin(x)


def argmaxs(x: Sequence, y: Sequence):
    if len(x) == 0:
        return np.array([])
    x = np.asarray(x)
    y = np.asarray(y)
    return y[x == np.nanmax(x)]


def argmins(x: Sequence, y: Sequence):
    if len(x) == 0:
        return np.array([])
    x = np.asarray(x)
    y = np.asarray(y)
    return y[x == np.nanmin(x)]


# %%
def flat1_max(x: Sequence):
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


# %%
def coef_var(x: Sequence):
    x = np.asarray(x)
    return 0 if len(x) == 0 else np.std(x) / np.mean(x)


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def sortby(x: Sequence, y: Sequence, ascending: bool = 1):
    x = np.asarray(x)
    y = np.asarray(y)
    return x[np.argsort(y)][::1 if ascending else -1]


# %%
def mon_itvl(x: Sequence, y: Sequence):
    """
    """
    x = np.asarray(x, dtype="datetime64[M]")
    y = np.asarray(y, dtype="datetime64[M]")
    sub = x - y
    ret = np.select([~np.isnat(sub), np.isnat(sub)],
                    [sub.astype(np.float32), np.nan])
    return ret


def day_itvl(x: Sequence, y: Sequence):
    """
    """
    x = np.asarray(x, dtype="datetime64[D]")
    y = np.asarray(y, dtype="datetime64[D]")
    sub = x - y
    ret = np.select([~np.isnat(sub), np.isnat(sub)],
                    [sub.astype(np.float32), np.nan])
    return ret


# %%
def cb_max(x: Sequence | int | float,
           y: Sequence | int | float):
    x = np.asarray(x)
    y = np.asarray(y)
    ret = np.select([x >= y,], [x,], y)
    return ret


def cb_min(x: Sequence | int | float,
           y: Sequence | int | float):
    x = np.asarray(x)
    y = np.asarray(y)
    ret = np.select([x <= y,], [x,], y)
    return ret


def cb_fst(x: Sequence | int | float,
           y: Sequence | int | float):
    x = np.asarray(x)
    y = np.asarray(y)
    flags = np.isnan(x)
    ret = np.select([~flags,], [x,], y)
    return ret


# %%
def get_hour(x: Sequence):
    x = np.asarray(x)
    ret = np.asarray(pd.to_datetime(x).hour)
    return ret


# %%
def map(x: Sequence, ref: dict, z: Any = None):
    ret = [0] * len(x)
    for idx, ele in enumerate(x):
        ret[idx] = ref.get(ele, z)
    return np.asarray(ret)


def sep_map(x: Sequence,
            ref: Mapping,
            sep_from: str = ",",
            sep_to: str = None):
    def ele_sep_map(val: str | tuple | list):
        vals = set()
        if isinstance(val, str):
            val = val.strip().split(sep_from)
        for ele in val:
            med = ref.get(ele)
            if med is not None:
                vals.add(med)
        if sep_to is None:
            return tuple(vals)
        else:
            return sep_to.join([str(ele) for ele in vals])
    ret = [ele_sep_map(ele) for ele in x]
    return np.array(ret, dtype="object")
