#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: callables.py
#   Author: xyy15926
#   Created: 2025-01-14 21:41:50
#   Updated: 2025-07-31 21:37:48
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any
from functools import wraps
from collections.abc import Mapping, Sequence, Callable, Container
import numpy as np
import pandas as pd
import warnings
# from IPython.core.debugger import set_trace

from ubears.modsbear.locale.calender import is_chn_busday, not_chn_busday


today = np.datetime64("today")
count = len
is_busiday = is_chn_busday
not_busiday = not_chn_busday
isnull = pd.isna
notnull = pd.notna
nan = np.nan


# %% -------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def max(x: Sequence):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN (axis|slice) encountered")
        return np.nan if len(x) == 0 else np.nanmax(x)


def min(x: Sequence):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN (axis|slice) encountered")
        return np.nan if len(x) == 0 else np.nanmin(x)


def sum(x: Sequence):
    return np.nansum(x)


def avg(x: Sequence):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        return np.nanmean(x)


def hist(x: Sequence, edges: list):
    return np.histogram(x, edges)[0]


def coef_var(x: Sequence):
    return 0 if len(x) == 0 else np.nanstd(x) / np.nanmean(x)


# %%
def nnfilter(x: Sequence):
    return [ele for ele in x if ele is not None]


def getn(x: Sequence, y: int):
    x = np.asarray(x)
    if y is None or y >= len(x):
        return None
    else:
        return x[y]


# TODO
def drop_duplicates(xs: list[Sequence]):
    """
    """
    if isinstance(xs, list) and len(xs) > 0 and not np.isscalar(xs[0]):
        df = pd.DataFrame(xs).T
        ret = df.drop_duplicates().values
    else:
        ser = (pd.Series(xs, dtype="O")
               if isinstance(xs, list) and len(xs) == 0
               else pd.Series(xs))
        ret = ser.drop_duplicates().values
    return ret


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


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def sadd(x: Sequence, y: Sequence):
    x = np.asarray(x)
    y = np.asarray(y)
    return x + y


def ssub(x: Sequence, y: Sequence):
    x = np.asarray(x)
    y = np.asarray(y)
    return x - y


def smul(x: Sequence, y: Sequence):
    x = np.asarray(x)
    y = np.asarray(y)
    return x * y


def sdiv(x: Sequence, y: Sequence):
    x = np.asarray(x)
    y = np.asarray(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / y


# %%
def sortby(x: Sequence, y: Sequence, ascending: bool = 1):
    x = np.asarray(x)
    y = np.asarray(y)
    return x[np.argsort(y)][::1 if ascending else -1]


# %%
def mon_itvl(x: Sequence, y: Sequence):
    x = np.asarray(x, dtype="datetime64[M]")
    y = np.asarray(y, dtype="datetime64[M]")
    sub = x - y
    ret = np.select([~np.isnat(sub), np.isnat(sub)],
                    [sub.astype(np.float32), np.nan])
    return ret


def day_itvl(x: Sequence, y: Sequence):
    x = np.asarray(x, dtype="datetime64[D]")
    y = np.asarray(y, dtype="datetime64[D]")
    sub = x - y
    ret = np.select([~np.isnat(sub), np.isnat(sub)],
                    [sub.astype(np.float32), np.nan])
    return ret


# %%
def cb_fst(x: Sequence | Any,
           y: Sequence | Any):
    """Combine the first not-NA from two alternatives.

    1. `pd.isna` is called instead of `np.isnan` so to be compatiable with
      non-numeric array.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ret = np.select([~pd.isna(x),], [x,], y)
    return ret


def cb_max(x: Sequence | int | float,
           y: Sequence | int | float):
    """Combine the larger one from two alternatives.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ret = np.select([(x >= y) | np.isnan(y),], [x,], y)
    return ret


def cb_min(x: Sequence | int | float,
           y: Sequence | int | float):
    """Combine the smaller one from two alternatives.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ret = np.select([(x <= y) | np.isnan(y),], [x,], y)
    return ret


# %%
def get_hour(x: Sequence):
    # Call `np.asarray` first so to ensure `pd.to_datetime` will return
    # a DatetimeIndex instead of a Series if a Series passed in.
    x = np.asarray(x)
    ret = np.asarray(pd.to_datetime(x).hour)
    return ret


# %%
def map(*args):
    """Map the arguments with provided callables or dict.

    Params:
    ---------------------
    args[-1]: Default value if not callable or a dict.
    args[-2] or args[-1]: Callables or dict as mapping references.
    args[:-2] or args[:-1]: Seqences to be zipped and mapped sequently.

    Return:
    ---------------------
    Mapping result.
    """
    if isinstance(args[-1], Callable) or isinstance(args[-1], dict):
        *args, ref = args
        # Use `np.nan` as default instead of `None` to prevent the numeric
        # result from being casted into object.
        default = np.nan
    elif isinstance(args[-2], Callable) or isinstance(args[-2], dict):
        *args, ref, default = args
    else:
        raise ValueError("No valid mapping reference is provided.")

    if isinstance(ref, dict):
        assert len(args) == 1, "Invalid mapping source."
        ret = [ref.get(ele, default) for ele in args[0]]
    else:
        if len(args) == 1:
            ret = [ref(ele) for ele in args[0]]
        else:
            ret = [ref(*ele) for ele in zip(*args)]

    return np.asarray(ret)


def sep_map(x: Sequence,
            ref: Mapping | Callable,
            sep_from: str = ",",
            sep_to: str = None):
    """Seperate element and then apply mapping.

    1. Duplicated mapping result or None will be dropped.

    Params:
    ---------------------
    x: Sequence of tuple, list or string.
    ref: Mapping reference.
    sep_from: String, if so, in `x` will splited with `sep_from` first.
    sep_to: Seperator to join the mapping result if provided.
      Else tuple will the result.

    Return:
    ---------------------
    np.ndarray of string or tuple.
    """
    def ele_sep_map(val: str | tuple | list):
        # Split first if string provided.
        if isinstance(val, str):
            val = val.strip().split(sep_from)
        # `val` maybe None, np.nan or other non-iterable.
        if not np.isscalar(val):
            sval = (set([ref.get(ele) for ele in val if ele in ref])
                    if hasattr(ref, "get")
                    else set([ref(ele) for ele in val]))
        else:
            sval = set()

        return (tuple(sval) if sep_to is None
                else sep_to.join([str(ele) for ele in sval]))

    ret = [ele_sep_map(ele) for ele in x]
    return np.array(ret, dtype="object")


# %%
def isin(x: Sequence | Any,
         y: Sequence | Container):
    """Check if `x` in `y`.

    1. If `x` is a sequence, then elements in `x` will be checked if existing
      in `y`.
    2. If `x` is a scalar, then `x` will be checked if existing in each
      element in `y`.
    """
    if np.isscalar(x):
        ret = [x in ele if isinstance(ele, Container) else False for ele in y]
    else:
        y = set(y)
        ret = [ele in y for ele in x]
    return np.array(ret, dtype=np.bool_)
