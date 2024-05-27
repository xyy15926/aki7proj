#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exgine.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2024-05-24 17:19:48
#   Description:
# ---------------------------------------------------------


# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Mapping
from collections import ChainMap
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.parser import EnvParser
from flagbear.fliper import rebuild_dict, extract_field
from flagbear.fliper import regex_caster


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
    return x.iloc[y]


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


def _ser_map(x: pd.Series, y: dict, z: Any = None):
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


EXGINE_ENV = {
    "today"     : pd.Timestamp.today(),
    "map"       : _ser_map,
    "cb_fst"    : pd.Series.combine_first,
    "cb_max"    : _cb_max,
    "cb_min"    : _cb_min,
    "mon_itvl"  : _mon_itvl,
    "day_itvl"  : lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.days,
    "count"     : len,
    "unique"    : np.unique,
    "sum"       : lambda x: x.sum(),
    "max"       : lambda x: x.max(),
    "min"       : lambda x: x.min(),
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
    "contains"  : lambda x, y: y in x,
}


# %%
def rebuild_rec2df(
    rec: str | dict,
    val_rules: list[tuple],
    index_rules: list[tuple] = None,
    env: Mapping = None,
    envp: EnvParser = None,
    explode: bool = False,
    range_index: str = None,
) -> pd.DataFrame:
    """Parse fields from record to construct DataFrame.

    Params:
    -------------------------
    rec: Record.
    val_rules: List of 2/3/4/5-Tuple of rules for value extraction.
      Extractions will be used as the value of the returned DataFrame.
      2-Tuple: [key, steps]
      3-Tuple: [key, steps, dtype]
      4-Tuple: [key, from_, steps, dtype]
      5-Tuple: [key, from_, steps, dtype, default]
        key: Key in the new dict.
        from_: Dependency and source from which get the value and will be
          passed to `extract_field` as `obj.`
        steps: Steps passed to `extract_field` as `steps`.
        dtype: Dtype passed to `extract_field` as `dtype`.
        default: Default value passed to `extract_field` as `dfill`.
      None: Keep original value.
    index_rules: List of 2/4/5-Tuple of rules for value extraction.
      Each of the extractions could be a scalar or a list with the same length
      as the value extractions. So the whole extractions could be broadcasted
      to be used as the index of the returned DataFrame.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.
    explode: If to explode the result into DataFrame with multiple rows when
      the values are lists.
      So the values must be all scalar or list with the same length if this is
      set.
    range_index: True value represents to add RangeIndex named with
      `range_index`.

    Return:
    -------------------------
    DataFrame with Columns[KEY in `val_rules`] and Index named with
      [KEY in `index_rules`].
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    # Extract some fields as index while keeping the record untouched.
    if val_rules is None:
        val_dict = {None: rec}
    else:
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON string: {rec}.")
                return pd.DataFrame()

        val_dict = rebuild_dict(rec, val_rules, envp)

    # In case that no valid values extracted.
    if not val_dict:
        vals = pd.DataFrame(columns=[i[0] for i in val_rules])
    # Convert extracted values to DataFrame.
    elif explode and isinstance(next(iter(val_dict.values())), list):
        vals = pd.DataFrame(val_dict)
    else:
        vals = pd.DataFrame.from_records([val_dict])

    index_arrays = []
    index_names = []
    if index_rules:
        index_dict = rebuild_dict(rec, index_rules, envp)
        index_names = list(index_dict.keys())
        # Convert `index_dict` into MultiIndex.
        for ival in index_dict.values():
            if isinstance(ival, list):
                assert (len(ival) == len(vals))
                index_arrays.append(ival)
            else:
                index_arrays.append([ival] * len(vals))

    # Check if to add a new level of index.
    if range_index:
        index_arrays.append(np.arange(len(vals), dtype=np.int_))
        index_names.append(range_index)

    # In case that no valid values extracted.
    if index_arrays:
        index_ = pd.MultiIndex.from_arrays(index_arrays,
                                           names=index_names)
        vals.index = index_

    return vals


# %%
def trans_on_df(
    df: pd.DataFrame,
    rules: list[tuple],
    env: Mapping = None,
    envp: EnvParser = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """Apply transformation on DataFrame.

    Params:
    --------------------
    df: DataFrame of data.
    rules: List of 2/3-Tuple.
      2-Tuple: [KEY, TRANS]
      3-Tuple: [KEY, COND, TRANS]
        KEY: Column name.
        COND: Execution string, passed to EnvParser, to filter rows in `df`,
          which should return boolean Series for filtering.
        TRANS: Execution string, passed to EnvParser, to apply
          element-granularity transformation.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.
    inplace: If modified the DataFrame passed in directly.

    Return:
    --------------------
    DataFrame with transformation columns.
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))
    envp.bind_env(df)

    if not inplace:
        df = df.copy()
    for rule in rules:
        # Match transformation rule.
        if len(rule) == 2:
            cond = None
            key, trans = rule
        elif len(rule) == 3:
            key, cond, trans = rule
        else:
            logger.warning(f"Invalid rule: {rule} for transformation on DataFrame.")
            continue

        # Transform.
        if not cond:
            df[key] = envp.bind_env(df).parse(trans)
        else:
            cond_flags = envp.bind_env(df).parse(cond)
            df.loc[cond_flags, key] = (envp.bind_env(df.loc[cond_flags])
                                       .parse(trans))

    return df


# %%
def agg_on_df(
    df: pd.DataFrame,
    rules: list[tuple],
    env: Mapping = None,
    envp: EnvParser = None,
) -> pd.Series:
    """Apply aggregation on DataFrame.

    No group-by will be done inside. The aggregation will be applied on
    the whole DataFrame passed in after filtering with COND in `rules`.
    Namely, this should be used like `DataFrame.groupby.apply(agg_on_df)` in
    most cases.

    Params:
    --------------------
    df: DataFrame of data.
    rules: List of 2/3-Tuple.
      2-Tuple: [KEY, AGGS]
      3-Tuple: [KEY, COND, AGGS]
        KEY: Field name.
        COND: Execution string, passed to EnvParser, to filter rows in `df`,
          which should return boolean Series for filtering.
        AGG: Execution string, passed to EnvParser, to apply column-granularity
          aggregation.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.

    Return:
    --------------------
    Series with Index[KEY from rules].
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    ret = {}
    # set_trace()
    for rule in rules:
        # Match aggregation rule.
        if len(rule) == 2:
            cond = None
            key, agg = rule
        elif len(rule) == 3:
            key, cond, agg = rule
        else:
            logger.warning(f"Invalid rule: {rule} for aggregation on DataFrame.")
            continue

        # Aggregation.
        if not cond:
            ret[key] = envp.bind_env(df).parse(agg)
        else:
            cond_flag = envp.bind_env(df).parse(cond)
            ret[key] = envp.bind_env(df.loc[cond_flag]).parse(agg)

    if ret:
        return pd.Series(ret)
    else:
        return pd.Series(dtype=object)
