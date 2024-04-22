#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exgine.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2024-04-19 22:37:51
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


EXGINE_ENV = {
    "today"     : pd.Timestamp.today(),
    "map"       : _ser_map,
    "cb_fst"    : pd.Series.combine_first,
    "cb_max"    : lambda x, y: pd.concat([x, y], axis=1).max(axis=1),
    "cb_min"    : lambda x, y: pd.concat([x, y], axis=1).min(axis=1),
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
    "argmax"    : lambda x: None if len(x) == 0 else np.argmax(x),
    "argmin"    : lambda x: None if len(x) == 0 else np.argmin(x),
    "getn"      : _getn,
    "head"      : lambda x, y: x.iloc[:y],
    "tail"      : lambda x, y: x.iloc[-y:],
    "sadd"      : lambda x, y: x + y,
    "ssub"      : lambda x, y: x - y,
    "smul"      : lambda x, y: x * y,
    "sdiv"      : lambda x, y: x / y,
}


# %%
def rebuild_rec2df(
    rec: str | dict,
    val_rules: list[tuple],
    index_rules: list[tuple] = None,
    env: Mapping = None,
    envp: EnvParser = None,
    explode: bool = False,
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

    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except json.JSONDecodeError as e:
            logger.warning(e)
    val_dict = rebuild_dict(rec, val_rules, envp)

    # In case that no valid values extracted.
    if not val_dict:
        vals = pd.DataFrame(columns=[i[0] for i in val_rules])
    # Convert extracted values to DataFrame.
    elif explode and isinstance(next(iter(val_dict.values())), list):
        vals = pd.DataFrame(val_dict)
    else:
        vals = pd.Series(val_dict).to_frame().T

    if index_rules:
        index_dict = rebuild_dict(rec, index_rules, envp)
        # Convert `index_dict` into MultiIndex.
        index_arrays = []
        # set_trace()
        for ival in index_dict.values():
            if isinstance(ival, list):
                assert (len(ival) == len(vals))
                index_arrays.append(ival)
            else:
                index_arrays.append([ival] * len(vals))

        # In case that no valid values extracted.
        if index_arrays:
            index_ = pd.MultiIndex.from_arrays(index_arrays,
                                               names=index_dict.keys())
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
        if cond is None:
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
        if cond is None:
            ret[key] = envp.bind_env(df).parse(agg)
        else:
            cond_flag = envp.bind_env(df).parse(cond)
            ret[key] = envp.bind_env(df.loc[cond_flag]).parse(agg)

    return pd.Series(ret)
