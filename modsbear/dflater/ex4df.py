#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ex4df.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2024-11-11 10:43:17
#   Description:
# ---------------------------------------------------------


# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable

import logging
import json
from collections import ChainMap
import numpy as np
import pandas as pd
# from IPython.core.debugger import set_trace

from flagbear.llp.parser import EnvParser
from modsbear.dflater.exenv import EXGINE_ENV

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


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
