#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ex2df.py
#   Author: xyy15926
#   Created: 2024-11-10 19:49:31
#   Updated: 2024-12-13 17:19:55
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
from flagbear.str2.fliper import rebuild_dict
from modsbear.dflater.exenv import EXGINE_ENV

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


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

    1. Values and Index are extracted seperately with `rebuild_dict`.
    2. The column names of returned DF must be specified explicitly along with
      the extraction rules in `val_rules`.
    3. `explode` only takes effect in value extraction, namely even both
      `val_rules` and `index_rules` brings list of the same lengths, error
      will be raised insteado of flattening the index list.
      So don't extract list in `index_rules` if `explode` is set.
    4. Make sure to set default value if possible, or the dtype of the
      DataFrame return will be Object, which will be hard to handle.

    Params:
    -------------------------
    rec: Record.
    val_rules: List of 2/3/4/5/6-Tuple of rules for value extraction.
      Extractions will be used as the value of the returned DataFrame.
      2-Tuple: [key, steps]
      3-Tuple: [key, steps, dtype]
      4-Tuple: [key, from_, steps, dtype]
      5-Tuple: [key, from_, steps, dtype, default]
      6-Tuple: [key, from_, steps, dtype, default, forced]
        key: Key in the new dict.
        from_: Dependency and source from which get the value and will be
          passed to `extract_field` as `obj.`
        steps: Steps passed to `extract_field` as `steps`.
        dtype: Dtype passed to `extract_field` as `dtype`.
        default: Default value passed to `extract_field` as `dfill`.
        forced: Forced-dtype conversion flag passed to `extract_field` as
          dforced.
      None: Keep original value.
    index_rules: List of 2/4/5-Tuple of rules for value extraction.
      Each of the extractions could be a scalar or a list with the same length
      as the value extractions. So the whole extractions could be broadcasted
      to be used as the index of the returned DataFrame.
      Note: RangeIndex will be used if None or empty list passed.
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

    # Return totally empty DF if no valid index-rule nor value-rule.
    if ((val_rules is None or len(val_rules) == 0)
            and (index_rules is None or len(index_rules) == 0)):
        logger.warning("Neither value-rules nor index-rules are specified, "
                       "empty DataFrame will be returned.")
        return pd.DataFrame()

    # Check if `rec` can't be deserialized to dict.
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON string: {rec}.")
            rec = {}

    # Extract values from `rec`.
    val_dict = (rebuild_dict(rec, val_rules, extended=True, envp=envp)
                if val_rules is not None and len(val_rules) > 0
                else {0: rec})

    # Construct DF with columns specifed by `val_rules`.
    # `vals` will always have the same columns, though in some cases that
    # no valid values extracted.
    if not val_dict:
        vals = pd.DataFrame(columns=[i[0] for i in val_rules])
    # Convert extracted values to muliple rows DataFrame.
    # Annotation: All values in `val_dict` must be list.
    elif explode and isinstance(next(iter(val_dict.values())), list):
        vals = pd.DataFrame(val_dict)
    # Convert extracted values to 1-row DataFrame.
    else:
        vals = pd.DataFrame.from_records([val_dict])

    index_arrays = []
    index_names = []
    # Extract index from `rec`.
    if index_rules is not None and len(index_rules) > 0:
        index_dict = rebuild_dict(rec, index_rules, extended=True, envp=envp)
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

    # In case that no valid index values extracted.
    if index_arrays:
        index_ = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
        vals.index = index_

    return vals


# %%
def compress_hierarchy(
    src: pd.Series[str, dict] | dict,
    confs: list,
    env: Mapping = None,
    envp: EnvParser = None,
    dropna: bool = True,
) -> pd.Series:
    """Compress the hierarchy of the records in Series.

    Extract part of the records to construct brand new Series with index
    updated step by step to compress the hierarchy of the record.
    1. Both values and index are specified step by step in `conf`, namely
      the part should be extracted step by step, which helps when the part
      of the record is nested with multi-levels and hard to extracted into
      DataFrame.
    2. So only one part should be specified step by step. Refer to
      `flat_records` for extracting multiple part once upon a time.
    3. Within one step, the index could be scalar of list with the same length
      of the values. And if no index is specified, the index for the level will
      the RangeIndex by default.

    Params:
    --------------------
    src: Series[INDEX, JSON-STRING | dict]
      Each item represents a record.
    confs: List of 2-Tuple[KEY, STEPS] of rules for value extraction.
      KEY: Key in the new dict.
      STEPS: [{content: CONTENT-STEP, key: [(KEY, KEY-STEP)]}, ]
        CONTENT-STEP: The extraction step for the values.
        KEY-STEP: The extraction step for the index.
        KEY: The index name.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    dropna: If to drop the NA in result.

    Return:
    --------------------
    Series[(idx[, idkey_1, idkey_2, ...]), partitions]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    REC2DF_COL = None
    range_index = False
    for step in confs:
        # Value rules.
        content_step = step.get("content")
        if content_step is not None:
            val_rules = [(REC2DF_COL, content_step)]
        else:
            val_rules = None
        # Index rules.
        index_rules = []
        for kname, kstep in step.get("key", []):
            if kstep == "RANGEINDEX":
                range_index = kname
            else:
                index_rules.append((kname, kstep))

        # Extract values.
        valid_values = []
        valid_index = []
        for idx, val in src.items():
            # `explode` is set to flatten the values extractions.
            val_df = rebuild_rec2df(val, val_rules, index_rules,
                                    envp=envp,
                                    explode=True,
                                    range_index=range_index)
            # In case empty DataFrame or None that represents unsuccessful
            # field extraction from records.
            if val_df is None or val_df.empty:
                continue
            valid_values.append(val_df.iloc[:, 0])
            valid_index.append(idx)

        # Save the original Index names before as the `pd.concat` may reset
        # the Index names.
        ori_index_names = src.index.names
        if len(valid_values):
            src = pd.concat(valid_values, keys=valid_index)
            # Update the index names with the result of `rebuild_rec2df`.
            ori_index_names += src.index.names[len(ori_index_names):]
            # Recover the Index names.
            src.index.set_names(ori_index_names, inplace=True)
        else:
            src = pd.Series(dtype=object)
            return src

        if dropna:
            src = src.dropna()

    return src


# %%
def flat_records(
    src: pd.Series[str, dict] | dict,
    confs: list,
    env: Mapping = None,
    envp: EnvParser = None,
    drop_rid: bool = True,
) -> pd.DataFrame:
    """Flat Series of records into DataFrame.

    Extract multiple fields from records once upon a time.
    1. Values of fields could be scalar or list of the same length. And list
      will be exploded vertically.
    2. As the `explode` is always set and no `index_rules` is provided, the
      result of `rebuild_rec2df` will always be a DataFrame with meaningless
      index, which will be dropped default.
    3. If dtype is specified, the casting function and default value in
      `REGEX_TOKEN_SPECS` will be used as default.
    4. Make sure to set default value if possible, or the dtype of the
      DataFrame return will be Object, which will be hard to handle.

    Params:
    ------------------
    src: Series[INDEX, JSON-STRING]
      Each item represents a record.
    confs: List of 2/3/4/5/6-Tuple of rules for value extraction.
      2-Tuple: [key, steps]
      3-Tuple: [key, steps, dtype]
      4-Tuple: [key, from_, steps, dtype]
      5-Tuple: [key, from_, steps, dtype, default]
      6-Tuple: [key, from_, steps, dtype, default, forced]
        key: Key in the new dict.
        from_: Dependency and source from which get the value and will be
          passed to `extract_field` as `obj.`
        steps: Steps passed to `extract_field` as `steps`.
        dtype: Dtype passed to `extract_field` as `dtype`.
        default: Default value passed to `extract_field` as `dfill`.
        forced: Forced-dtype conversion flag passed to `extract_field` as
          dforced.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    drop_rid: If to drop the meaningless last level of index created by
      `rebuild_rec2df` with `explode`.

    Return:
    ------------------
    DataFrame: Column[keys], Index[src.index]
    Exploded DataFrame: Column[keys], Index[(src.index, Range(N))]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    ret = src.apply(rebuild_rec2df,
                    val_rules=confs,
                    envp=envp,
                    explode=True)

    # In case empty result that doesn't support `pd.concat`.
    if ret.empty:
        ret = pd.DataFrame(columns=[i[0] for i in confs])
    else:
        ret = pd.concat(ret.values, keys=src.index)
        if drop_rid:
            ret = ret.droplevel(-1)

    return ret
