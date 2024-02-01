#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exgine.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2024-02-01 20:49:48
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.parser import EnvParser
from flagbear.fliper import rebuild_dict, extract_field

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def parse_2stages(
    src: pd.Series,
    pconfs: pd.DataFrame,
    fconfs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Parse Series of records into multiple partitions.

    1. Partitions should be defined explictly in `pconfs` with its level and
       responsible steps and identical key.
    2. Partitions will be represented by DataFrame with index consisting of
       identical keys for each level.

    Params:
    ----------------------
    src: Series of records.
    pconfs: DataFrame with Column[level, steps_<N>, idkey_<N>, ...]
      level: The level of the partition, namely the granularity the partition
        could describe.
      steps_<N>: Steps to get the level-N values in list format.
      idkey_<N>: Steps of key to identify the level-N info. The key should be
        unqiue and identical at least in level-N-1 or a list with the same
        length of values derived from `step_<N>`.
    fconfs: DataFrame with Columns[key, steps, from_, dtype]
      `key` and `steps` are necessary.
      `from_`: None will be used as default, namely the fields will be
         extracted from the original records.
      `dtype`: VARCHAR(255) will be used as default.


    """
    rets = {}
    for idx, pconf in pconfs.iterrows():
        psrc = parse_parts(src, pconf)
        part = pconf["part"]
        confs = fconfs[fconfs["part"] == part]
        ret = parse_2df(psrc, confs, 0)
        rets[part] = ret
    return rets


# %%
def parse_parts(
    src: pd.Series[str, dict],
    pconf: pd.Series | dict,
) -> dict:
    """Parse Series of records into dict of partitions.

    It's hard to parse bottom-up, since bottom is hard to approach. So it's
    sensible to parse top-down, along with reindexing the whole data source to
    identify the multi-level 1-N relations of the whole and the partitions.

    Params:
    --------------------
    src: Series of records.
    pconf: Series with Index[level, steps_<N>, idkey_<N>, ...] or dict with
      similar structure.
      level: The level of the partition, namely the granularity the partition
        could describe.
      steps_<N>: Steps to get the level-N values in list format.
      idkey_<N>: Steps of key to identify the level-N info. The key should be
        unqiue and identical at least in level-N-1 or a list with the same
        length of values derived from `step_<N>`.

    Return:
    --------------------
    dict[(idx, [idkey_1,] RANGE_1, [idkey_2,] RANGE_2, ...), partitions]
    """
    curl = 0
    level = pconf["level"]
    envp = EnvParser()
    while curl < level:
        rets = {}
        steps = pconf[f"steps_{curl}"]
        idkey_steps = pconf[f"idkey_{curl}"]
        for idx, rec in src.items():
            if rec is None:
                vals = None
            elif isinstance(rec, str):
                try:
                    rec = json.loads(rec)
                except json.JSONDecodeError as e:
                    logger.warning(e)

            # Extract values and id-keys.
            vals = extract_field(rec, steps, envp)
            if vals is None:
                continue
            if idkey_steps:
                idkey = extract_field(rec, idkey_steps, envp)
            else:
                idkey = None

            # Reconstruct data source with reindexing.
            if not isinstance(idx, tuple):
                idx = (idx, )
            if idkey is None:
                for vidx, val in enumerate(vals):
                    rets[idx + (vidx,)] = val
            elif isinstance(idkey, list):
                assert len(idkey) == len(vals)
                for vidx, (idk, val) in enumerate(zip(idkey, vals)):
                    rets[idx + (idk, vidx)] = val
            else:
                for vidx, val in enumerate(vals):
                    rets[idx + (idkey, vidx)] = val

        src = rets
        curl += 1

    return src


# %%
def parse_2df(
    src: pd.Series,
    confs: pd.DataFrame,
    level: int = 0
) -> pd.DataFrame:
    """Parse Series of records into DataFrame.

    1. Only level-0 and level-1 allowed here, or it will be hard to formatted
       with DataFrame.
    2. For level-0 items, `DataFrame.from_records` will be just fine for
       records, namely only one scalar will be got for each field.
    3. For level-1 items, `pd.concat` will be called to concatenate the
       DataFrames from records.

    Params:
    ------------------
    src: pd.Series[index, json-string]
      Each item represents a record.
    confs: pd.DataFrame with Columns[key, steps, from_, dtype]
      `key` and `steps` are necessary.
      `from_`: None will be used as default, namely the fields will be
         extracted from the original records.
      `dtype`: VARCHAR(255) will be used as default.
    level: The level of the DataFrame.

    Return:
    ------------------
    DataFrame for level-0: Column[keys], Index[src.index]
    DataFrame for level-1: Column[keys], Index[(src.index, Range(N))]
    """
    if "from_" not in confs:
        confs["from_"] = None
    if "dtype" not in confs:
        confs["dtype"] = "VARCHAR(255)"

    rules = confs[["key", "from_", "steps", "dtype"]].values
    envp = EnvParser()

    rets = {}
    for idx, rec in src.items():
        if rec is None:
            continue
        elif isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except json.JSONDecodeError as e:
                logger.warning(e)
        vals = rebuild_dict(rec, rules, envp)

        if level == 0:
            rets[idx] = vals
        else:
            # In case `vals` is a list of None or something else invaid.
            try:
                vals = pd.DataFrame(vals)
                rets[idx] = vals
            except ValueError as e:
                logger.warning(e)

    # In case no valid infos extracted from `src`.
    # For example, all fields are None when level if 2.
    if not rets:
        return pd.DataFrame(columns=confs["key"])

    if level == 0:
        dfrets = pd.DataFrame(rets).T
    else:
        dfrets = pd.concat(rets.values(), keys=rets.keys())

    return dfrets
