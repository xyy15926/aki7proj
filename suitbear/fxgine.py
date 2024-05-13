#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: fxgine.py
#   Author: xyy15926
#   Created: 2024-04-19 14:52:59
#   Updated: 2024-05-12 14:42:36
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Mapping
from collections import ChainMap
from itertools import product
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.parser import EnvParser
from flagbear.fliper import rebuild_dict, extract_field
from flagbear.fliper import regex_caster
from flagbear.exgine import rebuild_rec2df, trans_on_df, agg_on_df, EXGINE_ENV

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# TODO: Index setting, RangeIndex for example.
def compress_hierarchy(
    src: pd.Series[str, dict] | dict,
    conf: pd.Series | dict,
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
    conf: Dict of with keys[steps_<N>, idkey_<N>, idname_<N>,...] or Series
      with similar Index.
      steps_<N>: Steps to get the level-N values(list).
        Null indicates to not extract any field.
      idkey_<N>: Steps of key to identify the level-N info.
        The key should be unqiue and identical at least in level-N-1 as this
        will be used as key/Index in dict DataFrame to identify `steps_<N>`.
        Attention: Multiple keys are allowed with `,` as the seperator.
      idname_<N>: Names for `idkey_<N>.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.

    Return:
    --------------------
    Series[(idx[, idkey_1, idkey_2, ...]), partitions]
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    cur_lv = 0
    while f"steps_{cur_lv}" in conf:
        val_steps = conf[f"steps_{cur_lv}"]
        # Don't extract any fields.
        if pd.isna(val_steps):
            val_rules = None
        else:
            val_rules = [(None, conf[f"steps_{cur_lv}"])]
        index_rules = []
        range_index = False

        if f"idkey_{cur_lv}" in conf:
            for idname, idkey in zip(conf[f"idname_{cur_lv}"].split(","),
                                     conf[f"idkey_{cur_lv}"].split(",")):
                idname = idname.strip()
                idkey = idkey.strip()
                # Add an `RangeIndex` to the result.
                if idkey == "RANGEINDEX":
                    range_index = idname
                else:
                    index_rules.append((idname, idkey))

        # Never `pd.DF.apply`.
        # psrc = src.apply(rebuild_rec2df,
        #                  val_rules=val_rules,
        #                  index_rules=index_rules,
        #                  envp=envp,
        #                  explode=True,
        #                  range_index=range_index)

        valid_values = []
        valid_index = []
        for idx, val in src.iteritems():
            val_df = rebuild_rec2df(val, val_rules, index_rules,
                                    envp=envp,
                                    explode=True,
                                    range_index=range_index)
            # In case empty DataFrame or None that represents unsuccessful
            # field extraction from records.
            if val_df is None or val_df.empty:
                continue
            valid_values.append(val_df)
            valid_index.append(idx)

        # Save the original Index names before.
        ori_index_names = src.index.names
        src = pd.concat(valid_values, keys=valid_index)[None]

        # Recover the Index names.
        ori_index_names += src.index.names[len(ori_index_names):]
        src.index.set_names(ori_index_names, inplace=True)

        if dropna:
            src = src.dropna()

        cur_lv += 1

    return src


# %%
def flat_records(
    src: pd.Series[str, dict] | dict,
    confs: pd.DataFrame,
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

    Params:
    ------------------
    src: Series[INDEX, JSON-STRING]
      Each item represents a record.
    confs: DataFrame[key, steps, from_, dtype, default, use_default].
      key: Field name.
      steps: Steps to get values from `src`.
        `key` and `steps` are necessary.
      from_: The original field from where to extract current field.
        None will be used as default, namely the fields will be extracted
          from the original records.
      dtype: VARCHAR(255) will be used as default.
      default: Default value if dtype conversion failed.
      use_default: If use the default value.
        It's common to use `None`、`np.nan` or any other value as default, so
        `use_default`, as flag, indicating if corresponsible `default` will
        take effect is necessary.
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
    confs = confs.copy()
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    if "from_" not in confs:
        confs["from_"] = None

    if "dtype" not in confs:
        confs["dtype"] = "VARCHAR(255)"
    confs["dtype"] = confs["dtype"].str.upper()

    if "default" in confs and "use_default" in confs:
        confs.loc[~confs["use_default"].astype(bool), "default"] = None

    if "default" in confs:
        rules = confs[["key", "from_", "steps", "dtype", "default"]].values
    else:
        rules = confs[["key", "from_", "steps", "dtype"]].values

    ret = src.apply(rebuild_rec2df,
                    val_rules=rules,
                    envp=envp,
                    explode=True)

    # In case empty result that doesn't support `pd.concat`.
    if ret.empty:
        ret = pd.DataFrame()
    else:
        ret = pd.concat(ret.values, keys=src.index)
        if drop_rid:
            ret = ret.droplevel(-1)

    return ret


# %%
def agg_from_dfs(
    src: dict[str, pd.DataFrame],
    part_confs: pd.DataFrame,
    agg_confs: pd.DataFrame,
    trans_confs: pd.DataFrame = None,
    env: Mapping = None,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Apply aggregations on DataFrames.

    1. `part_confs` will be treated as the index to determine the order of the
      transformations and aggregations.
    2. As the aggregations are defined on the DataFrames step by step, the
      group keys could be pre-determined at table-granularity in `part_confs`
      and should be the primary key and join key for aggregations result.

    Params:
    ------------------------
    src: Dict of DataFrame.
    part_confs: DataFrame with C[part, level, from_, prikey].
      part: Part name.
      from_: Part names seperated by `,`.
      level: The level of the partition, namely the granularity that the
        partition represents, which determines the order of precess.
      join_key: Primary keys seperated by `,` while joining and grouping.
    agg_conf: DataFrame with C[part, key, cond, agg].
      part: Part name of the key.
      key: Field name.
      cond: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      agg: Execution string, passed to EnvParser, to apply column-granularity
        aggregation.
    trans_confs: DataFrame with C[part, key, cond, trans].
      part: Part name of the key.
      key: Column name.
      cond: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      trans: Execution string, passed to EnvParser, to apply
        element-granularity transformation.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.

    Return:
    ------------------------
    Dict[part, DataFrame of aggregation]
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    # 1. Apply transformation on DataFrames in `src`.
    if trans_confs is not None:
        for part_name, df in src.items():
            trans_rules = trans_confs.loc[trans_confs["part"] == part_name,
                                          ["key", "cond", "trans"]].values
            if trans_rules.size:
                src[part_name] = trans_on_df(df, trans_rules, envp=envp)

    agg_ret = {}
    # Both the source DataFrames and results in process will be used as the
    # searching space.
    df_space = ChainMap(agg_ret, src)
    # 2. Aggregate in `level`-decending order, namely bottom-up.
    for idx, pconf in part_confs.sort_values(by="level",
                                             ascending=False).iterrows():
        part_name = pconf["part"]
        from_ = pconf["from_"]
        prikey = [i.strip() for i in pconf["prikey"].split(",")]
        # Empty `from_` indicates invalid aggregation config.
        if pd.isna(from_) or (not from_):
            continue
        from_ = [i.strip() for i in from_.split(",")]

        # if part_name == "acc_special_accd":
        #     set_trace()

        # 2.1 Prepare DataFrame for aggregation.
        joined_df = df_space[from_[0]]
        if joined_df.empty:
            agg_ret[part_name] = pd.DataFrame()
            continue

        # The group-key for aggregation must be the join-key of the result
        # DataFrame.
        left_jk = part_confs.loc[part_confs["part"] == from_[0],
                                 "prikey"].iloc[0]
        left_jk = [i.strip() for i in left_jk.split(",")]

        # Join the DataFrames indicated by `from_` one-by-one.
        for rpart_name in from_[1:]:
            right_jk = part_confs.loc[part_confs["part"] == rpart_name,
                                      "prikey"].iloc[0]
            right_jk = [i.strip() for i in right_jk.split(",")]
            joined_df = joined_df.merge(df_space[rpart_name],
                                        how="outer",
                                        left_on=left_jk,
                                        right_on=right_jk)

        # 2.2 Aggregate.
        agg_rules = agg_confs.loc[agg_confs["part"] == part_name,
                                  ["key", "cond", "agg"]].values
        agg_ret[part_name] = (joined_df.groupby(prikey)
                              .apply(agg_on_df,
                                     rules=agg_rules,
                                     envp=envp))
    return agg_ret
