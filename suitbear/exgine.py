#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exgine.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2024-03-04 18:21:48
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
def flat1_max(x: pd.Series):
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


CALLS = {
    "today"     : pd.Timestamp.today(),
    "map"       : lambda x, y: x.map(y),
    "cb_fst"    : pd.Series.combine_first,
    "mon_itvl"  : lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.days / 30,
    "day_itvl"  : lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.days,
    "count"     : len,
    "sum"       : lambda x: x.sum(),
    "max"       : lambda x: x.max(),
    "min"       : lambda x: x.min(),
    "nnfilter"  : lambda x: [i for i in x if i is not None],
    "nncount"   : lambda x: len([i for i in x if i is not None]),
    "flat1_max" : flat1_max,
    "argmax"    : np.argmax,
    "argmin"    : np.argmin,
    "getn"      : lambda x, y: x.iloc[y],
    "head"      : lambda x, y: x.iloc[:y],
    "tail"      : lambda x, y: x.iloc[-y:],
}


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
      part: The name of the partition.
      level: The level of the partition, namely the granularity that the
        partition could describe.
      steps_<N>: Steps to get the level-N values(list).
      idkey_<N>: Steps of key to identify the level-N info.
        The key should be unqiue and identical at least in level-N-1 as this
        will be used as key/Index in dict DataFrame to identify `steps_<N>`.
        Attention: Multiple keys are allowed with `,` as the seperator.
    fconfs: DataFrame with Columns[key, steps, from_, dtype]
      key` and `steps` are necessary.
      from_: None will be used as default, namely the fields will be
        extracted from the original records.
      dtype: VARCHAR(255) will be used as default.

    Return:
    ----------------------
    dict[part, DataFrame of parsed result]
    """
    rets = {}
    for idx, pconf in pconfs.iterrows():
        psrc = parse_parts(src, pconf)
        part = pconf["part"]
        conf = fconfs[fconfs["part"] == part].copy()
        ret = parse_2df(psrc, conf, 0)
        rets[part] = ret
    return rets


# %%
def parse_parts(
    src: pd.Series[str, dict] | dict,
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
      level: The level of the partition, namely the granularity that the
        partition could describe.
      steps_<N>: Steps to get the level-N values(list).
      idkey_<N>: Steps of key to identify the level-N info.
        The key should be unqiue and identical at least in level-N-1 as this
        will be used as key/Index in dict DataFrame to identify `steps_<N>`.
        Attention: Multiple keys are allowed with `,` as the seperator.
      idname_<N>: Name for `idkey_<N>.

    Return:
    --------------------
    dict[(idx[, idkey_1, RANGE_1, idkey_2, RANGE_2,...]), partitions]
    or
    Series if `src` return directly.
    """
    curl = 0
    level = pconf["level"]
    envp = EnvParser()

    if isinstance(src, pd.Series):
        idx_names = [*src.index.names]
    else:
        idx_names = [None, ]

    # Parse part top-down.
    while curl < level:
        rets = {}
        steps = pconf[f"steps_{curl}"]
        # It's allowed to set multiple idkeys for one level.
        idkey_steps = pconf[f"idkey_{curl}"].split(",")
        idnames = pconf[f"idname_{curl}"].split(",")
        idx_names.extend(idnames)
        idx_names.append(f"RANGE_{len(idx_names)}")

        for idx, rec in src.items():
            if rec is None:
                vals = None
            elif isinstance(rec, str):
                try:
                    rec = json.loads(rec)
                except json.JSONDecodeError as e:
                    logger.warning(e)

            # Extract values.
            vals = extract_field(rec, steps, envp)
            if vals is None:
                continue

            # Extract idkeys from upper level.
            idkey = []
            for ele in idkey_steps:
                idk = extract_field(rec, ele, envp)
                idkey.append(idk)
            idkey = tuple(idkey)

            # Reconstruct data source with reindexing.
            if not isinstance(idx, tuple):
                idx = (idx, )
            for vidx, val in enumerate(vals):
                rets[idx + idkey + (vidx, )] = val

        src = rets
        curl += 1

    src = pd.Series(src, dtype=object)
    if not src.empty:
        src.index.set_names(idx_names, inplace=True)

    return src


# %%
def parse_2df(
    src: pd.Series,
    conf: pd.DataFrame,
    level: int = 0,
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
    conf: pd.DataFrame with Columns[key, steps, from_, dtype]
      key and `steps` are necessary.
      from_: None will be used as default, namely the fields will be
        extracted from the original records.
      dtype: VARCHAR(255) will be used as default.
      default: Default value if dtype conversion failed.
      use_default: If use the default value.
        It's common to use `None`、`np.nan` or any other value as default, so
        `use_default`, as flag, indicating if corresponsible `default` will
        take effect is necessary.
    level: The level of the DataFrame.

    Return:
    ------------------
    DataFrame for level-0: Column[keys], Index[src.index]
    DataFrame for level-1: Column[keys], Index[(src.index, Range(N))]
    """
    # Set dtype and default value.
    if "from_" not in conf:
        conf["from_"] = None
    if "dtype" not in conf:
        conf["dtype"] = "VARCHAR(255)"

    conf["dtype"] = conf["dtype"].str.upper()
    if "default" in conf and "use_default" in conf:
        rules = []
        for idx, rule_ser in conf.iterrows():
            if rule_ser["use_default"]:
                rules.append((rule_ser["key"], rule_ser["from_"],
                              rule_ser["steps"], rule_ser["dtype"],
                              rule_ser["default"]))
            else:
                rules.append((rule_ser["key"], rule_ser["from_"],
                              rule_ser["steps"], rule_ser["dtype"]))
    elif "default" in conf:
        rules = conf[["key", "from_", "steps", "dtype", "default"]].values
    else:
        rules = conf[["key", "from_", "steps", "dtype"]].values

    # Iterate over `src` to parse.
    idx_names = src.index.names
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
            # In case `vals` is a list of None or something else invalid.
            try:
                vals = pd.DataFrame(vals)
                rets[idx] = vals
            except ValueError as e:
                logger.warning(e)

    # In case no valid infos extracted from `src`.
    # For example, all fields are None when level if 2.
    if not rets:
        return pd.DataFrame(columns=conf["key"])

    if level == 0:
        dfrets = pd.DataFrame(rets).T
        dfrets.index.set_names(idx_names, inplace=True)
    else:
        dfrets = pd.concat(rets.values(), keys=rets.keys())
        dfrets.index.set_names(idx_names + [f"RANGE_{len(idx_names)}"],
                               inplace=True)

    return dfrets


# %%
def transform_part(
    df: pd.DataFrame,
    conf: pd.DataFrame,
    env: Mapping = None,
    envp: EnvParser = None,
) -> pd.DataFrame:
    """Apply transformation on DataFrame.

    Apply transformation on `df` and add the result to `df` directly.

    Params:
    --------------------
    df: DataFrame of data.
    conf: DataFrame with Columns["key", "conds", "trans"].
      key: Column name.
      conds: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      trans: Execution string, passed to EnvParser, to apply
        element-granularity transformation.
    env: Mapping to provide extra searching space, mapping reference for
      example.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.

    Return:
    --------------------
    `df` with transformation columns.
    """
    if envp is None:
        if env is None:
            envp = EnvParser(CALLS)
        else:
            envp = EnvParser(ChainMap(env, CALLS))
    envp.bind_env(df)

    for idx, item in conf.iterrows():
        key = item["key"]
        filter_ = item["conds"]
        apply = item["trans"]

        if filter_ is None:
            df[key] = envp.parse(apply)
        else:
            df.loc[envp.parse(filter_), key] = envp.parse(apply)

    return df


# %%
def agg_part(
    df: pd.DataFrame,
    conf: pd.DataFrame,
    group_key: tuple,
    env: Mapping = None,
    envp: EnvParser = None,
) -> pd.DataFrame:
    """Apply aggregation on DataFrame's groups.

    Aggregate on `df` according to `conf`.
    1. Aggregations are applied on groups grouped by index or columns alone
      indicated by `group_key`, which will be determined after testing if keys
      in `group_key` are included in columns.

    Params:
    --------------------
    df: DataFrame of data.
    conf: DataFrame with Columns["key", "conds", "aggs"].
      key: Column name.
      conds: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      aggs: Execution string, passed to EnvParser, to apply column-granularity
        aggregation.
    group_key: Group keys.
      Note: `DataFrame.groupby` will search the name of index automatically.
    env: Mapping to provide extra searching space, mapping reference for
      example.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.

    Return:
    --------------------
    DataFrame of aggregations.
    """
    if envp is None:
        if env is None:
            envp = EnvParser(CALLS)
        else:
            envp = EnvParser(ChainMap(env, CALLS))

    # Closure for apply on each group.
    def exec_agg(x):
        ret = {}
        for idx, item in conf.iterrows():
            key = item["key"]
            cond = item["conds"]
            agg = item["aggs"]

            if cond:
                cc = envp.bind_env(x).parse(cond)
                ret[key] = envp.bind_env(x[cc]).parse(agg)
            else:
                ret[key] = envp.bind_env(x).parse(agg)

        return pd.Series(ret)

    ret = df.groupby(group_key).apply(exec_agg)

    return ret


# %%
def apply_3stages(
    src: dict[str, pd.DataFrame],
    tconfs: pd.DataFrame,
    pconfs: pd.DataFrame,
    aconfs: pd.DataFrame,
    env: Mapping,
) -> dict[str, pd.DataFrame]:
    """Apply transformation and aggregation.

    1. Transform.
    2. Join.
    3. Aggragate on groups.

    Params:
    ------------------------
    src: Dict of DataFrame.
    tconfs:
      part: Part name.
      key: Column name.
      conds: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      trans: Execution string, passed to EnvParser, to apply
        element-granularity transformation.
    pconfs:
      part: Part name.
      level: The level of the partition, namely the granularity that the
        partition could describe.
      from_: Part names seperated by `,`.
      join_key: Primary keys seperated by `,` while joining and grouping.
    aconfs:
      part: Part name.
      key: Column name.
      conds: Execution string, passed to EnvParser, to filter rows in `df`,
        which should return boolean Series for filtering.
      aggs: Execution string, passed to EnvParser, to apply column-granularity
        aggregation.
    env: Mapping to provide extra searching space, mapping reference for
      example.

    Return:
    ------------------------
    dict[part, DataFrame of aggregation]
    """
    aggret = {}
    cm = ChainMap(aggret, src)
    envp = EnvParser(ChainMap(CALLS, env))

    # Apply transformation.
    for part, df in src.items():
        tconf = tconfs[tconfs["part"] == part]
        if tconf.empty:
            continue
        transform_part(df, tconf, envp=envp)

    # Apply aggregation.
    for curl in range(pconfs["level"].max(), pconfs["level"].min() - 1, -1):
        for idx, pconf in pconfs[pconfs["level"] == curl].iterrows():
            part = pconf["part"]
            # Skip if `from_` is None which indicating the `pconf` is not for
            # aggregating.
            if pd.isna(pconf["from_"]) or (not pconf["from_"]):
                continue
            from_ = pconf["from_"].split(",")

            # Join on the index.
            if len(from_) == 1:
                df = src[from_[0]]
            # Join on the determined columns.
            # Join-key could be index, as `merge` will search the index's name
            #   automatically if the index is named.
            # Attention: `from_` is presumed to be processed before, so that
            #   to get `join_key` from `jkeys` directly.
            else:
                ele = from_[0]
                df = cm[ele]
                if df.empty:
                    continue
                ljk = pconfs[pconfs["part"] == ele].iloc[0]["join_key"]
                ljk = ljk.split(",") if ljk else []
                for ele in from_[1:]:
                    rjk = pconfs[pconfs["part"] == ele].iloc[0]["join_key"]
                    rjk = rjk.split(",") if rjk else []
                    df = df.merge(cm[ele], how="outer", left_on=ljk,
                                  right_on=rjk)

            aconf = aconfs[aconfs["part"] == part]
            jk = pconf["join_key"].split(",") if pconf["join_key"] else []
            if df.empty:
                aggret[part] = pd.DataFrame(dtype=int)
            else:
                ret = agg_part(df, aconf, jk, envp=envp)
                aggret[part] = ret

    return aggret
