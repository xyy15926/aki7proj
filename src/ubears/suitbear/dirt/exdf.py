#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exdf.py
#   Author: xyy15926
#   Created: 2024-11-11 17:04:03
#   Updated: 2024-12-14 23:08:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple, Any
from collections.abc import Mapping
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.dflater import ex2df, ex4df, exenv, exoptim
    from ubears.modsbear.spanner import manidf
    from ubears.suitbear.pboc import confflat, confagg, conftrans
    reload(ex2df)
    reload(ex4df)
    reload(exenv)
    reload(exoptim)
    reload(manidf)
    reload(confflat)
    reload(conftrans)
    reload(confagg)

from itertools import chain
from collections import ChainMap
from tqdm import tqdm
# from IPython.core.debugger import set_trace

from ubears.flagbear.llp.parser import EnvParser
from ubears.flagbear.str2.dtyper import stype_spec
from ubears.modsbear.spanner.manidf import merge_dfs
from ubears.modsbear.dflater.ex2df import compress_hierarchy, flat_records
from ubears.modsbear.dflater.ex4df import trans_on_df, agg_on_df
from ubears.modsbear.dflater.exenv import EXGINE_ENV
from ubears.modsbear.dflater.exoptim import compile_deps
from ubears.suitbear.kgraph.gxgine import gagg_on_dfs, GRAPH_NODE

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def flat_ft_dfs(
    src: list | pd.Series | dict[str, pd.DataFrame],
    flat_pconfs: pd.DataFrame,
    flat_fconfs: pd.DataFrame,
    how: str = "inplace",
    env: Mapping = None,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Flat from or to DFs.

    Params:
    ------------------------------
    src: Data source.
      1. Series or list of records to be flattened.
      2. Dict of DataFrames for dtype checking.
    flat_pconfs: Part-conf for flation.
    flat_fconfs: Field-conf for flation.
    how: How to store modified columns.
      inplace: Modified `df` directly.
      copy: Make a copy of `df` to store transformed columns.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.

    Return:
    ---------------------------
    Dict[part-name, DataFrame of values of parts]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    # Unify list to pd.Series for flation.
    if isinstance(src, list):
        src = pd.Series(src)

    # Flat unstructured and convert dtypes for pd.Series.
    if isinstance(src, pd.Series):
        # Unify dtypes and defaults for `rebuild_rec2df`.
        fconfs = flat_fconfs.copy()
        fconfs["from_"] = None
        fconfs["dforced"] = True
        if "default" not in fconfs:
            fconfs["default"] = None

        # Compress hierarchy and then extract fields.
        flat_rets = {}
        for idx, pconf in flat_pconfs.iterrows():
            part_name = pconf["part"]
            psrc = compress_hierarchy(src, pconf["steps"])
            fconfs_ = fconfs.loc[
                fconfs["part"] == part_name,
                ["key", "from_", "step", "dtype", "default", "dforced"]]
            flat_ret = flat_records(psrc, fconfs_.values)
            flat_rets[pconf["part"]] = flat_ret
    # Check and covert dtype for columns in DFs.
    elif isinstance(src, dict):
        flat_rets = {}
        for idx, pconf in flat_pconfs.iterrows():
            part_name = pconf["part"]
            if part_name not in src:
                logger.info(f"{part_name} can't be found in DataFrames.")
                continue
            # Copy if not to covnert dtypes inplace.
            df = (src[part_name] if how == "inplace"
                  else src["part_name"].copy())
            fconfs_ = flat_fconfs.loc[flat_fconfs["part"] == part_name]
            for idx, fconf in fconfs_.iterrows():
                field_name = fconf["key"]
                dtype = fconf["dtype"]
                nptype = stype_spec(dtype, "nptype")
                if field_name not in df:
                    logger.info(f"{field_name} can't be found in `{part_name}`.")
                    continue
                df[field_name] = df[field_name].astype(nptype)

            flat_rets[part_name] = df

    return flat_rets


# %%
def trans_from_dfs(
    dfs: dict[str, pd.DataFrame],
    trans_pconfs: pd.DataFrame,
    trans_fconfs: pd.DataFrame,
    how: str = "auto",
    env: Mapping = None,
    envp: EnvParser = None,
    *,
    trans_rets: dict = None,
) -> dict[str, pd.DataFrame]:
    """Apply aggregations on DataFrames.

    1. `trans_pconfs` will be treated as the index to determine the order of
      the transformations.
    2. As the transformations are defined on the DataFrames step by step, the
      group keys could be pre-determined at table-granularity in
      `trans_pconfs`.

    Params:
    ------------------------
    dfs: Dict of DataFrame.
    trans_pconfs: Part-conf for aggregations.
    trans_fconfs: Field-conf for aggregations.
    how: How to store transformed columns.
      inplace: Modified `df` directly.
      new: Create a new DataFrame to store transformed columns.
      copy: Make a copy of `df` to store transformed columns.
      auto: Determine transformation type automatically.
        `inplace`: For transformation on single DF with the same name.
        `copy`: For transformation on single DF with the different name.
        `new`: For transformation on multiple DFs.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    trans_rets:

    Return:
    ------------------------
    Dict[part, DataFrame of transformation]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    trans_rets = {} if trans_rets is None else trans_rets
    # Both the source DataFrames and results in process will be used as the
    # searching space.
    df_space = ChainMap(trans_rets, dfs)
    # Apply transformation according to `trans_pconfs`.
    pbar_rows = tqdm(trans_pconfs.iterrows())
    for idx, pconf in pbar_rows:
        part_name = pconf["part"]
        pbar_rows.set_description(f"{part_name} transformation.")

        # Determine transformation type.
        if how == "auto":
            if len(pconf["from_"]) == 1:
                trans_how = ("inplace" if pconf["from_"][0] == part_name
                             else "copy")
            else:
                trans_how = "new"
        else:
            trans_how = how

        join_key = pconf["joinkey"]
        if join_key:
            # All the index's names and columns are set properly from flation
            # to transformation, so the join keys will be always exist and
            # `merge_dfs` could be called directly and empty dataframes are
            # no longer problems.
            joined_df = merge_dfs([df_space[dn] for dn in pconf["from_"]],
                                  hows="left", ons=join_key)
        else:
            joined_df = dfs[pconf["from_"][0]]

        # Transform.
        trans_rules = trans_fconfs.loc[trans_fconfs["part"] == part_name,
                                       ["key", "cond", "trans"]].values
        tdf = trans_on_df(joined_df, trans_rules, how=trans_how, envp=envp)

        # Concat transformation result if `new` is specified, which often ocurrs
        # when combining transformations result from multiple DFs together.
        if trans_how == "new" and part_name in trans_rets:
            trans_rets[part_name] = pd.concat([trans_rets[part_name], tdf],
                                              axis=1)
        # Or override the original DF directly.
        else:
            trans_rets[part_name] = tdf

    return trans_rets


# %%
def agg_from_dfs(
    dfs: dict[str, pd.DataFrame],
    agg_pconfs: pd.DataFrame,
    agg_fconfs: pd.DataFrame,
    env: Mapping = None,
    envp: EnvParser = None,
    *,
    agg_rets: dict[str, pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """Apply aggregations on DataFrames.

    1. `agg_pconfs` will be treated as the index to determine the order of the
      aggregations.
    2. As the aggregations are defined on the DataFrames step by step, the
      group keys could be pre-determined at table-granularity in `agg_pconfs`.

    Params:
    ------------------------
    dfs: Dict of DataFrame.
    agg_pconfs: Part-conf for aggregations.
    agg_fconfs: Field-conf for aggregations.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    agg_rets:

    Return:
    ------------------------
    Dict[part, DataFrame of aggregation]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    agg_rets = {} if agg_rets is None else agg_rets
    # Both the source DataFrames and results in process will be used as the
    # searching space.
    df_space = ChainMap(agg_rets, dfs)
    # Aggregate in `level`-decending order, namely bottom-up, so the
    # `agg_pconfs` should be sorted manually first.
    for idx, pconf in agg_pconfs.iterrows():
        part_name = pconf["part"]
        if part_name in agg_rets:
            continue
        prikey = pconf["prikey"]
        join_key = pconf["joinkey"]
        if join_key:
            # All the index's names and columns are set properly from flation
            # to transformation, so the join keys will be always exist and
            # `merge_dfs` could be called directly and empty dataframes are
            # no longer problems.
            joined_df = merge_dfs([df_space[dn] for dn in pconf["from_"]],
                                  hows="left", ons=join_key)
        else:
            joined_df = dfs[pconf["from_"][0]]

        # Aggregate.
        agg_rules = agg_fconfs.loc[agg_fconfs["part"] == part_name,
                                   ["key", "cond", "agg"]].values
        tqdm.pandas(desc=part_name)
        agg_ret = (joined_df.groupby(prikey)
                   .progress_apply(agg_on_df, rules=agg_rules, envp=envp))
        agg_rets[part_name] = agg_ret

    return agg_rets


# %%
def agg_from_graphdf(
    dfs: dict[str, pd.DataFrame],
    gpconfs: pd.DataFrame,
    gfconfs: pd.DataFrame,
    env: Mapping = None,
    envp: EnvParser = None,
    *,
    agg_rets: dict[str, pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate from DataFrame representing the knowledge graph.

    Params:
    -----------------------------
    dfs: Dict of DataFrame as searching space.
    gpconfs: Part confs for aggregation on graph.
    gfconfs: Field confs for aggregation on graph.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    agg_rets: Dict to store the aggregation result, which will be updated and
      returned.

    Return:
    -----------------------------
    DataFrame of aggregation result of all nodes.
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    node_df = dfs[GRAPH_NODE]
    agg_rets = {} if agg_rets is None else agg_rets
    # No additional sortation is applied here.
    # So ensure `pconf` rows in `gpconfs` are preorder logically.
    for idx, pconf in gpconfs.iterrows():
        part_name = pconf["part"]
        ntype = pconf["ntype"]
        # Select nodes from `node_df` of specified types.
        nodes = node_df.loc[node_df["ntype"].isin(ntype), "nid"]
        if part_name in agg_rets:
            continue

        agg_rules = gfconfs.loc[gfconfs["part"] == part_name,
                                ["key", "cond", "agg"]].values
        tqdm.pandas(desc=part_name)
        # Apply `gagg_on_dfs` for each selected node.
        agg_ret = nodes.progress_apply(gagg_on_dfs, dfs=dfs, rules=agg_rules,
                                       envp=envp)
        # Set node-id as index.
        agg_rets[part_name] = agg_ret.set_index("nid", drop=True)

    return agg_rets


# %%
def dep_from_fconfs(
    targets: list[str],
    fconfs: pd.DataFrame,
    envp: EnvParser = None,
) -> pd.DataFrame:
    """Get the dependences for the targets.

    Params:
    ---------------------------
    target: Target keys.
    fconfs: Field confs of condition, aggregation and transformation, with
      columns[key, cond, trans, agg].
    envp: EnvParser to compile condition, aggregation and transformation to
      determine the dependences.

    Return:
    ---------------------------
    Field confs of dependences.
    """
    envp = EnvParser() if envp is None else envp

    dep_keys = set(targets)
    # Loop until no more dependences updates.
    while len(targets):
        rule_df = fconfs.loc[fconfs["key"].isin(targets)]
        rules = rule_df[["key", "cond", "trans", "agg"]].values
        dep_dict = compile_deps(rules, envp=envp)
        targets = list(chain.from_iterable(dep_dict.values()))
        targets = set(targets) - dep_keys
        dep_keys.update(targets)

    fconfs = fconfs[fconfs["key"].isin(dep_keys)]
    return fconfs
