#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: exdf.py
#   Author: xyy15926
#   Created: 2024-11-11 17:04:03
#   Updated: 2024-11-11 18:34:48
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
from collections.abc import Mapping
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from modsbear.dflater import ex2df, ex4df, exenv
    from modsbear.spanner import manidf
    from suitbear.pboc import confflat, confagg, conftrans
    reload(ex2df)
    reload(ex4df)
    reload(exenv)
    reload(manidf)
    reload(confflat)
    reload(conftrans)
    reload(confagg)

from collections import ChainMap
from tqdm import tqdm
# from IPython.core.debugger import set_trace

from flagbear.llp.parser import EnvParser
from modsbear.spanner.manidf import merge_dfs
from modsbear.dflater.ex4df import agg_on_df
from modsbear.dflater.exenv import EXGINE_ENV
from suitbear.kgraph.gxgine import gagg_on_dfs, GRAPH_NODE

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


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
      transformations and aggregations.
    2. As the aggregations are defined on the DataFrames step by step, the
      group keys could be pre-determined at table-granularity in `agg_pconfs`
      and should be the primary key and join key for aggregations result.

    Params:
    ------------------------
    dfs: Dict of DataFrame.
    agg_pconfs: Part-conf for aggregations.
    agg_fconfs: Field-conf for aggregations.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    agg_rets:
    engine:
    table_prefix:

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
    # 2. Aggregate in `level`-decending order, namely bottom-up.
    for idx, pconf in agg_pconfs.iterrows():
        part_name = pconf["part"]
        if part_name in agg_rets:
            continue
        prikey = pconf["prikey"]
        join_key = pconf["joinkey"]
        if join_key:
            jdfs = []
            # Skip the aggregation if any source DataFrame is empty.
            for dn in pconf["from_"]:
                if df_space[dn].empty:
                    joined_df = pd.DataFrame()
                    break
                jdfs.append(df_space[dn])
            else:
                joined_df = merge_dfs(jdfs, hows="left", ons=join_key)
        else:
            joined_df = dfs[pconf["from_"][0]]

        # Skip the aggregation if merged DataFrame is empty.
        if joined_df.empty:
            agg_rets[part_name] = pd.DataFrame()
            continue

        # 2.2 Aggregate.
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
        agg_rets[part_name] = agg_ret

    return agg_rets
