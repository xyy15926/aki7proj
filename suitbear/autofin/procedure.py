#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-10-06 15:02:13
#   Updated: 2024-11-04 11:04:45
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
    from flagbear import patterns, parser, fliper
    from modsbear import exgine
    from suitbear import finer
    from suitbear.autofin import confflat, conftrans
    from suitbear.autofin import confagg
    from suitbear.kgraph import kgenum, afrels, pbocrels
    from suitbear.autofin import graphagg
    from suitbear.kgraph import gxgine
    reload(patterns)
    reload(parser)
    reload(fliper)
    reload(exgine)
    reload(finer)
    reload(confflat)
    reload(conftrans)
    reload(confagg)
    reload(kgenum)
    reload(afrels)
    reload(pbocrels)
    reload(graphagg)
    reload(gxgine)

import os
from collections import ChainMap
import sqlalchemy as sa
from tqdm import tqdm
from IPython.core.debugger import set_trace

from flagbear.patterns import REGEX_TOKEN_SPECS
from flagbear.parser import EnvParser
from flagbear.fliper import extract_field
from modsbear.exgine import (agg_on_df,
                             compress_hierarchy,
                             flat_records,
                             trans_on_df,
                             EXGINE_ENV)
from suitbear.finer import (tmp_file,
                            save_with_excel,
                            load_from_pickle,
                            get_tmp_path,
                            get_assets_path)

from suitbear.autofin.confflat import df_flat_confs
from suitbear.autofin.conftrans import (df_trans_confs,
                                        TRANS_ENV,
                                        TRANS_CONF,
                                        MAPPERS,
                                        MAPPERS_CHN,
                                        MAPPERS_CODE)
from suitbear.autofin.confagg import (df_agg_confs,
                                      PERSONAL_CONF,
                                      MASS_CONF)
from suitbear.kgraph.kgenum import (NodeType,
                                    RoleType,
                                    df_enum_confs,
                                    ROLE_TYPE_MAPPER)
from suitbear.autofin.graphagg import (df_graph_agg_confs,
                                       GRAPH_REL,
                                       GRAPH_NODE)
from suitbear.kgraph.gxgine import gagg_on_dfs

TRANS_ENV["today"] = pd.Timestamp.today()
AUTOFIN_AGG_CONF = {**PERSONAL_CONF, **MASS_CONF}

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def write_autofin_confs(conf_file: str):
    """Write AutoFin parsing confs to Excel.
    """
    dfs = {}
    flat_pconfs, flat_fconfs = df_flat_confs()
    dfs["autofin_flat_parts"] = flat_pconfs
    dfs["autofin_flat_fields"] = flat_fconfs

    dfs["autofin_vars_trans"] = df_trans_confs()
    mappers = pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()], axis=0,
                        keys=MAPPERS.keys())
    dfs["autofin_vars_maps"] = mappers

    agg_pconfs, agg_fconfs = df_agg_confs(AUTOFIN_AGG_CONF)
    dfs["autofin_vars_parts"] = agg_pconfs
    dfs["autofin_vars_aggs"] = agg_fconfs

    dfs["graph_enum_types"] = df_enum_confs()

    pboc_pconfs, pboc_node_confs, pboc_rel_confs = pbocrels.df_graph_confs()
    dfs["pboc_graph_parts"] = pboc_pconfs
    dfs["pboc_graph_rels"] = pboc_rel_confs
    dfs["pboc_graph_nodes"] = pboc_node_confs

    af_gconfs, af_rel_confs, af_node_confs = afrels.df_graph_confs()
    dfs["autofin_graph_parts"] = af_gconfs
    dfs["autofin_graph_rels"] = af_rel_confs
    dfs["autofin_graph_nodes"] = af_node_confs

    graph_agg_pconfs, graph_agg_fconfs = df_graph_agg_confs()
    dfs["graph_var_parts"] = graph_agg_pconfs
    dfs["graph_var_aggs"] = graph_agg_fconfs

    return save_with_excel(dfs, conf_file)


# %%
def agg_from_dfs(
    dfs: dict[str, pd.DataFrame],
    agg_pconfs: pd.DataFrame,
    agg_fconfs: pd.DataFrame,
    env: Mapping = TRANS_ENV,
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
    agg_rets: Dict to store the aggregation result, which will be updated and
      returned.

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
    # 2. Aggregate in original order, since confs with different granularity
    # may share the same level as the level-0 may be both for person and group.
    for idx, pconf in agg_pconfs.iterrows():
        part_name = pconf["part"]
        from_ = pconf["from_"]
        join_key = pconf.get("joinkey")
        if not isinstance(join_key, (tuple, list)):
            join_key = [None]
        prikey = pconf["prikey"]

        # Skip if the result already exists
        # Or no aggregation should be applied as the empty `from_` indicates.
        if (part_name in agg_rets) or not isinstance(join_key, (tuple, list)):
            continue

        # 2.1 Prepare DataFrame for aggregation.
        joined_df = df_space[from_[0]]
        ljoin_key = join_key[0]
        if joined_df.empty:
            agg_rets[part_name] = pd.DataFrame()
            continue

        # Join the DataFrames indicated by `from_` one-by-one.
        for rpart_name, rjoin_key in zip(from_[1:], join_key[1:]):
            joined_df = joined_df.merge(df_space[rpart_name],
                                        how="left",
                                        left_on=ljoin_key,
                                        right_on=rjoin_key)

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
    env: Mapping = TRANS_ENV,
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


# %%
def autofin_vars(
    dfs: dict[str, pd.DataFrame],
    agg_key_mark: pd.Series | set | list = None,
    env: Mapping = TRANS_ENV,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Calculate index from PBOC records.

    Params:
    ---------------------------
    dfs: Series of PBOC records.
    agg_key_mark: The necessary aggregation to be calculated.

    Return:
    ---------------------------
    Dict[part-name, aggregation result]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    # 1. Apply transformation on DataFrames in `dfs`.
    for part_name, conf in TRANS_CONF.items():
        trans_rules = [(key, cond, trans) for key, trans, cond, desc
                       in conf["trans"]]
        df = dfs[part_name]

        # Prepare dataframe before transformation.
        pre_trans = conf.get("pre_trans")
        if pre_trans is not None:
            df = pre_trans(df)

        if not df.empty:
            dfs[part_name] = trans_on_df(df, trans_rules, envp=envp)

    # 2. Construct and apply aggregation config.
    agg_rets = {}
    agg_pconfs, agg_fconfs = df_agg_confs(AUTOFIN_AGG_CONF)
    if agg_key_mark is not None:
        agg_fconfs = agg_fconfs[agg_fconfs["key"].isin(agg_key_mark)]
    agg_from_dfs(dfs, agg_pconfs, agg_fconfs, envp=envp, agg_rets=agg_rets)

    # 3. Construct and apply graph aggregations.
    gpconfs, gfconfs = df_graph_agg_confs()
    if agg_key_mark is not None:
        gfconfs = gfconfs[gfconfs["key"].isin(agg_key_mark)]
    agg_from_graphdf(dfs, gpconfs, gfconfs, envp=envp, agg_rets=agg_rets)

    return dfs, agg_rets


# %%
if __name__ == "__main__":
    write_autofin_confs("autofin/autofin_confs.xlsx")
    mock_file = get_assets_path() / "autofin/autofin_mock_20241101.xlsx"
    TRANS_ENV["today"] = pd.Timestamp("20241101")
    fname = get_tmp_path() / "infocode_cats_latest.xlsx"
    df = pd.read_excel(fname)
    TRANS_ENV["infocode_map"] = df.set_index("infocode")["cats"]

    # Prepare mock data.
    xlr = pd.ExcelFile(mock_file)
    dfs = {}
    for shname in xlr.sheet_names:
        dfs[shname] = pd.read_excel(xlr, sheet_name=shname)
    afrel_df, afnode_df = afrels.build_graph_df(dfs)
    pbrel_df, pbnode_df = pbocrels.build_graph_df(dfs)
    rel_df = (pd.concat([afrel_df, pbrel_df])
              .sort_values("update")
              .drop_duplicates(subset=["source", "target"], keep="last"))
    node_df = (pd.concat([afnode_df, pbnode_df])
               .drop_duplicates())
    dfs["GRAPH_REL"] = rel_df
    dfs["GRAPH_NODE"] = node_df
    for ele in NodeType:
        dfs[ele.value] = node_df

    dfs, agg_rets = autofin_vars(dfs)
    save_with_excel(agg_rets, "autofin/agg_rets")
    save_with_excel(dfs, "autofin/dfs")

    import networkx as nx
    from pyvis.network import Network
    rel_dfc = rel_df.copy()
    rel_dfc["update"] = rel_df["update"].dt.strftime("%Y%m%d")
    dig = nx.from_pandas_edgelist(rel_dfc, source="source",
                                  target="target",
                                  edge_attr=True)
    net = Network()
    net.from_nx(dig)
    fname = tmp_file("autofin/autofin_rels_graph.html")
    # `Network` only support reltive path.
    net.save_graph(os.path.relpath(fname))
