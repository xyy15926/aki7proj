#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: gxgine.py
#   Author: xyy15926
#   Created: 2024-10-23 20:30:03
#   Updated: 2024-11-03 20:44:52
#   Description:
# ---------------------------------------------------------


# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable
from collections import ChainMap
import logging

import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.parser import EnvParser
from modsbear.exgine import EXGINE_ENV
from suitbear.kgraph.kgenum import NodeType, ROLE_TYPE_MAPPER

# Default DataFrame key name in searching space for relations and nodes.
GRAPH_REL = "GRAPH_REL"
GRAPH_NODE = "GRAPH_NODE"

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def gagg_on_dfs(
    nid: Any,
    dfs: dict[str, pd.DataFrame],
    rules: list[tuple],
    env: Mapping = None,
    envp: EnvParser = None,
) -> pd.Series:
    """Aggregate on DataFrame indexed with dict to imitate the graph.

    1. There are 3 steps for just 1-degree's condition application.
    1.1 Locate all relations for selected nodes.
    1.2 Filter relations.
    1.3 Filter related nodes.

    *NOTE: The aggregations will be applied on the nodes.

    Params:
    --------------------------
    nid: Seed ids of the node.
    dfs: Dict of DataFrames as the searching space. Some mandatory elements:
      GRAPH_REL: Relation Dataframe with columns:
        source: Source node id.
        target: Target node id.
        source_role: Source node role.
        target_role: Target node role.
      GRAPH_NODE: Node index DataFrame, which will be used to locate the seed
        nodes and determine their types.
        nid: Node id.
        ntype: Node type.
      NodeTypes: DataFrames for each NodeTypes, since the attrs may be
        different for different NodeTypes.
    rules: List of 2/3-Tuple.
      2-Tuple: [KEY, AGGS]
      3-Tuple: [KEY, COND, AGGS]
        KEY: Field name.
        COND: List of 3-Tuple[DIRECTION, REL_COND, NODE_COND]
          DIRECTION: "source", "target" or "both" determining whether the
            nodes are the source or the target.
          REL_COND: Execution string, passed to EnvParser, to filter relations,
            which should return boolean Series for filtering.
          NODE_COND: Execution string, passed to EnvParser, to filter nodes,
            which should return boolean Series for filtering.
        AGG: Execution string, passed to EnvParser, to apply column-granularity
          aggregation on DataFrame of nodes.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.

    Return:
    --------------------------
    pd.Series with `nid` and the aggregation results as the index.
    """
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    node_df = dfs[GRAPH_NODE]
    rel_df = dfs[GRAPH_REL]
    ntype = node_df.loc[node_df["nid"] == nid, "ntype"]
    if ntype.empty:
        return pd.Series(dtype="object")
    snodes = dfs[ntype.iloc[0]].loc[dfs[ntype.iloc[0]]["nid"] == nid]
    # The settings for determining the nodes in filtered relations.
    # FITER-TYPE: [(CUR-NODE-FIELD in rel_df, TARGET-NODE-FIELD in rel_df,
    #               NODE-ROLE-FIELD in rel_df)]
    role_sets = {
        "source": [("source", "target", "target_role")],
        "target": [("target", "source", "source_role")],
        "both": [("source", "target", "target_role"),
                 ("target", "source", "source_role")],
    }

    ret = {}
    for rule in rules:
        cur_nodes = snodes
        # Match aggregation rule.
        if len(rule) == 2:
            conds = []
            key, agg = rule
        elif len(rule) == 3:
            key, conds, agg = rule
        else:
            logger.warning(f"Invalid rule: {rule} for aggregation on DataFrame.")
            continue

        # rel_cond: cond-repr for relations-DF
        # node_cond: cond-repr for nodes-DF
        for direction, rel_cond, node_cond in conds:
            all_nodes = []
            for cur_nid, tgt_nid, role_field in role_sets[direction]:
                # 1. Filter edges with current nodes' id.
                cur_rel = rel_df[rel_df[cur_nid].isin(cur_nodes["nid"])]
                # 2. Filter edges with the `rel_cond`.
                if rel_cond:
                    rel_flag = envp.bind_env(cur_rel).parse(rel_cond)
                    cur_rel = cur_rel.loc[rel_flag, [tgt_nid, role_field]]
                # 3. Filter nodes with the `node_cond`.
                if cur_rel.shape[0] > 0:
                    # Determine the node-type and correspondant DataFrame
                    # of the node-attrs.
                    role_type = cur_rel.iloc[0, 1]
                    ntype = ROLE_TYPE_MAPPER[role_type][-1]
                    one_nodes = dfs[ntype]
                    # Locate the related nodes.
                    one_nodes = one_nodes[one_nodes["nid"]
                                          .isin(cur_rel.iloc[:, 0])]
                    # Filter nodes with the `node_cond`.
                    if node_cond:
                        node_flag = envp.bind_env(one_nodes).parse(node_cond)
                        one_nodes = one_nodes[node_flag]
                    all_nodes.append(one_nodes)
                else:
                    all_nodes.append(
                        pd.DataFrame(columns=cur_nodes.columns))
            # Drop duplicate nodes.
            cur_nodes = pd.concat(all_nodes, axis=0).drop_duplicates("nid")
        # Apply agggregation on the final nodes.
        ret[key] = envp.bind_env(cur_nodes).parse(agg)

    ret["nid"] = nid
    if ret:
        return pd.Series(ret)
    else:
        return pd.Series(dtype=object)
