#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ex4df.py
#   Author: xyy15926
#   Created: 2024-01-24 10:30:18
#   Updated: 2025-06-14 20:14:19
#   Description:
# ---------------------------------------------------------


# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Mapping, Sequence, Callable

import logging
import json
from collections import ChainMap
from functools import lru_cache
import numpy as np
import pandas as pd
# from IPython.core.debugger import set_trace

from ubears.flagbear.llp.parser import EnvParser

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
    how: str = "inplace",
    env: Mapping = None,
    envp: EnvParser = None,
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
    how: How to store transformed columns.
      inplace: Modified `df` directly.
      new: Create a new DataFrame to store transformed columns.
      copy: Make a copy of `df` to store transformed columns.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored is `envp` is passed.

    Return:
    --------------------
    DataFrame with transformation columns.
    """
    envp = EnvParser(env) if envp is None else envp
    envp.bind_env(df)

    # Make a copy if transformation is not applied on `df`.
    if how != "inplace":
        new = df.copy()
    else:
        new = df
    tcols = []

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

        tcols.append(key)
        # Transform.
        if not cond:
            new[key] = envp.bind_env(new).parse(trans)
        else:
            cond_flags = envp.bind_env(new).parse(cond)
            new.loc[cond_flags, key] = (envp.bind_env(new.loc[cond_flags])
                                        .parse(trans))

    # Return only tranformed columns.
    if how == "new":
        return new.loc[:, tcols]
    else:
        return new


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
    envp = EnvParser(env) if envp is None else envp

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


# %%
# TODO:
# 1. Put more fields to `node_ref` and `edge_ref` so not to searching
#   `ref_dfs` everytime.
# 2. Aggregation on edges.
class DFKGraph:
    """Knowledge-Graph represented with DFs.

    Aggragetion on nodes could be done on the DFKGraph.
    1. `node_ref` and `edge_ref` will be built during the initiation to cache
      the locations of each nodes and edges so to accelarate the aggregation.
    2. Filter before aggregation will be applied on different DFs in `ref_dfs`
      according to the Node-Type or Edge-Type.
      2.1. Locate all edges for selected nodes.
      2.2. Filter edges.
      2.3. Get the oppsite nodes of the edges.
      2.4. Filter the oppsite nodes.

    Attrs:
    ---------------------------
    node_ref: pd.DataFrame[`fnid`, `fntype`, __iloc__].
      DF of all nodes identified by `fnid`.
      1. `fnid`: Node-ID that should be unqiue among all the nodes.
      2. `fntype`: Could be used to get the DF of the exact type nodes in
        `ref_dfs`, as the the attributions of different nodes may be different.
      3. __iloc__: The i-loc of the node in correspondant DF in `ref_dfs`.
    edge_ref: pd.DataFrame[`fsrc`, `ftgt`, `fetype`, `edge_joinkey`].
      DF of all edges identified by `edge_joinkey`.
      1. `fsrc`: Source-Node-ID in `node_ref`.
      2. `ftgt`: Target-Node-ID in `node_ref`.
      3. `fetype`: Could be used to get the DF of the exact type edges in
        `ref_dfs`, as the the attributions of different edges may be different.
      4. `edge_joinkey`：The join-key for join the DFs of edges, which should
        be unique among all the edges, not just the edges of the same type.
        `[fsrc, ftgt]` will be set as the default.
      5. __iloc__: The i-loc of the node in correspondant DF in `ref_dfs`.
    ref_dfs: Dict[str, pd.DataFrame].
      Dict storing all the DFs of different kinds of Nodes and Edges.
    fnid: Str.
      Field name of the Node-Id in the DF of nodes.
    fsrc: Str.
      Field name of the Source-Node-Id in the DF of edges.
    ftgt: Str.
      Field name of the Target-Node-Id in the DF of edges.
    fntype: Str.
      Field name of the Node-Type in the DF of nodes.
    fetype: Str.
      Field name of the Edge-Type in the DF of edges.
    edge_joinkey: Str | tuple | list.
      Field names of the join-key for join the DF of edges.
    """
    def __init__(
        self,
        node_df: pd.DataFrame = None,
        edge_df: pd.DataFrame = None,
        ref_dfs: dict[str, pd.DataFrame] = None,
        *,
        fnid: str = "nid",
        fsrc: str = "source",
        ftgt: str = "target",
        fntype: str = "ntype",
        fetype: str = "etype",
        edge_joinkey: str | tuple | list = ("source", "target"),
        env: Mapping = None,
        envp: EnvParser = None,
    ):
        """Init a Knowledge-Graph represented with DFs.

        Params:
        ------------------------
        node_df：DF[`fnid`[, `fntype`]] of all nodes.
        edge_df: DF[`fsrc`, `ftgt`[, `fetype`, ...]] of all edges.
        ref_dfs: Dict of DFs of different type of nodes of edges.
          One of (`node_df`, `edge_df`) or `ref_dfs` must be provided to build
          `node_ref` and `edge_ref`.
        fnid: Field name of the Node-Id in the DF of nodes.
        fsrc: Field name of the Source-Node-Id in the DF of edges.
        ftgt: Field name of the Target-Node-Id in the DF of edges.
        fntype: Field name of the Node-Type in the DF of nodes.
        fetype: Field name of the Edge-Type in the DF of edges.
        edge_joinkey: Field names of the join-key for join the DF of edges.
        env: Mapping to provide extra searching space for EnvParser.
        envp: EnvParser to execute string.
          ATTENTION: `env` will be ignored if `envp` is passed.
        """
        self.fnid = fnid
        self.fsrc = fsrc
        self.ftgt = ftgt
        self.fntype = fntype
        self.fetype = fetype
        self.edge_joinkey = edge_joinkey

        self.node_ref = None
        self.edge_ref = None
        self.ref_dfs = ref_dfs.copy() if ref_dfs is not None else {}
        self.init_refs(node_df, edge_df)

        self.envp = EnvParser(env) if envp is None else envp

    def init_refs(
        self,
        node_df: pd.DataFrame = None,
        edge_df: pd.DataFrame = None,
    ):
        """Init the reference DF.

        1. `fntype` in `node_df` and `fetype` in `edge_df` will be used to
          join with the entity DF in `ref_dfs` to construct the reference DF.
        2. So if `fntype` doesn't exists in `node_df`, no more additional
          attributions of nodes represented DFs in `ref_dfs` could be used
          for filter or aggregation.
          So the `fetype` does.
        3. And `__NODE__`, `__EDGE__` will be used to point the `node_df` and
          `edge_df` for entity with no correspondant DF in `ref_dfs`.

        Params:
        ------------------------
        node_df：DF[`fnid`[, `fntype`]] of all nodes.
        edge_df: DF[`fsrc`, `ftgt`[, `fetype`, ...]] of all edges.

        Return:
        ------------------------
        None
        """
        ref_dfs = self.ref_dfs
        fnid = self.fnid
        fsrc = self.fsrc
        ftgt = self.ftgt
        fntype = self.fntype
        fetype = self.fetype
        edge_joinkey = self.edge_joinkey

        # Init node reference for searching the nodes to locate the DF of
        # the attributions of the nodes.
        if node_df is None:
            assert ref_dfs, "Nodes and entity's DF can't all be None."
            node_refs = []
            for rdf_name, rdf in ref_dfs.items():
                # Skip the DF that is not about nodes.
                if fnid not in rdf:
                    continue
                node_ref_ = pd.DataFrame({
                    "__iloc__": np.arange(rdf.shape[0]),
                    fntype: rdf_name,
                }, index=rdf[fnid].values)
                node_refs.append(node_ref_)
            node_ref = pd.concat(node_refs)
            # The index's name will be kept.
            node_ref.index.set_names(fnid, inplace=True)
        elif fntype not in node_df:
            logger.warning("No additional DF of nodes will be used.")
            node_ref = pd.DataFrame({
                "__iloc__": np.arange(node_df.shape[0]),
                fntype: "__NODE__",
            }, index=node_df[fnid].values)
            # Set `node_df` as the default searching DF of nodes.
            ref_dfs["__NODE__"] = node_df
        else:
            # Gather all node-types in node-DF.
            all_ntypes = np.unique(node_df[fntype].values)
            node_ref = pd.DataFrame({
                "__iloc__": np.arange(node_df.shape[0]),
                fntype: node_df[fntype].values,
            }, index=node_df[fnid])
            # set_trace()

            for ntype in all_ntypes:
                if ntype not in ref_dfs:
                    ref_dfs[ntype] = node_df
                else:
                    rdf = ref_dfs[ntype]
                    # Inner-join will keep the order of the `rdf`.
                    _mref = pd.merge(rdf, node_ref, on=fnid, how="inner")
                    assert _mref.shape[0] == rdf.shape[0], (
                        "All nodes must be included in `node_df`.")
                    # node_ref.loc[_mref[fnid], "__iloc__"] = np.arange(_mref.shape[0])
                    # node_ref.loc[_mref[fnid], fntype] = ntype
                    node_ref.iloc[_mref["__iloc__"].values,
                                  node_ref.columns.get_loc("__iloc__")] = \
                        np.arange(_mref.shape[0])
                    node_ref.iloc[_mref["__iloc__"].values,
                                  node_ref.columns.get_loc(fntype)] = ntype
        self.node_ref = node_ref

        # Init edge reference for searching the edges to locate the DF of
        # the attributions of the edges.
        # Note: I tried to construct two reference DF with source-node-id
        #   and target-node-id as the index seperately to located edges with
        #   node-ids. But `DF.loc` will return duplicates when replicated
        #   index passed and another drop for duplicates is necessary.
        if np.isscalar(edge_joinkey):
            edge_joinkey = [edge_joinkey, ]
        if edge_df is None:
            assert ref_dfs, "Nodes and entity's DF can't all be None."
            nec_keys = np.unique([*edge_joinkey, fsrc, ftgt]).tolist()
            edge_refs = []
            for rdf_name, rdf in ref_dfs.items():
                # Skip the DF that is not about nodes.
                if fsrc not in rdf or ftgt not in rdf:
                    continue
                edge_ref_ = rdf[nec_keys].copy()
                edge_ref_["__iloc__"] = np.arange(rdf.shape[0])
                edge_ref_[fetype] = rdf_name
                edge_refs.append(edge_ref_)
            edge_ref = pd.concat(edge_refs, axis=0).reset_index(drop=True)
        elif fetype not in edge_df:
            logger.warning("No additional DF of edges will be used.")
            nec_keys = np.unique([*edge_joinkey, fsrc, ftgt]).tolist()
            edge_ref = edge_df[nec_keys].reset_index(drop=True)
            edge_ref["__iloc__"] = np.arange(edge_df.shape[0])
            edge_ref[fetype] = "__EDGE__"
            # Set `edge_df` as the default searching DF of edges.
            ref_dfs["__EDGE__"] = edge_df
        else:
            all_etypes = np.unique(edge_df[fetype].values)
            nec_keys = np.unique([*edge_joinkey, fsrc, ftgt, fetype]).tolist()
            edge_ref = edge_df[nec_keys].reset_index(drop=True)
            edge_ref["__iloc__"] = np.arange(edge_df.shape[0])
            for etype in all_etypes:
                if etype not in ref_dfs:
                    ref_dfs[etype] = edge_df
                else:
                    rdf = ref_dfs[etype]
                    # Inner-join will keep the order of the `rdf`.
                    _mref = pd.merge(rdf, edge_ref, on=edge_joinkey, how="inner")
                    assert _mref.shape[0] == rdf.shape[0], (
                        "All edges must be included in `edge_df`.")
                    edge_ref.iloc[_mref["__iloc__"].values,
                                  edge_ref.columns.get_loc("__iloc__")] = \
                        np.arange(_mref.shape[0])
                    edge_ref.iloc[_mref["__iloc__"].values,
                                  edge_ref.columns.get_loc(fetype)] = etype
        self.edge_ref = edge_ref

    def agg_on_nodes(
        self,
        nids: Any,
        rules: list[tuple],
        env: Mapping = None,
        envp: EnvParser = None,
    ) -> pd.Series:
        """Aggregate starting for nodes and on nodes.

        Filter before aggregation will be applied on different DFs in `ref_dfs`
        according to the Node-Type or Edge-Type.
        1. Locate all edges for selected nodes.
        2. Filter edges.
        3. Get the oppsite nodes of the edges.
        4. Filter the oppsite nodes.

        Params:
        ------------------------
        nids: Node-ID or list of Node-ID.
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
        if envp:
            envp = envp
        elif env:
            envp = EnvParser(env)
        else:
            envp = self.envp
        if np.isscalar(nids):
            nids = [nids, ]
        node_ref = self.node_ref
        edge_ref = self.edge_ref
        ref_dfs = self.ref_dfs
        fnid = self.fnid
        fsrc = self.fsrc
        ftgt = self.ftgt
        fntype = self.fntype
        fetype = self.fetype
        edge_joinkey = self.edge_joinkey

        # The correspondant nodes to be target or source are always need to
        # be clarified even when the direction is "both".
        direction_proc = {
            "source": [(fsrc, ftgt)],
            "target": [(ftgt, fsrc)],
            "both": [(fsrc, ftgt), (ftgt, fsrc)],
        }

        nec_keys = np.unique([*edge_joinkey, fsrc, ftgt, fetype]).tolist()

        def apply_edge_cond(
            edge_ref: pd.Dataframe,
            edge_cond: str,
            ref_dfs: dict[str, pd.DataFrame],
            envp: EnvParser,
        ) -> pd.DataFrame:
            cur_edges = (ref_dfs[edge_ref.name]
                         .iloc[edge_ref["__iloc__"].values])
            # set_trace()
            # cur_edges[fetype] = edge_ref.name
            if edge_cond:
                edge_flag = envp.bind_env(cur_edges).parse(edge_cond)
                cur_edges = cur_edges.loc[edge_flag, nec_keys]
            return cur_edges

        def apply_node_cond(
            node_ref: pd.DataFrame,
            node_cond: str,
            ref_dfs: dict[str, pd.DataFrame],
            envp: EnvParser,
        ) -> pd.DataFrame:
            cur_nodes = (ref_dfs[node_ref.name]
                         .iloc[node_ref["__iloc__"].values])
            # set_trace()
            # cur_nodes[fntype] = node_ref.name
            if node_cond:
                node_flag = envp.bind_env(cur_nodes).parse(node_cond)
                # The `cur_nodes` will be used to aggragate, so all the fields
                # must be kept.
                cur_nodes = cur_nodes.loc[node_flag]
                # cur_nodes = cur_nodes.loc[node_flag, [fntype, fnid]]
            return cur_nodes

        ret = {}
        for rule in rules:
            cur_nids = nids
            if len(rule) == 2:
                conds = []
                key, agg = rule
            elif len(rule) == 3:
                key, conds, agg = rule
            else:
                logger.warning(f"Invalid rule: {rule} for aggregation on DataFrame.")
                continue

            # `direction` may be 1 of: source, target, both.
            for direction, edge_cond, node_cond in conds:
                all_nodes = []
                for fn, tn in direction_proc[direction]:
                    # 1. Filter edges with current nodes' id.
                    edge_ref_ = edge_ref.loc[edge_ref[fn].isin(cur_nids)]
                    # `.loc` will lead to duplicated edge records since the
                    # index is the source-node-id or target-node-id in edge
                    # records that will always be duplicated.
                    # So the `.isin` may be better?
                    # edge_ref_ = edge_ref_.drop_duplicates(subset=edge_joinkey)
                    if edge_ref_.empty:
                        continue

                    # 2. Filter edges with the `edge_cond`.
                    # cur_edges = (ref_dfs[edge_ref_.iloc[0][fetype]]
                    #              .iloc[edge_ref_["__iloc__"].values])
                    # if edge_cond and not cur_edges.empty:
                    #     edge_flag = envp.bind_env(cur_edges).parse(edge_cond)
                    #     cur_edges = cur_edges.loc[edge_flag]
                    cur_edges = (edge_ref_.groupby(fetype, as_index=False)
                                 [edge_ref_.columns]
                                 .apply(apply_edge_cond,
                                        edge_cond=edge_cond,
                                        ref_dfs=ref_dfs,
                                        envp=envp)
                                 .reset_index(drop=True))

                    # 3. Get correspondant nodes of the filtered edges.
                    oneway_node_ref_ = node_ref.loc[pd.unique(cur_edges[tn].values)]
                    if oneway_node_ref_.empty:
                        continue

                    # 4. Filter nodes with the `node_cond`.
                    # oneway_nodes = (ref_dfs[oneway_node_ref_.iloc[0][fntype]]
                    #                 .iloc[oneway_node_ref_["__iloc__"].values])
                    # if node_cond and not oneway_nodes.empty:
                    #     node_flag = envp.bind_env(oneway_nodes).parse(node_cond)
                    #     oneway_nodes = oneway_nodes.loc[node_flag]
                    # set_trace()
                    oneway_nodes = (oneway_node_ref_.groupby(fntype, as_index=False)
                                    [oneway_node_ref_.columns]
                                    .apply(apply_node_cond,
                                           node_cond=node_cond,
                                           ref_dfs=ref_dfs,
                                           envp=envp)
                                    .reset_index(drop=True))
                    # set_trace()

                    all_nodes.append(oneway_nodes)

                # Concat nodes in `all_nodes`
                if all_nodes:
                    cur_nodes = pd.concat(all_nodes, axis=0).drop_duplicates(fnid)
                else:
                    cur_nodes = pd.DataFrame(columns=[fntype, fnid])
                cur_nids = cur_nodes[fnid]

            # Aggragate on the filtered nodes.
            ret[key] = envp.bind_env(cur_nodes).parse(agg)

        return pd.Series(ret) if ret else pd.Series(dtype=object)
