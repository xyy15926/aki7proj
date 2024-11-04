#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pbocrels.py
#   Author: xyy15926
#   Created: 2024-10-15 21:02:53
#   Updated: 2024-11-03 20:50:12
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

from itertools import product
from suitbear.kgraph.kgenum import RelType, NodeType, RoleType, ROLE_TYPE_MAPPER

# %%
REL_BASIC_INFO = {
    "part": "pboc_basic_info_rel",
    "from_": ["pboc_basic_info"],
    "etype": RelType.PBOC,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        # 通讯地址
        "PB01AQ02": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.RES_ADDR, }
        },
        # 户籍地
        "PB01AQ03": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.DOMI_ADDR, },
        },
        # 配偶身份证号
        "PB020I01": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.SPOS_CERTNO, },
        },
        # 配偶工作单位
        "PB020Q02": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.SPOS_COMP_NAME, }
        },
        # 配偶联系电话
        "PB020Q03": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.SPOS_TEL, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "PB01AQ02",
            "update": "PA01AR01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB01AQ03",
            "update": "PA01AR01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB020I01",
            "update": "PA01AR01",
            "rid": "rid",
        },{
            "source": "PB020I01",
            "target": "PB020Q02",
            "update": "PA01AR01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB020Q03",
            "update": "PA01AR01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB020Q02",
            "update": "PA01AR01",
            "rid": "rid",
        }
    ]
}


# %%
REL_MOBILE = {
    "part": "pboc_mobile_rel",
    "from_": ["pboc_mobile"],
    "etype": RelType.PBOC,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, }
        },
        "PB01BQ01": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.APPLIER_TEL, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "PB01BQ01",
            "update": "PB01BR01",
            "rid": "rid",
        }
    ]
}


# %%
REL_ADDRESS = {
    "part": "pboc_address_rel",
    "from_": ["pboc_address"],
    "etype": RelType.PBOC,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        # 居住地址
        "PB030Q01": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.RES_ADDR, }
        },
        # 居住电话
        "PB030Q02": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.RES_TEL, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "PB030Q01",
            "update": "PB030R01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB030Q02",
            "update": "PB030R01",
            "rid": "rid",
        }
    ]
}


# %%
REL_COMPANY = {
    "part": "pboc_company_rel",
    "from_": ["pboc_company"],
    "etype": RelType.PBOC,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, }
        },
        # 工作单位
        "PB040Q01": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.EMP_COMP_NAME, }
        },
        # 单位地址
        "PB040Q02": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.EMP_COMP_ADDR, },
        },
        # 单位电话
        "PB040Q03": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.EMP_COMP_TEL, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "PB040Q01",
            "update": "PB040R01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB040Q02",
            "update": "PB040R01",
            "rid": "rid",
        },{
            "source": "certno",
            "target": "PB040Q03",
            "update": "PB040R01",
            "rid": "rid",
        }
    ]
}


# %%
REL_HOUSING_FUND = {
    "part": "pboc_housing_fund_rel",
    "from_": ["pboc_housing_fund"],
    "etype": RelType.PBOC,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        "PF05AQ04": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.EMP_COMP_NAME, }
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "PF05AQ04",
            "update": "PF05AR03",
            "rid": "rid",
        }
    ]
}


# %%
def update_rattr(confs: dict):
    etype = confs["etype"]
    for rel in confs["rels"]:
        for nn in ["source", "target"]:
            noden = rel[nn]
            rattrs = confs["nodes"][noden]["rattr"]
            for attr, aval in rattrs.items():
                rel[f"{nn}_{attr}"] = aval
        rel["etype"] = etype
    return confs


PBOC_GRAPH_CONFS = {
    REL_BASIC_INFO["part"]: REL_BASIC_INFO,
    REL_MOBILE["part"]: REL_MOBILE,
    REL_ADDRESS["part"]: REL_ADDRESS,
    REL_COMPANY["part"]: REL_COMPANY,
    REL_HOUSING_FUND["part"]: REL_HOUSING_FUND,
}
for key, val in PBOC_GRAPH_CONFS.items():
    update_rattr(val)


# %%
def node_role_reprs(
    node_type: int = NodeType.CERTNO,
    direction: str = "source",
) -> list[tuple[str]]:
    """Node possible roles for each node type.

    Params:
    -----------------------
    node_type: Node type
    direction: The position of the node in the relation.
      source: The node could be the source of the relation according its
        type.
      target: The node could be the target of the relation.
      both: The node could be both the source and the target of the relation.

    Return:
    -----------------------
    [key-part, cond, desc]
    """
    reprs = []
    for rec_conf in PBOC_GRAPH_CONFS.values():
        node_types_D = rec_conf["nodes"]
        # Traverse all the relations for those with `node_type` as the
        # `source_role`.
        for rel in rec_conf["rels"]:
            from_node, to_node = rel["source"], rel["target"]
            source_role, target_role = rel["source_role"], rel["target_role"]
            from_key, from_desc, ntype = ROLE_TYPE_MAPPER[source_role]
            to_key, to_desc, ntype = ROLE_TYPE_MAPPER[target_role]
            if (direction == "source"
                    and node_type == node_types_D[from_node]["nattr"]["ntype"]):
                repr_ = (f"{from_key}_to_{to_key}",
                         f"(source_role == {source_role}) "
                         f"& (target_role == {target_role})",
                         f"作为{from_desc}关联{to_desc}")
                reprs.append(repr_)
            elif (direction == "target"
                    and node_type == node_types_D[to_node]["nattr"]["ntype"]):
                repr_ = (f"{to_key}_fm_{from_key}",
                         f"(source_role == {source_role}) "
                         f"& (target_role == {target_role})",
                         f"作为{to_desc}被关联至{from_desc}")
                reprs.append(repr_)
            elif (direction == "both"
                    and node_type == node_types_D[from_node]["nattr"]["ntype"]
                    and node_type == node_types_D[to_node]["nattr"]["ntype"]):
                repr_ = (f"{to_key}_ft_{from_key}",
                         f"(source_role == {source_role}) "
                         f"& (target_role == {target_role})",
                         f"作为{to_desc}或{from_desc}相关联")
                reprs.append(repr_)

    return reprs


# %%
def df_graph_confs(
    confs: dict = None
) -> tuple[pd.DataFrame]:
    """Construct graph confs into DataFrame.

    Params:
    ----------------------
    confs: Dict acting as the graph conf with keys:
      part: The part name for the graph.
      from_: The part name from where to construct the graph.
      etype: Relation label indicating the relation type.
      nodes: Dict of the fields acting as the nodes.
        Mandatory labels:
          ntype: Node label indicating the node type.
      rels: List dict representing the relations.
        Mandatory labels:
          source: Source node ids
          target: Target node ids.
          src_role: The role of the source.
          tgt_role: The role of the target.
        Mandatory attrs:
          update: Update time of the record.
          rid: The record id.

    Return:
    ----------------------
    pconfs: DataFrame[part, from_, etype]
    node_confs: Node conf
    rel_confs: Relation conf
    """
    import pandas as pd

    if confs is None:
        confs = PBOC_GRAPH_CONFS

    pconfs = []
    node_confs = {}
    rel_confs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"],
                       pconf["from_"],
                       pconf["etype"]))
        nc = (pd.DataFrame({nn: nv["nattr"]
                            for nn, nv in pconf["nodes"].items()}).T
              .reset_index()
              .rename({"index": "node"}, axis=1))
        node_confs[pname] = nc
        rel_confs[pname] = pd.DataFrame(pconf["rels"])

    # Concat the confs.
    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "from_", "etype"])
    node_confs = pd.concat(node_confs.values(),
                           keys=node_confs.keys()).droplevel(level=1)
    node_confs.index.set_names("part", inplace=True)
    node_confs = node_confs.reset_index()

    rel_confs = pd.concat(rel_confs.values(),
                          keys=rel_confs.keys()).droplevel(level=1)
    rel_confs.index.set_names("part", inplace=True)
    rel_confs = rel_confs.reset_index()

    return pconfs, node_confs, rel_confs


# %%
def build_graph_df(
    dfs: dict,
    graph_confs: dict[str, dict] = None,
) -> tuple[pd.DataFrame]:
    """Build DataFrame representing graph from DataFrames.

    Params:
    ----------------------------
    dfs:DataFrames of records from which to build graph-DF.
    graph_confs: The conf dicts for build graph-DF. Each of the conf dict
      should be with these elements:
      part: The part name for the graph.
      from_: The part name from where to construct the graph.
      etype: Relation label indicating the relation type.
      nodes: Dict of the fields acting as the nodes.
        Mandatory labels:
          ntype: Node label indicating the node type.
      rels: List dict representing the relations.
        Mandatory labels:
          source: Field of source node ids
          target: Field of target node ids.
          src_role: The role of the source.
          tgt_role: The role of the target.
        Mandatory attrs:
          update: Update time of the record.
          rid: The record id.

    Return:
    ----------------------------
    node_df: DataFrame representing the node:
      id: Node identifier
      ntype: Node type
    rel_df: DataFrame representing the relations:
      source: Source node id
      target: Target node id
      src_role: The role of the source
      tgt_role: The role of the target
    """
    import pandas as pd

    if graph_confs is None:
        graph_confs = PBOC_GRAPH_CONFS

    rels = []
    nodes = []
    for part, rel_conf in graph_confs.items():
        df = dfs[rel_conf["from_"][0]]
        # Construct DataFrame of edges among which the columns are the
        # attributions of the edges.
        for rel in rel_conf["rels"]:
            rel_vals = {}
            for key, val in rel.items():
                if val in df:
                    rel_vals[key] = df[val].values
                else:
                    rel_vals[key] = [val] * df.shape[0]
            rel_one = (pd.DataFrame(rel_vals)
                       .dropna(axis=0,
                               how="any",
                               subset=["source", "target"]))
            rels.append(rel_one)
        # Construct DataFrame of nodes among which the columns are the
        # attributions of the nodes.
        for node, aattrs in rel_conf["nodes"].items():
            na = {}
            # Use `nid` as the node indentifier.
            na["nid"] = df[node]
            nattrs = aattrs["nattr"]
            for attr_key, attr_val in nattrs.items():
                if attr_key not in df:
                    na[attr_key] = attr_val
                else:
                    na[attr_key] = df[attr_val]
            # Drop nan nodes.
            na = pd.DataFrame(na).dropna(subset="nid")
            nodes.append(na)
    rel_df = (pd.concat(rels)
              .sort_values("update")
              .drop_duplicates(subset=["source", "target"], keep="last"))
    node_df = pd.concat(nodes).drop_duplicates()

    return rel_df, node_df


# %%
def build_digraph(dfs: dict):
    import networkx as nx

    rel_df = build_graph_df(dfs)
    dig = nx.from_pandas_edgelist(rel_df, source="source",
                                  target="target",
                                  edge_attr=True)
    return dig
