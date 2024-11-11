#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confgraph.py
#   Author: xyy15926
#   Created: 2024-09-24 21:33:27
#   Updated: 2024-11-11 11:14:10
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

import logging
from itertools import product

from suitbear.kgraph.kgenum import (
    RelSrc,
    NodeType, NODE_TYPE_MAPPER,
    RoleType, ROLE_TYPE_MAPPER,
    LinkType, LINK_TYPE_MAPPER
)
from suitbear.autofin.confagg import LOAN_REPAYMENT, df_agg_confs

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
REL_AUTOFIN_PRETRIAL = {
    "part": "rel_autofin_pretrial",
    "from_": ["autofin_pretrial"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        "tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.APPLIER_TEL, },
        },
        "saler_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.SP_CERTNO, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "saler_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_FIN_N,
        },
    ]
}


# %%
REL_AUTOFIN_SECTRIAL = {
    "part": "rel_autofin_sectrial",
    "from_": ["autofin_sectrial"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        "tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.APPLIER_TEL, },
        },
        "spos_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.SPOS_CERTNO, },
        },
        "spos_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.SPOS_TEL, },
        },
        # 至多 5 联系人
        "link1_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.LINK_CERTNO, },
        },
        "link1_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.LINK_TEL, },
        },
        "link2_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.LINK_CERTNO, },
        },
        "link2_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.LINK_TEL, },
        },
        "link3_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.LINK_CERTNO, },
        },
        "link3_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.LINK_TEL, },
        },
        "link4_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.LINK_CERTNO, },
        },
        "link4_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.LINK_TEL, },
        },
        "link5_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.LINK_CERTNO, },
        },
        "link5_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.LINK_TEL, },
        },
        # 车、交易、工作、地址
        "vin": {
            "nattr": {"ntype": NodeType.VIN, },
            "rattr": {"role": RoleType.APPLYIED_VIN, },
        },
        "retailer_orgno": {
            "nattr": {"ntype": NodeType.ORGNO, },
            "rattr": {"role": RoleType.RETAILER_ORGNO, },
        },
        "retailer_name": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.RETAILER_NAME, },
        },
        "res_addr": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.RES_ADDR, },
        },
        "res_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.RES_TEL, },
        },
        "domi_addr": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.DOMI_ADDR, },
        },
        "comp_name": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.EMP_COMP_NAME, },
        },
        "comp_addr": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.EMP_COMP_ADDR, },
        },
        "comp_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.EMP_COMP_TEL, },
        },
        "spos_comp_name": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.SPOS_COMP_NAME, },
        },
        "spos_comp_addr": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.SPOS_COMP_NAME, },
        },
        "spos_comp_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.SPOS_COMP_TEL, },
        },
    },
    # 身份证号、手机号、车架号、统一代码关联
    "rels": [
        {
            "source": "certno",
            "target": "tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "spos_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_SPOS,
        },{
            "source": "spos_certno",
            "target": "spos_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },
        # 联系人
        {
            "source": "certno",
            "target": "link1_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_CNTC,
        },{
            "source": "link1_certno",
            "target": "link1_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "link1_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },{
            "source": "certno",
            "target": "link2_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_CNTC,
        },{
            "source": "link2_certno",
            "target": "link2_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "link2_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },{
            "source": "certno",
            "target": "link3_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_CNTC,
        },{
            "source": "link3_certno",
            "target": "link3_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "link3_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },{
            "source": "certno",
            "target": "link4_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_CNTC,
        },{
            "source": "link4_certno",
            "target": "link4_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "link4_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },{
            "source": "certno",
            "target": "link5_certno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_CNTC,
        },{
            "source": "link5_certno",
            "target": "link5_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "link5_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },
        # 车、交易
        {
            "source": "certno",
            "target": "vin",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_CAR,
        },{
            "source": "certno",
            "target": "retailer_name",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_PURCH_M,
        },
        # 居住地
        {
            "source": "certno",
            "target": "res_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_RESI,
        },{
            "source": "certno",
            "target": "res_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "certno",
            "target": "domi_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_RESI,
        },
        # 工作
        {
            "source": "certno",
            "target": "comp_name",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_COMP,
        },{
            "source": "comp_name",
            "target": "comp_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.MO_TEL,
        },{
            "source": "certno",
            "target": "comp_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },{
            "source": "comp_name",
            "target": "comp_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.M_RESI,
        },{
            "source": "certno",
            "target": "comp_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_RESI,
        },
        # 配偶信息
        {
            "source": "spos_certno",
            "target": "spos_comp_name",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_COMP,
        },{
            "source": "spos_comp_name",
            "target": "spos_comp_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.M_RESI,
        },{
            "source": "spos_certno",
            "target": "spos_comp_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_RESI,
        },{
            "source": "spos_comp_name",
            "target": "spos_comp_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.MO_TEL,
        },{
            "source": "spos_certno",
            "target": "spos_comp_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NL_TEL,
        },
    ]
}


# %%
REL_AUTOFIN_RETAIL = {
    "part": "rel_autofin_retail",
    "from_": ["autofin_retail"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "rep_certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.RETAILER_REP_CERTNO, },
        },
        "rep_tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.RETAILER_REP_TEL, },
        },
        "orgno": {
            "nattr": {"ntype": NodeType.ORGNO, },
            "rattr": {"role": RoleType.RETAILER_ORGNO, },
        },
        "orgname": {
            "nattr": {"ntype": NodeType.ORGNAME, },
            "rattr": {"role": RoleType.RETAILER_NAME, },
        },
        "org_addr": {
            "nattr": {"ntype": NodeType.ADDR, },
            "rattr": {"role": RoleType.RETAILER_ADDR, },
        },
    },
    "rels": [
        {
            "source": "rep_certno",
            "target": "orgname",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.N_COMP,
        },{
            "source": "rep_certno",
            "target": "rep_tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },{
            "source": "orgname",
            "target": "org_addr",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.M_RESI,
        },{
            "source": "orgname",
            "target": "orgno",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.M_ORGNO,
        }
    ]
}


# %%
REL_AUTOFIN_SALER = {
    "part": "rel_autofin_saler",
    "from_": ["autofin_saler"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.SP_CERTNO, },
        },
        "tel": {
            "nattr": {"ntype": NodeType.TEL, },
            "rattr": {"role": RoleType.SP_TEL, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "tel",
            "update": "apply_date",
            "rid": "order_no",
            "ltype": LinkType.NO_TEL,
        },
    ]
}


# %%
REL_LOAN_ACC_INFO = {
    "part": "rel_loan_acc_info",
    "from_": ["loan_acc_info"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        "debit_card": {
            "nattr": {"ntype": NodeType.PACCNO, },
            "rattr": {"role": RoleType.BUCKLE_ACCNO, },
        }
    },
    "rels": [
        {
            "source": "certno",
            "target": "debit_card",
            "update": "loan_date",
            "rid": "order_no",
            "ltype": LinkType.N_PACC,
        }
    ]
}

REL_REPAYMENT_MONTHLY = {
    "part": "rel_repayment_monthly",
    "from_": ["loan_repayment_monthly"],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {
            "nattr": {"ntype": NodeType.CERTNO, },
            "rattr": {"role": RoleType.APPLIER_CERTNO, },
        },
        "repay_card": {
            "nattr": {"ntype": NodeType.PACCNO, },
            "rattr": {"role": RoleType.REPAY_ACCNO, },
        },
    },
    "rels": [
        {
            "source": "certno",
            "target": "repay_card",
            "update": "repay_date",
            "rid": "order_no",
            "ltype": LinkType.N_PACC,
        }
    ]
}


# %%
NODE_LOAN_REPAYMENT = {
    "part": "node_loan_repayment_monthly",
    "from_": [LOAN_REPAYMENT["part"],],
    "etype": RelSrc.AUTOFIN,
    "nodes": {
        "certno": {key: key for key in df_agg_confs(
            {LOAN_REPAYMENT["part"]: LOAN_REPAYMENT})[1]["key"].values},
    },
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


AUTOFIN_GRAPH_CONFS = {
    REL_AUTOFIN_RETAIL["part"]: REL_AUTOFIN_RETAIL,
    REL_AUTOFIN_SALER["part"]: REL_AUTOFIN_SALER,
    REL_AUTOFIN_PRETRIAL["part"]: REL_AUTOFIN_PRETRIAL,
    REL_AUTOFIN_SECTRIAL["part"]: REL_AUTOFIN_SECTRIAL,
    REL_LOAN_ACC_INFO["part"]: REL_LOAN_ACC_INFO,
    REL_REPAYMENT_MONTHLY["part"]: REL_REPAYMENT_MONTHLY,
}
for key, val in AUTOFIN_GRAPH_CONFS.items():
    update_rattr(val)


# %%
# 分类过多、太细，无意义
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
    for rec_conf in AUTOFIN_GRAPH_CONFS.values():
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
                    and (node_type == node_types_D[from_node]["nattr"]["ntype"]
                         or node_type == node_types_D[to_node]["nattr"]["ntype"])):
                repr_ = (f"{to_key}_ft_{from_key}",
                         f"(source_role == {source_role}) "
                         f"& (target_role == {target_role})",
                         f"作为{to_desc}或{from_desc}相关联")
                reprs.append(repr_)

    return reprs


# %%
def link_type_reprs(
    ntype: NodeType,
    direction: str = "both"
) -> list[tuple]:
    # Collect all possible link types for each kind of node types.
    ns_ltypes = {}
    for rec_conf in AUTOFIN_GRAPH_CONFS.values():
        node_types_D = rec_conf["nodes"]
        for rel in rec_conf["rels"]:
            for drt in ("source", "target"):
                node = rel[drt]
                cur_ntype = node_types_D[node]["nattr"]["ntype"]
                n_ltypes = ns_ltypes.setdefault(cur_ntype, {})
                _ltypes = n_ltypes.setdefault(drt, set())
                _ltypes.add(rel["ltype"])

    if ntype not in ns_ltypes:
        logger.warning(f"No relations fits for the node type {ntype.name}.")
        return []

    # Construct reprs.
    if direction == "source":
        ltypes = (ns_ltypes[ntype].get(direction, []), [])
    elif direction == "target":
        ltypes = ([], ns_ltypes[ntype].get(direction, []))
    else:
        ltypes = (ns_ltypes[ntype].get("source", []),
                  ns_ltypes[ntype].get("target", []))

    reprs = []
    for lt in ltypes[0]:
        kdesc, cdesc = LINK_TYPE_MAPPER[lt][0]
        reprs.append((kdesc, f"ltype == {lt.value}", cdesc))
    for lt in ltypes[1]:
        kdesc, cdesc = LINK_TYPE_MAPPER[lt][1]
        reprs.append((kdesc, f"ltype == {lt.value}", cdesc))

    if len(reprs) == 0:
        logger.warning(f"No relations fits for the node type {ntype.name}.")

    return reprs


# %%
def ntype_reprs(ntypes: list[NodeType] = None) -> list[tuple]:
    ntypes = NodeType if ntypes is None else ntypes
    reprs = []
    for nt in ntypes:
        kd, cd = NODE_TYPE_MAPPER[nt]
        reprs.append((kd, f"ntype == {nt.value}", cd))
    reprs += [(None, None, None), ]

    return reprs


# %%
# TODO
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
          source_role: The role of the source.
          target_role: The role of the target.
        Mandatory attrs:
          update: Update time of the record.
          rid: The record id.

    Return:
    ----------------------
    pconfs: DataFrame[part, from_, etype]
    node_confs: DateFrame[node, ntype]
    rel_confs: DataFrame[
    """
    import pandas as pd
    if confs is None:
        confs = AUTOFIN_GRAPH_CONFS

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
      nodes: Dict of the fields acting as the nodes.
        Mandatory labels:
          ntype: Node label indicating the node type.
      rels: List dict representing the relations.
        Mandatory labels:
          source: Field of source node ids
          target: Field of target node ids.
          source_role: The role of the source.
          target_role: The role of the target.
        Mandatory attrs:
          update: Update time of the record.
          rid: The record id.

    Return:
    ----------------------------
    node_df: DataFrame representing the node:
      id: Node identifier.
      ntype: Node type.
    rel_df: DataFrame representing the relations:
      source: Source node ids.
      target: Target node ids.
      source_role: The role of the source.
      target_role: The role of the target.
    """
    import pandas as pd

    if graph_confs is None:
        graph_confs = AUTOFIN_GRAPH_CONFS

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
    # Drop duplicated relations.
    rel_df = (pd.concat(rels)
              .sort_values("update")
              .drop_duplicates(subset=["source", "target"], keep="last"))
    node_df = pd.concat(nodes).drop_duplicates()

    return rel_df, node_df


# %%
def mock_digraph(dfs: dict):
    import networkx as nx

    rel_df = build_graph_df(dfs)
    dig = nx.from_pandas_edgelist(rel_df, source="source",
                                  target="target",
                                  edge_attr=True)
    return dig
