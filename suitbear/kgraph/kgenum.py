#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: kgenum.py
#   Author: xyy15926
#   Created: 2024-10-15 21:00:54
#   Updated: 2024-10-31 15:18:27
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar
import enum
from enum import IntEnum


# %%
# 1. 使用枚举类型表示特殊的节点、边属性，对应 Neo4j 中 “标签” 概念，
#    用于从逻辑角度区分节点、边
# 2. 对节点，标签为使用单个字段 `ntype` 取枚举类型 `NodeType`
# 3. 对边，暂时未使用标签区分，但两个字段 `from_role`、`to_role` 取枚举类型
#    `RoleType`
# `IntEnum` is JSON-serializable.
@enum.unique
class RelType(IntEnum):
    PBOC = 1
    AUTOFIN = 2


# %%
@enum.unique
class NodeType(IntEnum):
    CERTNO = 1
    ORGNO = 2
    VIN = 3
    TEL = 4
    ORGNAME = 5
    ADDR = 6
    PACCNO = 7


NODE_TYPE_MAPPER = {
    NodeType.CERTNO: ("certno", "身份证号"),
    NodeType.ORGNO: ("orgno", "统一社会信用代码"),
    NodeType.VIN: ("vin", "车架号"),
    NodeType.TEL: ("tel", "电话号码"),
    NodeType.ORGNAME: ("orgname", "机构名称"),
    NodeType.ADDR: ("addr", "地址"),
    NodeType.PACCNO: ("paccno", "银行卡号"),
}


# %%
@enum.unique
class RoleType(IntEnum):
    # 1XX: certno
    # 2XX: telephone
    # 3XX: vin
    # 4XX: orgno
    # 5XX: address
    # 6XX: org name
    APPLIER_CERTNO = 101
    APPLIER_TEL = 201
    SPOS_CERTNO = 102
    SPOS_TEL = 202
    LINK_CERTNO = 103
    LINK_TEL = 203
    APPLYIED_VIN = 301
    RETAILER_ORGNO = 401
    RES_ADDR = 501
    RES_TEL = 204
    DOMI_ADDR = 502
    EMP_COMP_NAME = 601
    EMP_COMP_ADDR = 503
    EMP_COMP_TEL = 205
    SPOS_COMP_NAME = 602
    SPOS_COMP_ADDR = 504
    SPOS_COMP_TEL = 206
    RETAILER_ADDR = 505
    RETAILER_REP_CERTNO = 104
    RETAILER_REP_TEL = 207
    SP_CERTNO = 105
    SP_TEL = 208
    REPAY_ACCNO = 701


ROLE_TYPE_MAPPER = {
    RoleType.APPLIER_CERTNO: ("app_certno", "申请人身份证号", NodeType.CERTNO),
    RoleType.APPLIER_TEL: ("app_tel", "申请人手机号", NodeType.TEL),
    RoleType.SPOS_CERTNO: ("spos_certno", "配偶身份证号", NodeType.CERTNO),
    RoleType.SPOS_TEL: ("spos_tel", "配偶手机号", NodeType.TEL),
    RoleType.LINK_CERTNO: ("link_certno", "联系人身份证号", NodeType.CERTNO),
    RoleType.LINK_TEL: ("link_tel", "联系人手机号", NodeType.TEL),
    RoleType.APPLYIED_VIN: ("app_vin", "提报车架号", NodeType.VIN),
    RoleType.RETAILER_ORGNO: ("ret_orgno", "门店统一代码", NodeType.ORGNO),
    RoleType.RES_ADDR: ("res_addr", "住址", NodeType.ADDR),
    RoleType.RES_TEL: ("res_tel", "住址电话", NodeType.TEL),
    RoleType.DOMI_ADDR: ("domi_addr", "籍贯", NodeType.ADDR),
    RoleType.EMP_COMP_NAME: ("emp_comp_name", "公司名称", NodeType.ORGNAME),
    RoleType.EMP_COMP_TEL: ("emp_comp_tel", "公司电话", NodeType.TEL),
    RoleType.EMP_COMP_ADDR: ("emp_comp_addr", "公司地址", NodeType.ADDR),
    RoleType.SPOS_COMP_NAME: ("spos_comp_name", "配偶公司名称", NodeType.ORGNAME),
    RoleType.SPOS_COMP_ADDR: ("spos_comp_addr", "配偶公司地址", NodeType.ADDR),
    RoleType.SPOS_COMP_TEL: ("spos_comp_tel", "配偶公司电话", NodeType.TEL),
    RoleType.RETAILER_ADDR: ("ret_addr", "门店地址", NodeType.ADDR),
    RoleType.RETAILER_REP_CERTNO: ("ret_rep_certno", "门店法人代表身份证号", NodeType.CERTNO),
    RoleType.RETAILER_REP_TEL: ("ret_rep_tel", "门店法人代表手机号", NodeType.TEL),
    RoleType.SP_CERTNO: ("sp_certno", "SP身份证号", NodeType.CERTNO),
    RoleType.SP_TEL: ("sp_tel", "SP手机号", NodeType.TEL),
    RoleType.REPAY_ACCNO: ("repay_accno", "还款卡号", NodeType.PACCNO),
}


# %%
ENUM_MAPPERS = {
    "node_type": NODE_TYPE_MAPPER,
    "role_type": ROLE_TYPE_MAPPER,
}


def df_enum_confs(enum_M: dict = ENUM_MAPPERS):
    import pandas as pd

    enum_descs = {}
    for mkey, mp in enum_M.items():
        mpp = [(ee.name, ee.value, kd, chnd)
               for ee, (kd, chnd, *____) in mp.items()]
        enum_descs[mkey] = pd.DataFrame.from_records(
            mpp, columns=["name", "value", "keyname", "desc"])
    enum_df = pd.concat(enum_descs.values(), keys=enum_descs.keys())
    enum_df.index.set_names(["ENUM", "NO"], inplace=True)

    return enum_df
