#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pboc_aggs.py
#   Author: xyy15926
#   Created: 2024-04-22 10:13:57
#   Updated: 2024-04-23 21:44:07
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from flagbear import exgine
    from suitbear import fxgine, crosconf
    from azkaban import pboc_conf
    reload(exgine)
    reload(fxgine)
    reload(crosconf)
    reload(pboc_conf)

import os
from IPython.core.debugger import set_trace
from flagbear.fliper import extract_field
from suitbear.fxgine import compress_hierarchy, flat_records, agg_from_dfs
from suitbear.crosconf import agg_confs_from_dict, cross_aggs_from_lower
from azkaban.pboc_conf import LV1_AGG_CONF, LV2_AGG_CONF, LV20_AGG_CONF
from azkaban.pboc_conf import MAPPERS, TRANS_CONF

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
ASSETS = os.path.join(os.curdir, "assets")
PBOC_PARTS = os.path.join(ASSETS, "pboc_parts.csv")
PBOC_FIELDS = os.path.join(ASSETS, "pboc_fields.csv")

MAPPERS_ = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in MAPPERS.items()}
MAPPERS_["today"] = pd.Timestamp.today()


def sdatetime(x):
    try:
        return pd.to_datetime(x)
    except ValueError as e:
        logger.warning(e)
        return pd.NaT


# Default value settings for extracting fields.
DTYPE_DEFAULT = {
    "INT": np.nan,
    "FLO": np.nan,
    "VAR": "",
    "CHA": "",
    "DAT": sdatetime,
}
DTYPE_USE_DEFAULT = {
    "INT": 1,
    "FLO": 1,
    "VAR": 0,
    "CHA": 0,
    "DAT": 1,
}


# %%
def concat_confs():
    """Concatnate PBOC aggregations configs.
    """
    # Generate Level-2-1 and Level-1-0 PBOC aggregations configs.
    lv2_pconfs, lv2_aconfs = agg_confs_from_dict(LV2_AGG_CONF)
    lv1_pconfs, lv1_aconfs = agg_confs_from_dict(LV1_AGG_CONF)

    # Generate Level-2-1-0 aggregation configs.
    lv20_pconfs = []
    lv20_aconfs = []
    for pname, pconf in LV20_AGG_CONF.items():
        pname = pconf["part"]
        lv20_pconfs.append({
            "part": pname,
            "from_": pconf["from_"],
            "prikey": pconf["prikey"],
            "level": pconf["level"],
        })
        lv2_aconfs_ = lv2_aconfs[lv2_aconfs["part"] == pconf["lower_from"]]
        lv20_aconfs_ = lv2_aconfs_.apply(cross_aggs_from_lower,
                                         axis=1,
                                         cros_D=pconf["cros"],
                                         aggs_D=pconf["agg"],
                                         filters_D=pconf["cond"],
                                         key_fmt=pconf["key_fmt"])
        lv20_aconfs_ = pd.concat(lv20_aconfs_.values)
        lv20_aconfs_["part"] = pname
        lv20_aconfs.append(lv20_aconfs_)

    lv20_pconfs = pd.DataFrame.from_records(lv20_pconfs)
    lv20_aconfs = pd.concat(lv20_aconfs)

    # Concatenate configs.
    pconfs = pd.concat([lv2_pconfs, lv1_pconfs, lv20_pconfs])
    aconfs = pd.concat([lv2_aconfs, lv1_aconfs, lv20_aconfs])

    return pconfs, aconfs


# %%
def cal_vars(
    src: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Calculate index from PBOC records.

    Params:
    ---------------------------
    src: Series of PBOC records.

    Return:
    ---------------------------
    Dict[part-name, aggregation result]
    """
    # Read fields extraction config and addup some default settings.
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)
    fconfs["default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_DEFAULT)
    fconfs["use_default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_USE_DEFAULT)

    # Extract fields.
    dfs = {}
    for idx, pconf in pconfs.iterrows():
        pconf = pconf.dropna()
        psrc = compress_hierarchy(src, pconf)
        fconfs_ = fconfs[fconfs["part"] == pconf["part"]]
        ret = flat_records(psrc, fconfs_)
        dfs[pconf["part"]] = ret

    # Construct transformation and aggregation config.
    part_confs, agg_confs = concat_confs()
    trans_confs = [pd.DataFrame(val, columns=["key", "trans", "cond", "cmt"])
                   for val in TRANS_CONF.values()]
    trans_confs = (pd.concat(trans_confs, keys=TRANS_CONF.keys())
                   .droplevel(1)
                   .reset_index()
                   .rename(columns={"index": "part"}))

    # Apply aggregations.
    part_confs_ = pd.concat([pconfs, part_confs])
    ret = agg_from_dfs(dfs, part_confs_, agg_confs, trans_confs, MAPPERS_)

    return ret


# %%
if __name__ == "__main__":
    PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})
    ret = cal_vars(src)

    xlw = pd.ExcelWriter("pboc_vars.xlsx")
    for parts, var_df in ret.items():
        var_df.to_excel(xlw, sheet_name=parts)
    xlw.close()
