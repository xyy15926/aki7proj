#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pboc_aggs.py
#   Author: xyy15926
#   Created: 2024-04-22 10:13:57
#   Updated: 2024-09-05 10:47:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from flagbear import fliper
    from modsbear import exgine
    from suitbear import fxgine, crosconf
    from azkaban import pboc_conf
    reload(fliper)
    reload(exgine)
    reload(fxgine)
    reload(crosconf)
    reload(pboc_conf)

import os
from IPython.core.debugger import set_trace
from flagbear.fliper import extract_field
from modsbear.exgine import rebuild_rec2df, agg_on_df
from suitbear.fxgine import compress_hierarchy, flat_records, agg_from_dfs
from suitbear.crosconf import agg_confs_from_dict, cross_aggs_from_lower
from azkaban.pboc_conf import LV1_AGG_CONF, LV2_AGG_CONF, LV20_AGG_CONF
from azkaban.pboc_conf import MAPPERS, TRANS_CONF
from suitbear.finer import get_assets_path, get_tmp_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
PBOC_PARTS = get_assets_path() / "pboc_parts.csv"
PBOC_FIELDS = get_assets_path() / "pboc_fields.csv"

MAPPERS_ = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in MAPPERS.items()}
MAPPERS_["today"] = pd.Timestamp.today()


# %%
def concat_confs() -> Tuple[pd.DataFrame]:
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
        # Cross to get the aggregation configs of the `lv2_aconfs`.
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

    if np.any(aconfs["key"].duplicated()):
        logger.error("Duplicated key in aggregation config.")

    return pconfs, aconfs


# %%
def pboc_fields(
    src: pd.Series,
    write_fields: bool = False,
) -> dict[str, pd.DataFrame]:
    """Extract fields from PBOC records.

    Params:
    ---------------------------
    src: Series of PBOC records.
    write_fields: The filename for writing the extractions to.

    Return:
    ---------------------------
    Dict[part-name, DataFrame of values of parts]
    """
    # Read fields extraction config and addup some default settings.
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)

    # Extract fields.
    dfs = {}
    for idx, pconf in pconfs.iterrows():
        pconf = pconf.dropna()
        psrc = compress_hierarchy(src, pconf)
        fconfs_ = fconfs[fconfs["part"] == pconf["part"]]
        ret = flat_records(psrc, fconfs_)
        dfs[pconf["part"]] = ret

    if write_fields:
        xlw = pd.ExcelWriter(write_fields)
        for parts, var_df in dfs.items():
            if not var_df.empty:
                var_df.to_excel(xlw, sheet_name=parts)
        xlw.close()

    return dfs


# %%
def pboc_vars(
    dfs: dict[str, pd.DataFrame],
    agg_key_mark: pd.Series | set | list = None,
    write_aggconf: bool = False,
    write_vars: bool = False,
) -> dict[str, pd.DataFrame]:
    """Calculate index from PBOC records.

    Params:
    ---------------------------
    src: Series of PBOC records.
    write_aggconf: The filename for writing the aggregations configs to.
    write_vars: The filename for writing the variables to.

    Return:
    ---------------------------
    Dict[part-name, aggregation result]
    """
    # Construct transformation and aggregation config.
    part_confs, agg_confs = concat_confs()
    trans_confs = [pd.DataFrame(val, columns=["key", "trans", "cond", "cmt"])
                   for val in TRANS_CONF.values()]
    trans_confs = (pd.concat(trans_confs, keys=TRANS_CONF.keys())
                   .droplevel(1)
                   .reset_index()
                   .rename(columns={"index": "part"}))

    if agg_key_mark is not None:
        agg_confs = agg_confs[agg_confs["key"].isin(agg_key_mark)]

    # Read `PBOC_PARTS` for primary key of each part.
    pconfs = pd.read_csv(PBOC_PARTS)
    part_confs_ = pd.concat([pconfs, part_confs])

    if write_aggconf:
        xlw = pd.ExcelWriter(write_aggconf)
        part_confs.to_excel(xlw, sheet_name="pboc_vars_parts")
        trans_confs.to_excel(xlw, sheet_name="pboc_vars_trans")
        agg_confs.to_excel(xlw, sheet_name="pboc_vars_aggs")
        pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()],
                  axis=0,
                  keys=MAPPERS.keys()).to_excel(xlw,
                                                sheet_name="pboc_vars_maps")
        xlw.close()

    # Apply aggregations.
    ret = agg_from_dfs(dfs, part_confs_, agg_confs, trans_confs, MAPPERS_)

    if write_vars:
        xlw = pd.ExcelWriter(write_vars)
        for part, var_df in ret.items():
            if not var_df.empty:
                stop = 0
                while stop * 10000 < var_df.shape[1]:
                    var_df.iloc[:, stop * 10000: (stop + 1) * 10000].to_excel(
                        xlw, sheet_name=f"{part}_part{stop}")
                    stop += 1
        xlw.close()

    return ret


# %%
def from_files(
    files: List,
    today: str = "report",
    fields_file: str = "pboc_fields.xlsx",
    aggconf_file: str = "pboc_aggconf.xlsx",
    vars_file: str = "pboc_vars.xlsx",
    agg_keys: str = None,
) -> pd.DataFrame:
    """Extract fields from files and apply aggregations.
    """
    # files = (get_tmp_path() / "pboc_20240708").iterdir()
    # today = "report"
    # fields_file = "pboc_fields.xlsx"
    # aggconf_file = "pboc_aggconf.xlsx"
    # vars_file = "pboc_vars.xlsx"
    # agg_keys = None

    report_recs = {}
    for file in files:
        report = open(file, "r").read()
        report_id = extract_field(report, "PRH:PA01:PA01A:PA01AI01")
        report_recs[report_id] = report

    dfs = pboc_fields(pd.Series(report_recs), fields_file)

    if today == "report":
        basic_info = dfs["pboc_basic_info"]
        report_dates = basic_info.set_index("PA01AI01")["PA01AR01"].rename("today")
        for part_name, df in dfs.items():
            if part_name == "pboc_basic_info" or df.empty:
                continue
            df = pd.merge(df, report_dates, how="left",
                          left_on="rid", right_index=True)
            dfs[part_name] = df

    if isinstance(agg_keys, (str, os.PathLike)) and os.path.isfile(agg_keys):
        agg_keys = pd.read_excel(agg_keys)["key"]

    ret = pboc_vars(dfs, agg_keys, aggconf_file, vars_file)

    return dfs, ret


# %%
if __name__ == "__main__":
    # PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
    files = (get_assets_path() / "pboc_reports").iterdir()
    today = "report"
    fields_file = "pboc_fields.xlsx"
    aggconf_file = "pboc_aggconf.xlsx"
    vars_file = "pboc_vars.xlsx"
    agg_keys_file = get_assets_path() / "pboc_aggconf_mark.xlsx"
    fields, aggs = from_files(files, today,
                              fields_file,
                              aggconf_file,
                              vars_file)
