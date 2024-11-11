#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-04-22 10:13:57
#   Updated: 2024-11-11 20:15:07
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

import os
from collections import ChainMap
import sqlalchemy as sa
from tqdm import tqdm
from IPython.core.debugger import set_trace

from flagbear.llp.patterns import REGEX_TOKEN_SPECS
from flagbear.llp.parser import EnvParser
from flagbear.str2.fliper import extract_field
from flagbear.slp.finer import get_assets_path, get_tmp_path
from flagbear.slp.pdsl import save_with_excel
from modsbear.spanner.manidf import merge_dfs
from modsbear.dflater.ex2df import compress_hierarchy, flat_records
from modsbear.dflater.ex4df import trans_on_df, agg_on_df
from modsbear.dflater.exenv import EXGINE_ENV
from suitbear.dirt.exdf import agg_from_dfs
from suitbear.pboc.confflat import PBOC_PARTS, df_flat_confs
from suitbear.pboc.conftrans import MAPPERS, MAPPERS_CODE, TRANS_CONF, df_trans_confs
from suitbear.pboc.confagg import df_agg_confs, LV2_AGG_CONF, LV20_AGG_CONF, LV1_AGG_CONF

MAPPERS_CODE["today"] = pd.Timestamp.today()
PBOC_AGG_CONF = {**LV2_AGG_CONF, **LV20_AGG_CONF,
                 **LV1_AGG_CONF}
# agg_pconfs, agg_aconfs = agg_confs()
# flat_parts, flat_fields = flat_confs()


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def write_pboc_confs(conf_file: str = "pboc/pboc_aggconf"):
    """Write POBC parsing confs to Excel.
    """
    dfs = {}

    trans_confs = df_trans_confs()
    dfs["pboc_vars_trans"] = trans_confs
    mappers = pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()],
                        axis=0,
                        keys=MAPPERS.keys())
    dfs["pboc_vars_maps"] = mappers

    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    dfs["pboc_vars_parts"] = agg_pconfs
    dfs["pboc_vars_aggs"] = agg_fconfs

    save_with_excel(dfs, conf_file)


# %%
def flat_fields(
    src: pd.Series,
    today: str = "report",
) -> dict[str, pd.DataFrame]:
    """Extract fields from PBOC records.

    Params:
    ---------------------------
    src: Series of PBOC records.

    Return:
    ---------------------------
    Dict[part-name, DataFrame of values of parts]
    """
    # Read fields extraction config and addup some default settings.

    # Extract fields.
    dfs = {}
    for pconf in PBOC_PARTS.values():
        psrc = compress_hierarchy(src, pconf["steps"])
        fconf = []
        for key, step, dtype, desc in pconf["fields"]:
            # Set the default values to get the proper dtype for DataFrame
            # constructed in `flat_records` automatically.
            if dtype in REGEX_TOKEN_SPECS:
                fconf.append((key, None, step, dtype,
                              REGEX_TOKEN_SPECS[dtype][-1]))
            else:
                fconf.append((key, step))
        ret = flat_records(psrc, fconf)
        dfs[pconf["part"]] = ret

    if today == "report":
        basic_info = dfs["pboc_basic_info"]
        report_dates = basic_info.set_index("PA01AI01")["PA01AR01"].rename("today")
        for part_name, df in dfs.items():
            if part_name == "pboc_basic_info" or df.empty:
                continue
            df = pd.merge(df, report_dates, how="left",
                          left_on="rid", right_index=True)
            dfs[part_name] = df

    return dfs


# %%
def pboc_vars(
    dfs: dict[str, pd.DataFrame],
    agg_key_mark: pd.Series | set | list = None,
    env: Mapping = MAPPERS_CODE,
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
        if not df.empty:
            dfs[part_name] = trans_on_df(df, trans_rules, envp=envp)

    # 2. Construct and apply aggregation config.
    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    if agg_key_mark is not None:
        agg_fconfs = agg_fconfs[agg_fconfs["key"].isin(agg_key_mark)]
    ret = agg_from_dfs(dfs, agg_pconfs, agg_fconfs, envp=envp)

    return dfs, ret


# %%
if __name__ == "__main__":
    # PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
    aggconf_file = "pboc/pboc_aggconf.xlsx"
    write_pboc_confs(aggconf_file)

    # Read and flatten reports.
    files = list((get_assets_path() / "pboc_reports").iterdir())
    report_recs = {}
    for file in files:
        report = open(file, "r").read()
        report_id = extract_field(report, "PRH:PA01:PA01A:PA01AI01")
        report_recs[report_id] = report
    today = "report"
    dfs = flat_fields(pd.Series(report_recs), today)

    # Read aggregation keys.
    # agg_keys = get_assets_path() / "pboc_aggconf_mark.xlsx"
    # if isinstance(agg_keys, (str, os.PathLike)) and os.path.isfile(agg_keys):
    #     agg_keys = pd.read_excel(agg_keys)["key"]
    agg_keys = None

    # Apply aggregation.
    fields_file = "pboc/pboc_fields.xlsx"
    vars_file = "pboc/pboc_vars.xlsx"
    dfs, agg_rets = pboc_vars(dfs, agg_keys)
    save_with_excel(dfs, fields_file)
    save_with_excel(agg_rets, vars_file)
