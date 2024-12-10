#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-04-22 10:13:57
#   Updated: 2024-12-10 22:31:27
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
from collections.abc import Mapping, Sequence
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from modsbear.dflater import ex2df, ex4df, exenv, exoptim
    from modsbear.spanner import manidf
    from suitbear.dirt import exdf
    from suitbear.pboc import confflat, confagg, conftrans, confmark
    reload(ex2df)
    reload(ex4df)
    reload(exenv)
    reload(exoptim)
    reload(manidf)
    reload(exdf)
    reload(confflat)
    reload(conftrans)
    reload(confagg)
    reload(confmark)

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
from suitbear.dirt.exdf import trans_from_dfs, agg_from_dfs, dep_from_fconfs
from suitbear.pboc.confflat import PBOC_PARTS, df_flat_confs
from suitbear.pboc.conftrans import MAPPERS, MAPPERS_CODE, df_trans_confs
from suitbear.pboc.confagg import df_agg_confs, LV2_AGG_CONF, LV20_AGG_CONF, LV1_AGG_CONF
from suitbear.pboc.confmark import df_mark_confs

# MAPPERS_CODE["today"] = pd.Timestamp.today()
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
def write_pboc_confs(conf_file: str = "pboc/pboc_conf.xlsx"):
    """Write POBC confs to Excel.
    """
    dfs = {}

    flat_pconfs, flat_fconfs = df_flat_confs()
    dfs["pboc_flat_parts"] = flat_pconfs
    dfs["pboc_flat_fields"] = flat_fconfs

    trans_pconfs, trans_fconfs = df_trans_confs()
    dfs["pboc_trans_parts"] = trans_pconfs
    dfs["pboc_trans_fields"] = trans_fconfs
    mappers = pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()],
                        axis=0,
                        keys=MAPPERS.keys())
    dfs["pboc_trans_maps"] = mappers

    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    dfs["pboc_agg_parts"] = agg_pconfs
    dfs["pboc_agg_fields"] = agg_fconfs

    mark_pconfs, mark_fconfs = df_mark_confs()
    dfs["pboc_mark_parts"] = mark_pconfs
    dfs["pboc_mark_fields"] = mark_fconfs

    # Compile to get only necessary fields for `mark_fconfs`.
    cpconfs = pd.concat([trans_pconfs, agg_pconfs, mark_pconfs], axis=0)
    cfconfs = pd.concat([trans_fconfs, agg_fconfs, mark_fconfs], axis=0)
    targets = mark_fconfs["key"].tolist()
    ncconfs = dep_from_fconfs(targets, cfconfs)
    dfs["pboc_all_parts"] = cpconfs
    dfs["pboc_all_fields"] = cfconfs
    dfs["pboc_nece_fields"] = ncconfs

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
    key_mark: Sequence | str = "min",
    env: Mapping = MAPPERS_CODE,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Calculate indicators from PBOC records.

    Params:
    ---------------------------
    dfs: Series of PBOC records.
    agg_key_mark: The necessary aggregation to be calculated.

    Return:
    ---------------------------
    dfs: Dict[part-name, DF updated by transformation result]
    trans_ret: Dict[part-name, DF with transformation result]
    agg_ret: Dict[part-name, DF of aggregation result]
    mark_ret: Dict[part-name, DF of mark]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    trans_pconfs, trans_fconfs = df_trans_confs()
    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    mark_pconfs, mark_fconfs = df_mark_confs()

    if not np.isscalar(key_mark):
        agg_fconfs = agg_fconfs[agg_fconfs["key"].isin(key_mark)]
        cfconfs = pd.concat([trans_fconfs, agg_fconfs, mark_fconfs], axis=0)
    elif key_mark == "max":
        cfconfs = pd.concat([trans_fconfs, agg_fconfs, mark_fconfs], axis=0)
    else:
        cfconfs = pd.concat([trans_fconfs, agg_fconfs, mark_fconfs], axis=0)
        targets = mark_fconfs["key"].tolist()
        cfconfs = dep_from_fconfs(targets, cfconfs, envp=envp)

    # 1. Apply in-place transformation on DataFrames in `dfs`.
    # Transfrom with `auto` or `inplace` will be all fine, as the `from_`s in
    # `trans_pconfs` are all len-1 list.
    trans_ret = trans_from_dfs(dfs, trans_pconfs, cfconfs,
                               how="auto", envp=envp)

    # 2. Construct and apply aggregation config.
    # Only transformed parts will be registered in `trans_ret`, so the original
    # `dfs` with transformed parts will be used.
    agg_ret = agg_from_dfs(dfs, agg_pconfs, cfconfs, envp=envp)
    # agg_ret = {}
    # agg_ret = agg_from_dfs(dfs, agg_pconfs, agg_fconfs, envp=envp,
    #                        agg_rets=agg_ret)

    # 3. Generate marks.
    mark_ret = trans_from_dfs(ChainMap(agg_ret, dfs),
                              mark_pconfs, cfconfs, how="new",
                              envp=envp)

    return dfs, trans_ret, agg_ret, mark_ret


# %%
if __name__ == "__main__":
    from flagbear.slp.pdsl import save_with_pickle, load_from_pickle

    # PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
    pboc_conf_file = "pboc/pboc_conf.xlsx"
    write_pboc_confs(pboc_conf_file)

    # Read and flatten reports.
    files = list((get_assets_path() / "pboc_reports").iterdir())
    report_recs = {}
    for file in files:
        report = open(file, "r").read()
        report_id = extract_field(report, "PRH:PA01:PA01A:PA01AI01")
        report_recs[report_id] = report
    today = "report"
    dfs = flat_fields(pd.Series(report_recs), today)
    # save_with_pickle(dfs, "pboc/flat_dfs")
    # dfs = load_from_pickle("pboc/flat_dfs")

    # Read aggregation keys.
    # agg_keys = get_assets_path() / "pboc_aggconf_mark.xlsx"
    # if isinstance(agg_keys, (str, os.PathLike)) and os.path.isfile(agg_keys):
    #     agg_keys = pd.read_excel(agg_keys)["key"]
    key_mark = "min"
    dfs, trans_ret, agg_ret, mark_ret = pboc_vars(dfs, key_mark)
    assert len(trans_ret.keys() - dfs.keys()) == 0
    # save_with_pickle(agg_ret, "pboc/agg_dfs")
    # agg_ret = load_from_pickle("pboc/agg_dfs")

    flats_file = f"pboc/pboc_flats_{key_mark}.xlsx"
    aggs_file = f"pboc/pboc_aggs_{key_mark}.xlsx"
    marks_file = f"pboc/pboc_marks_{key_mark}.xlsx"
    save_with_excel(dfs, flats_file)
    save_with_excel(agg_ret, aggs_file)
    save_with_excel(mark_ret, marks_file)
