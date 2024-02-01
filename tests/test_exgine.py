#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_exgine.py
#   Author: xyy15926
#   Created: 2024-02-01 10:07:31
#   Updated: 2024-02-01 18:35:04
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from flagbear import fliper

    from suitbear import exgine
    from importlib import reload
    reload(fliper)
    reload(exgine)

import numpy as np
import pandas as pd
import os
ASSETS = os.path.join(os.curdir, "assets")
from suitbear.exgine import parse_2df, parse_parts, parse_2stages
from flagbear.fliper import extract_field


# %%
def test_parse2df():
    pboc = open(os.path.join(ASSETS, "pboc_utf8.json"), "r").read()
    src = pd.Series({"xfy": pboc, "xfy2": pboc})
    pconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="parts")
    fconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="fields")

    confs = fconfs[fconfs["part"] == "pboc_basic_info"].copy()
    basic_info = parse_2df(src, confs, 0)
    assert not np.all(basic_info == None)

    confs = fconfs[fconfs["part"] == "pboc_mobile"].copy()
    confs["steps"] = (pconfs.loc[pconfs["part"] == "pboc_mobile", "steps_0"].iloc[0]
                      + ":" + confs["steps"])
    mobile = parse_2df(src, confs, 1)
    assert not np.all(mobile == None)


def test_parse_parts():
    pboc = open(os.path.join(ASSETS, "pboc_utf8.json"), "r").read()
    src = pd.Series({"xfy": pboc, "xfy2": pboc})
    pconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="parts")
    fconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="fields")

    pconf = pconfs.iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc == None)

    pconf = pconfs[pconfs["level"] == 1].iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc == None)

    pconf = pconfs[pconfs["level"] == 2].iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc == None)
    part = pconf["part"]
    confs = fconfs[fconfs["part"] == part].copy()
    dest = parse_2df(psrc, confs, 0)


def test_parse_2stages():
    pboc = open(os.path.join(ASSETS, "pboc_utf8.json"), "r").read()
    src = pd.Series({"xfy": pboc, "xfy2": pboc})
    pconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="parts")
    fconfs = pd.read_excel(os.path.join(ASSETS, "pboc_execution_20240201.xlsx"),
                           sheet_name="fields")

    rets = {}
    for idx, pconf in pconfs.iterrows():
        part = pconf["part"]
        psrc = parse_parts(src, pconf)
        confs = fconfs[fconfs["part"] == part].copy()
        rets[part] = parse_2df(psrc, confs, 0)


