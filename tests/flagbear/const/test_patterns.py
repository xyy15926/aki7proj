#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_patterns.py
#   Author: xyy15926
#   Created: 2023-12-11 20:31:04
#   Updated: 2023-12-11 21:30:23
#   Description:
# ---------------------------------------------------------

# %%
import re
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear.const import patterns
    reload(patterns)

from flagbear.const.patterns import REGEXS


# %%
def test_regexs_int():
    assert re.fullmatch(REGEXS["int"], "0") is not None
    assert re.fullmatch(REGEXS["int"], "2342") is not None
    assert re.fullmatch(REGEXS["int"], "-2342") is None
    assert re.fullmatch(REGEXS["int"], "+2342") is None
    assert re.fullmatch(REGEXS["int"], "2,342") is None
    assert re.fullmatch(REGEXS["int"], "2.342") is None

    assert re.fullmatch(REGEXS["sint"], "0") is not None
    assert re.fullmatch(REGEXS["sint"], "2342") is not None
    assert re.fullmatch(REGEXS["sint"], "-2342") is not None
    assert re.fullmatch(REGEXS["sint"], "+2342") is not None
    assert re.fullmatch(REGEXS["sint"], "2,342") is None
    assert re.fullmatch(REGEXS["sint"], "2.342") is None

    assert re.fullmatch(REGEXS["mint"], "0") is not None
    assert re.fullmatch(REGEXS["mint"], "2342") is not None
    assert re.fullmatch(REGEXS["mint"], "-2342") is not None
    assert re.fullmatch(REGEXS["mint"], "+2342") is not None
    assert re.fullmatch(REGEXS["mint"], "2,342") is not None
    assert re.fullmatch(REGEXS["mint"], "2.342") is None


def test_regexs_float():
    assert re.fullmatch(REGEXS["float"], "2342") is None
    assert re.fullmatch(REGEXS["float"], ".23") is not None
    assert re.fullmatch(REGEXS["float"], "0.23") is not None
    assert re.fullmatch(REGEXS["float"], "2342.23") is not None
    assert re.fullmatch(REGEXS["float"], "-2342.23") is None
    assert re.fullmatch(REGEXS["float"], "+2342.23") is None
    assert re.fullmatch(REGEXS["float"], "2,342.23") is None
    assert re.fullmatch(REGEXS["float"], "-2,342.23") is None
    assert re.fullmatch(REGEXS["float"], "+2,342.23") is None

    assert re.fullmatch(REGEXS["sfloat"], "2342") is None
    assert re.fullmatch(REGEXS["sfloat"], ".23") is not None
    assert re.fullmatch(REGEXS["sfloat"], "0.23") is not None
    assert re.fullmatch(REGEXS["sfloat"], "2342.23") is not None
    assert re.fullmatch(REGEXS["sfloat"], "-2342.23") is not None
    assert re.fullmatch(REGEXS["sfloat"], "+2342.23") is not None
    assert re.fullmatch(REGEXS["sfloat"], "2,342.23") is None
    assert re.fullmatch(REGEXS["sfloat"], "-2,342.23") is None
    assert re.fullmatch(REGEXS["sfloat"], "+2,342.23") is None

    assert re.fullmatch(REGEXS["mfloat"], "2342") is None
    assert re.fullmatch(REGEXS["mfloat"], ".23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "0.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "2342.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "-2342.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "+2342.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "2,342.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "-2,342.23") is not None
    assert re.fullmatch(REGEXS["mfloat"], "+2,342.23") is not None


# %%
def test_date():
    assert re.fullmatch(REGEXS["date"], "2023-01-01") is not None
    assert re.fullmatch(REGEXS["date"], "2023-01-28") is not None
    assert re.fullmatch(REGEXS["date"], "2023-01-29") is not None
    assert re.fullmatch(REGEXS["date"], "2023-01-30") is not None
    assert re.fullmatch(REGEXS["date"], "2023-01-31") is not None

    assert re.fullmatch(REGEXS["date"], "2023/01/01") is not None
    assert re.fullmatch(REGEXS["date"], "2023/01/28") is not None
    assert re.fullmatch(REGEXS["date"], "2023/01/29") is not None
    assert re.fullmatch(REGEXS["date"], "2023/01/30") is not None
    assert re.fullmatch(REGEXS["date"], "2023/01/31") is not None

    assert re.fullmatch(REGEXS["date"], "2023-02-01") is not None
    assert re.fullmatch(REGEXS["date"], "2023-02-28") is not None
    # Leap year is hard to handle
    assert re.fullmatch(REGEXS["date"], "2023-02-29") is not None
    assert re.fullmatch(REGEXS["date"], "2023-02-30") is None
    assert re.fullmatch(REGEXS["date"], "2023-02-31") is None

    assert re.fullmatch(REGEXS["date"], "2024-02-01") is not None
    assert re.fullmatch(REGEXS["date"], "2024-02-28") is not None
    assert re.fullmatch(REGEXS["date"], "2024-02-29") is not None
    assert re.fullmatch(REGEXS["date"], "2024-02-30") is None
    assert re.fullmatch(REGEXS["date"], "2024-02-31") is None

    assert re.fullmatch(REGEXS["date"], "2023-11-01") is not None
    assert re.fullmatch(REGEXS["date"], "2023-11-28") is not None
    assert re.fullmatch(REGEXS["date"], "2023-11-29") is not None
    assert re.fullmatch(REGEXS["date"], "2023-11-30") is not None
    assert re.fullmatch(REGEXS["date"], "2023-11-31") is None

    assert re.fullmatch(REGEXS["date"], "2023-12-01") is not None
    assert re.fullmatch(REGEXS["date"], "2023-12-28") is not None
    assert re.fullmatch(REGEXS["date"], "2023-12-29") is not None
    assert re.fullmatch(REGEXS["date"], "2023-12-30") is not None
    assert re.fullmatch(REGEXS["date"], "2023-12-31") is not None


# %%
def test_regexs_idcard():
    assert re.fullmatch(REGEXS["idcard"], "360425199101011234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199101281234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199101291234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199101301234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199101311234") is not None
    assert re.fullmatch(REGEXS["idcard"], "36042519910128123X") is not None
    assert re.fullmatch(REGEXS["idcard"], "36042519910129123x") is not None

    assert re.fullmatch(REGEXS["idcard"], "360425199102011234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199102281234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199102291234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425199102301234") is None
    assert re.fullmatch(REGEXS["idcard"], "360425199102311234") is None
    assert re.fullmatch(REGEXS["idcard"], "36042519910228123X") is not None
    assert re.fullmatch(REGEXS["idcard"], "36042519910229123x") is not None

    assert re.fullmatch(REGEXS["idcard"], "360425210002011234") is None
    assert re.fullmatch(REGEXS["idcard"], "360425179902281234") is None
    assert re.fullmatch(REGEXS["idcard"], "360425180102291234") is not None
    assert re.fullmatch(REGEXS["idcard"], "360425209902291234") is not None


# %%
def test_regexs_mobile():
    assert re.fullmatch(REGEXS["mobile"], "13111112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13311112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13411112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13911112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "13011112222") is not None

    assert re.fullmatch(REGEXS["mobile"], "14111112222") is None
    assert re.fullmatch(REGEXS["mobile"], "14211112222") is None
    assert re.fullmatch(REGEXS["mobile"], "14311112222") is None
    assert re.fullmatch(REGEXS["mobile"], "14411112222") is None
    assert re.fullmatch(REGEXS["mobile"], "14511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "14611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "14711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "14811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "14911112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "14011112222") is None

    assert re.fullmatch(REGEXS["mobile"], "15111112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15311112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15411112222") is None
    assert re.fullmatch(REGEXS["mobile"], "15511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15911112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "15011112222") is not None

    assert re.fullmatch(REGEXS["mobile"], "16111112222") is None
    assert re.fullmatch(REGEXS["mobile"], "16211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "16311112222") is None
    assert re.fullmatch(REGEXS["mobile"], "16411112222") is None
    assert re.fullmatch(REGEXS["mobile"], "16511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "16611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "16711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "16811112222") is None
    assert re.fullmatch(REGEXS["mobile"], "16911112222") is None
    assert re.fullmatch(REGEXS["mobile"], "16011112222") is None

    assert re.fullmatch(REGEXS["mobile"], "17111112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17311112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17411112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "17911112222") is None
    assert re.fullmatch(REGEXS["mobile"], "17011112222") is not None

    assert re.fullmatch(REGEXS["mobile"], "18111112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18311112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18411112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18911112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "18011112222") is not None

    assert re.fullmatch(REGEXS["mobile"], "19111112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19211112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19311112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19411112222") is None
    assert re.fullmatch(REGEXS["mobile"], "19511112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19611112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19711112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19811112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19911112222") is not None
    assert re.fullmatch(REGEXS["mobile"], "19011112222") is not None


# %%
def test_regexs_na():
    assert re.fullmatch(REGEXS["na"], "NA") is not None
    assert re.fullmatch(REGEXS["na"], "na") is not None
    assert re.fullmatch(REGEXS["na"], "Na") is not None
    assert re.fullmatch(REGEXS["na"], "nA") is not None
    assert re.fullmatch(REGEXS["na"], "NAT") is not None
    assert re.fullmatch(REGEXS["na"], "nat") is not None
    assert re.fullmatch(REGEXS["na"], "NaT") is not None
    assert re.fullmatch(REGEXS["na"], "NAN") is not None
    assert re.fullmatch(REGEXS["na"], "nan") is not None
    assert re.fullmatch(REGEXS["na"], "NaN") is not None
    assert re.fullmatch(REGEXS["na"], "NN") is None
    assert re.fullmatch(REGEXS["na"], "19111112222") is None


# %%
def test_regexs_time():
    assert re.fullmatch(REGEXS["time"], "00:00:00") is not None
    assert re.fullmatch(REGEXS["time"], "00:00:01") is not None
    assert re.fullmatch(REGEXS["time"], "00:01:00") is not None
    assert re.fullmatch(REGEXS["time"], "01:01:00") is not None
    assert re.fullmatch(REGEXS["time"], "01:01:01") is not None

    assert re.fullmatch(REGEXS["time"], "20:00:00") is not None
    assert re.fullmatch(REGEXS["time"], "20:00:01") is not None
    assert re.fullmatch(REGEXS["time"], "20:01:00") is not None
    assert re.fullmatch(REGEXS["time"], "21:01:00") is not None
    assert re.fullmatch(REGEXS["time"], "21:01:01") is not None

    assert re.fullmatch(REGEXS["time"], "24:00:00") is None
    assert re.fullmatch(REGEXS["time"], "24:00:01") is None
    assert re.fullmatch(REGEXS["time"], "24:01:00") is None
    assert re.fullmatch(REGEXS["time"], "24:01:00") is None
    assert re.fullmatch(REGEXS["time"], "24:01:01") is None

    assert re.fullmatch(REGEXS["time"], "00:00:59") is not None
    assert re.fullmatch(REGEXS["time"], "00:00:59") is not None
    assert re.fullmatch(REGEXS["time"], "00:01:59") is not None
    assert re.fullmatch(REGEXS["time"], "01:01:59") is not None
    assert re.fullmatch(REGEXS["time"], "01:01:59") is not None

    assert re.fullmatch(REGEXS["time"], "00:55:59") is not None
    assert re.fullmatch(REGEXS["time"], "00:55:59") is not None
    assert re.fullmatch(REGEXS["time"], "00:55:59") is not None
    assert re.fullmatch(REGEXS["time"], "01:55:59") is not None
    assert re.fullmatch(REGEXS["time"], "01:55:59") is not None

    assert re.fullmatch(REGEXS["time"], "00:00:60") is None
    assert re.fullmatch(REGEXS["time"], "00:00:60") is None
    assert re.fullmatch(REGEXS["time"], "00:01:60") is None
    assert re.fullmatch(REGEXS["time"], "01:01:60") is None
    assert re.fullmatch(REGEXS["time"], "01:01:60") is None

    assert re.fullmatch(REGEXS["time"], "00:00:69") is None
    assert re.fullmatch(REGEXS["time"], "00:00:69") is None
    assert re.fullmatch(REGEXS["time"], "00:01:69") is None
    assert re.fullmatch(REGEXS["time"], "01:01:69") is None
    assert re.fullmatch(REGEXS["time"], "01:01:69") is None


# %%
def test_regexs_interval():
    assert re.fullmatch(REGEXS["interval"], "(1, 0)") is not None
    assert re.fullmatch(REGEXS["interval"], "(-1, 0)") is not None
    assert re.fullmatch(REGEXS["interval"], "(-1, +1)") is not None
    assert re.fullmatch(REGEXS["interval"], "(-1.0, .1)") is not None

    assert re.fullmatch(REGEXS["interval"], "[1, 0)") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1, 0)") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1, +1)") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1.0, .1)") is not None

    assert re.fullmatch(REGEXS["interval"], "[1, 0]") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1, 0]") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1, +1]") is not None
    assert re.fullmatch(REGEXS["interval"], "[-1.0, .1]") is not None

    assert re.fullmatch(REGEXS["interval"], "[1,0]") is not None
    assert re.fullmatch(REGEXS["interval"], "[1,  0]") is not None
    assert re.fullmatch(REGEXS["interval"], "[ 1, 0]") is None
    assert re.fullmatch(REGEXS["interval"], "[ 1, 0 ]") is None


# %%
def test_regexs_fset():
    assert re.fullmatch(REGEXS["set"], "{1}") is not None
    assert re.fullmatch(REGEXS["set"], "{1, 2}") is not None
    assert re.fullmatch(REGEXS["set"], "{1,2}") is not None
    assert re.fullmatch(REGEXS["set"], "{1.0,2}") is not None
    assert re.fullmatch(REGEXS["set"], "{1.0,abc}") is not None
    assert re.fullmatch(REGEXS["set"], "{ 1.0,abc}") is None
