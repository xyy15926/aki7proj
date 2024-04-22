#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_crosconf.py
#   Author: xyy15926
#   Created: 2024-04-19 21:11:08
#   Updated: 2024-04-22 09:40:48
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from suitbear import crosconf
    reload(crosconf)

from suitbear.crosconf import cproduct_aggs_and_filters, cross_aggs_and_filters
from suitbear.crosconf import cross_aggs_from_lower, agg_confs_from_dict

# %%
def test_cproduct_aggs_and_filters():
    aggs = {
        "cnt": ("cnt"       , "count(_)"        , "数量"),
        "max": ("max"       , "max(acc_lmt)"    , "额度最大值"  , ["amt_max"]),
        "sum": ("sum"       , "sum(acc_lmt)"    , "额度和"      , ["amt_sum", "amt_max"]),
    }
    filters = {
        "orgs": [
            ("org_bank"     , "acc_org_cat < 20"        , "银行账户"),
            ("org_nbank"    , "(acc_org_cat > 20) & (acc_org_cat < 60)" , "非银账户"),
            ("org_other"    , "acc_org_cat > 90"        , "其他机构账户")
        ],
        "mixed_lvl5_status": [
            ("lvl5_nor"     , "mixed_lvl5_status == 1"      , "五级分类正常"),
            ("lvl5_con"     , "mixed_lvl5_status == 2"      , "五级分类关注"),
            ("lvl5_inf"     , "mixed_lvl5_status >= 3"      , "五级分类次级及以上"),
        ],
    }
    agg_conf = cproduct_aggs_and_filters(aggs.values(), filters.values(), "{}_{}")
    cond_prod_n = 1
    for filter_ in filters.values():
        cond_prod_n *= len(filter_)
    assert(agg_conf.shape[0] == len(aggs) * cond_prod_n)


# %%
def test_cross_aggs_and_filters():
    aggs = {
        "cnt": ("cnt"       , "count(_)"        , "数量"),
        "max": ("max"       , "max(acc_lmt)"    , "额度最大值"  , ["amt_max"]),
        "sum": ("sum"       , "sum(acc_lmt)"    , "额度和"      , ["amt_sum", "amt_max"]),
    }
    filters = {
        "orgs": {
            "bank": ("org_bank"     , "acc_org_cat < 20"        , "银行账户"),
            "nbank": ("org_nbank"   , "(acc_org_cat > 20) & (acc_org_cat < 60)" , "非银账户"),
            "other": ("org_other"   , "acc_org_cat > 90"        , "其他机构账户")
        },
        "mixed_lvl5_status": [
            ("lvl5_nor"     , "mixed_lvl5_status == 1"      , "五级分类正常"),
            ("lvl5_con"     , "mixed_lvl5_status == 2"      , "五级分类关注"),
            ("lvl5_inf"     , "mixed_lvl5_status >= 3"      , "五级分类次级及以上"),
        ],
    }
    cros = [
        (["cnt", ]          , ["orgs", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , ["orgs", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , ["orgs.[bank, nbank]", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , [("orgs", "bank", "nbank"), "mixed_lvl5_status"]),
    ]

    agg_conf_1 = cross_aggs_and_filters(cros[:1], aggs, filters, "{}_{}")
    agg_conf_2 = cross_aggs_and_filters(cros[1:2], aggs, filters, "{}_{}")
    assert len(agg_conf_2) == len(agg_conf_1) * 2

    agg_conf_3 = cross_aggs_and_filters(cros[2:3], aggs, filters, "{}_{}")
    agg_conf_4 = cross_aggs_and_filters(cros[3:4], aggs, filters, "{}_{}")
    assert len(agg_conf_3) == len(agg_conf_4)
    assert len(agg_conf_3) * 3 == len(agg_conf_2) * 2

    return cross_aggs_and_filters(cros, aggs, filters, "{}_{}")


# %%
def test_agg_confs_from_dict():
    aggs = {
        "cnt": ("cnt"       , "count(_)"        , "数量"),
        "max": ("max"       , "max(acc_lmt)"    , "额度最大值"  , ["amt_max"]),
        "sum": ("sum"       , "sum(acc_lmt)"    , "额度和"      , ["amt_sum", "amt_max"]),
    }
    filters = {
        "orgs": {
            "bank": ("org_bank"     , "acc_org_cat < 20"        , "银行账户"),
            "nbank": ("org_nbank"   , "(acc_org_cat > 20) & (acc_org_cat < 60)" , "非银账户"),
            "other": ("org_other"   , "acc_org_cat > 90"        , "其他机构账户")
        },
        "mixed_lvl5_status": [
            ("lvl5_nor"     , "mixed_lvl5_status == 1"      , "五级分类正常"),
            ("lvl5_con"     , "mixed_lvl5_status == 2"      , "五级分类关注"),
            ("lvl5_inf"     , "mixed_lvl5_status >= 3"      , "五级分类次级及以上"),
        ],
    }
    cros = [
        (["cnt", ]          , ["orgs", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , ["orgs", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , ["orgs.[bank, nbank]", "mixed_lvl5_status"]),
        (["cnt", "sum"]     , [("orgs", "bank", "nbank"), "mixed_lvl5_status"]),
    ]
    conf = {
        "foo": {
            "part": "foo",
            "level": 1,
            "from_": None,
            "prikey": None,
            "aggs": aggs,
            "conds": filters,
            "cros": cros,
            "key_fmt": "foo_{}_{}",
        },
        "bar": {
            "part": "bar",
            "level": 1,
            "from_": None,
            "prikey": None,
            "aggs": aggs,
            "conds": filters,
            "cros": cros,
            "key_fmt": "bar_{}_{}",
        }
    }
    pconf, aconf = agg_confs_from_dict(conf)
    assert len(pconf) == len(conf)


# %%
def test_cross_aggs_from_lower():
    aggs = {
        "sum": ("{}_sum", "sum({})", "{}_之和"),
        "max": ("{}_max", "max({})", "{}_最大值"),
        "amt_sum": ("{}_sum", "smul(sum({}), acc_exchange_rate)", "{}_之和（本币）"),
        "amt_max": ("{}_sum", "smul(max({}), acc_exchange_rate)", "{}_最大值（本币）"),
    }
    filters_D = {
        "acc_cat": {
            "r2": ("r2"         , "acc_cat == 4"                        , "r2"),
            "d1r41": ("d1r41"   , "acc_cat <= 3"                        , "d1r41"),
            "r23": ("r23"       , "(acc_cat >= 4) & (acc_cat <= 5)"     , "r23"),
        }
    }
    cros_D = {
        "sum": ["acc_cat", ],
        "max": ["acc_cat", ],
        "amt_sum": ["acc_cat", ],
        "amt_max": ["acc_cat", ],
    }
    lower_aconfs = test_cross_aggs_and_filters()
    upper_aconfs = lower_aconfs.apply(cross_aggs_from_lower,
                                      axis=1,
                                      cros_D=cros_D,
                                      aggs_D=aggs,
                                      filters_D=filters_D)
    upper_aconfs = pd.concat(upper_aconfs.values)
    assert (lower_aconfs["lvup_flags"]
            .fillna("").apply(len).sum() * 3 == upper_aconfs.shape[0])
