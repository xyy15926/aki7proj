#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: repayment.py
#   Author: xyy15926
#   Created: 2024-04-09 20:02:08
#   Updated: 2025-05-13 10:39:22
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear.timser import ovdd
    from ubears.suitbear.monetary import repayment
    from ubears.modsbear.spanner import manidf
    reload(ovdd)
    reload(repayment)
    reload(manidf)

from ubears.ringbear.timser.ovdd import month_date
from ubears.modsbear.spanner.manidf import merge_dfs
from ubears.suitbear.monetary.repayment import (
    DUM_OVDD,
    DUM_OVDP,
    ob4ovd,
    addup_obovd,
    addup_obrec,
    edge_crosstab,
    mob_align,
)


# %%
def ovd_recs():
    recs = pd.DataFrame([
        ("2021-12-11"   , 0     , 0     , 2300),
        ("2022-01-11"   , 23    , 100   , 2200),
        # Repay date overpassing the observation date within the next duepay date.
        ("2022-02-11"   , 0     , 100   , 2100),
        # Repay date overpassing the next duepay date.
        ("2022-03-11"   , 37    , 100   , 2000),
        # 37 - 31 = 6 > 4: Invalid overdue days.
        # Repay date within the observation date.
        ("2022-04-11"   , 4     , 100   , 1900),
        ("2022-05-11"   , 35    , 100   , 1800),
        ("2022-06-11"   , 75    , 100   , 1700),
        # Increasing overdue days over 4 periods.
        ("2022-07-11"   , 98    , 100   , 1600),
        ("2022-08-11"   , 75    , 100   , 1500),
        ("2022-09-11"   , 46    , 100   , 1400),
        # Repay date overpassing the observation date within the next duepay date.
        ("2022-10-11"   , 20    , 100   , 1300),
        ("2022-11-11"   , 0     , 100   , 1200),
        ("2022-12-11"   , 17    , 100   , 1100),
        ("2023-01-11"   , 17    , 100   , 1000),
        ("2023-02-11"   , 28    , 100   , 900),
        ("2023-03-11"   , 31    , 100   , 800),
        ("2023-04-11"   , 0     , 100   , 700),
        # No ending tail.
        ("2023-05-11"   , 34    , 100   , 600),
    ], columns=["due_date", "ovd_days", "due_amt", "rem_amt"])
    recs["due_date"] = recs["due_date"].astype("M8[s]")

    return recs


# %%
def test_ob4ovd():
    recs = ovd_recs()
    due_date = np.asarray(recs["due_date"], "M8[D]")
    obd = np.concatenate([due_date + 1,
                          due_date + 16,
                          month_date(due_date, "monthend")])
    obd.sort()
    obret = ob4ovd(recs, obd)
    assert obd.shape[0] == obret.shape[0]

    recs["rep_date"] = recs["due_date"] + recs["ovd_days"].astype("m8[D]")
    recs_rep = recs.sort_values("rep_date")
    # Merge observation records with the correspondant records that really
    # determine the overdue days.
    merged_pd = pd.merge_asof(obret, recs_rep,
                              left_on="ob_date",
                              right_on="rep_date",
                              direction="forward")
    merged_ubears = merge_dfs([obret, recs_rep],
                              ["ob_date", "rep_date"],
                              direction="forward",
                              tolerance=None)
    assert np.all(merged_pd == merged_ubears)
    # One wrong verfication was placed knowingly for `2022-04-11`, which this
    # bill was written off before `2022-03-11`.
    # So the date must excluded first.
    right_merged = merged_ubears[merged_ubears["due_date"]
                                 != pd.Timestamp("2022-04-11")]
    assert np.all(right_merged["ovd_days"] >= right_merged["stop_ovdd"])


# %%
def test_addup_obovd():
    recs = ovd_recs()
    recs.index = list(range(100, 118))
    df = addup_obovd(recs)
    assert df.loc[100, "ever_ovdd"] == np.timedelta64(0, "D")
    assert df.loc[100, "ever_ovdp"] == 0
    assert np.all(df["MOB"].values == np.arange(len(df)))

    recs["START"] = pd.Timestamp("2021-10")
    moved_df = addup_obovd(recs)
    assert np.all(moved_df["MOB"].values > np.arange(len(df)))

    noend_df = addup_obovd(recs, ob_date="nextdue_noend")
    end_df = addup_obovd(recs, ob_date="nextdue")
    assert np.all(noend_df.iloc[:-1, noend_df.columns.get_loc("ever_ovdd")]
                  == end_df.iloc[:-1, end_df.columns.get_loc("ever_ovdd")])
    assert np.all(noend_df.iloc[-1, noend_df.columns.get_loc("ever_ovdd")]
                  > end_df.iloc[-1, end_df.columns.get_loc("ever_ovdd")])

    recs = ovd_recs()
    recs["DUMMY"] = pd.Timestamp("2022-09")
    dum_df = addup_obovd(recs)
    assert np.all(dum_df.loc[10:, "ever_ovdd"] == DUM_OVDD)
    assert np.all(dum_df.loc[10:, "ever_ovdp"] == DUM_OVDP)


def test_addup_obovd_unsorted():
    recs = ovd_recs()
    df = addup_obovd(recs)
    unsorted = [ele for idx, ele in ovd_recs().iterrows()]
    unsorted = pd.DataFrame().from_records(unsorted)

    sorted_df = addup_obovd(unsorted)
    for col in sorted_df:
        assert np.all(sorted_df[col].values == df[col].values)


# %%
def test_addup_obrec():
    rec_one = ovd_recs()
    N = 5
    recs = pd.concat([rec_one,] * N,
                     keys=[f"OID-{i:03}" for i in range(1, N + 1)],
                     names=["oid", "rid"])
    recs = recs.reset_index(level=0)

    df = addup_obrec(recs)
    df1 = df[df["oid"] == "OID-001"]
    df2 = df[df["oid"] == "OID-002"]

    assert np.all(df1.iloc[:, 1:] == df2.iloc[:, 1:])


# %%
def ob_records(rec_n: int = 101):
    recs = pd.DataFrame([
        ("OID-001"  , "2021-12-11"  , 1),
        ("OID-001"  , "2022-01-11"  , 2),
        ("OID-001"  , "2022-02-11"  , 3),
        ("OID-001"  , "2022-03-11"  , 4),
        ("OID-001"  , "2022-04-11"  , 5),
        ("OID-001"  , "2022-05-11"  , 6),
    ], columns=["oid", "ob_date", "MOB"])
    recs_L = []
    for i in range(rec_n):
        recs_ = recs.copy()
        recs_["ever_ovdp"] = np.random.randint(7, size=len(recs))
        recs_["stop_ovdp"] = np.random.randint(7, size=len(recs))
        recs_["oid"] = [f"OID-{i:03}"] * len(recs)
        start_date = np.random.choice(["2021-12", "2022-01", "2022-02", "2022-03"])
        recs_["ob_date"] = (np.datetime64(start_date, "M")
                            + np.arange(len(recs)).astype("m8[M]")
                            + np.timedelta64(10, "D"))

        recs_L.append(recs_)

    return pd.concat(recs_L)


# %%
def test_edge_crosstab():
    rec_n = 101
    recs = ob_records(rec_n)
    rolls = edge_crosstab(recs, 1)
    rolls = edge_crosstab(recs, 1, 4)
    assert rolls.loc["SUM", "SUM"] == rec_n
    assert np.all(rolls.sum(1) == rolls["SUM"] + 1)
    rolls = edge_crosstab(recs, 1, 4, normalize=False)
    assert np.all(rolls.sum(1) == rolls["SUM"] * 2)

    rolls = edge_crosstab(recs, "2022-03-11")
    rolls = edge_crosstab(recs, "2022-03-11", "2022-04-11")
    assert rolls.loc["SUM", "SUM"] == rec_n

    mrolls = edge_crosstab(recs, "2022-03")
    mrolls = edge_crosstab(recs, "2022-03", "2022-04-11")
    assert np.all(mrolls == rolls)

    rolls = edge_crosstab(recs, 1, factor="stop_ovdp")
    assert rolls.loc["SUM", "SUM"] == rec_n

    srolls = edge_crosstab(recs, 1, values="stop_ovdp", aggfunc="sum")
    rolls = edge_crosstab(recs, 1, values="stop_ovdp", aggfunc="sum")
    assert np.all(np.isclose(srolls, rolls, equal_nan=True))


# %%
def test_mob_align():
    recs = ovd_recs()
    rec_one = addup_obovd(recs)
    N = 5
    recs = pd.concat([rec_one,] * N,
                     keys=[f"OID-{i:03}" for i in range(1, N + 1)],
                     names=["oid", "rid"])
    recs = recs.reset_index(level=0)
    precs = mob_align(recs)

    tags = pd.Series({
        "OID-001": "2021-12-11",
        "OID-002": "2021-12-11",
        "OID-003": "2022-01-11",
        "OID-004": "2022-01-11",
        "OID-005": "2021-12-11",
    })
    tprecs = mob_align(recs, tags)
    assert np.all(precs == tprecs.sum(0))

    # M1+ overdue rate vintage.
    agg_rules = [
        ("cnt", None, "count(_)"),
        ("M1p", "ever_ovdp > 0", "count(_)")
    ]
    trans_rules = [("M1pR", None, "sdiv(M1p, cnt)")]
    precs = mob_align(recs, agg_rules=agg_rules, trans_rules=trans_rules)
    assert len(precs.keys()) == 3
    assert np.all(precs["M1pR"] <= 1)
