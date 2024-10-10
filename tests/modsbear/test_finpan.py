#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_panels.py
#   Author: xyy15926
#   Created: 2024-04-09 20:02:08
#   Updated: 2024-05-27 17:07:58
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ringbear import tsarr
    from suitbear import finpan
    reload(tsarr)
    reload(finpan)

from ringbear.tsarr import month_date
from suitbear.finpan import pivot_tags, sequeeze_named_columns
from suitbear.finpan import addup_ob_records, DUM_OVDD, DUM_OVDP
from suitbear.finpan import roll_from_ob_records, pivot_ob_records


# %%
def test_pivot_tags():
    def count_n(seq: list, sep=","):
        cm = {}
        for ele in seq:
            for i in ele.split(sep):
                cm.setdefault(i, 0)
                cm[i] += 1
        return cm

    ser = pd.Series(["a,b,c", "a,b", "a,c", "c"])
    pt = pivot_tags(ser)
    cn = count_n(ser)
    assert np.all(pt.sum() == pd.Series(cn))

    ser = pd.Series(["a,b,c", "a,b", "a,c", "c,c"])
    pt = pivot_tags(ser)
    cn = count_n(ser)
    assert np.all(pt.sum() == pd.Series(cn))


# %%
def test_sequeeze_named_columns():
    ser = pd.Series(["a,b,c", "a,b", "a,c", "c", None, ""])
    pt = pivot_tags(ser)
    pt["c.1"] = (1 - pt["c"]).astype(int)
    sequeezed = sequeeze_named_columns(pt)
    assert np.all(sequeezed["c"] == 1)


# %%
def test_addup_ob_records():
    recs = pd.DataFrame([
        ("2021-12-11"   , None  , 0     , 2200),
        ("2022-01-11"   , 0     , 0     , 2200),
        # Repay date overpassing the observation date within the next duepay date.
        ("2022-02-11"   , 23     , 100   , 2100),
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

    df = addup_ob_records(recs)
    assert df.loc[0, "ever_ovdd"] == np.timedelta64(0, "D")
    assert df.loc[0, "ever_ovdp"] == 0
    assert np.all(df["MOB"].values == np.arange(len(df)))

    moved_df = addup_ob_records(recs, start_date="2021-10")
    assert np.all(moved_df["MOB"].values > np.arange(len(df)))

    end_df = addup_ob_records(recs, ob_date="nextdue")
    assert np.all(df.iloc[:-1, df.columns.get_loc("ever_ovdd")]
                  == end_df.iloc[:-1, end_df.columns.get_loc("ever_ovdd")])
    assert np.all(df.iloc[-1, df.columns.get_loc("ever_ovdd")]
                  > end_df.iloc[-1, end_df.columns.get_loc("ever_ovdd")])

    dum_df = addup_ob_records(recs, dum_mon="2022-09")
    assert np.all(dum_df.loc[10:, "ever_ovdd"].astype("m8[D]") == DUM_OVDD)
    assert np.all(dum_df.loc[10:, "ever_ovdp"] == DUM_OVDP)

    return df.join(recs)


def test_addup_ob_records_sort():
    recs = pd.DataFrame([
        ("2022-05-11"   , 35    , 100   , 1800),
        ("2021-12-11"   , None  , 0     , 2200),
        ("2022-01-11"   , 0     , 0     , 2200),
        # Repay date overpassing the observation date within the next duepay date.
        ("2022-02-11"   , 23     , 100   , 2100),
        # Repay date overpassing the next duepay date.
        ("2022-03-11"   , 37    , 100   , 2000),
        # 37 - 31 = 6 > 4: Invalid overdue days.
        # Repay date within the observation date.
        ("2022-04-11"   , 4     , 100   , 1900),
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

    sorted_df = addup_ob_records(recs)
    df = test_addup_ob_records()
    for col in sorted_df:
        assert np.all(sorted_df[col] == df[col])


# %%
def ob_records():
    recs = pd.DataFrame([
        ("OID-001"  , "2021-12-11"  , 1),
        ("OID-001"  , "2022-01-11"  , 2),
        ("OID-001"  , "2022-02-11"  , 3),
        ("OID-001"  , "2022-03-11"  , 4),
        ("OID-001"  , "2022-04-11"  , 5),
        ("OID-001"  , "2022-05-11"  , 6),
    ], columns=["order_id", "ob_date", "MOB"])
    recs_L = []
    for i in range(101):
        recs_ = recs.copy()
        recs_["ever_ovdp"] = np.random.randint(7, size=len(recs))
        recs_["stop_ovdp"] = np.random.randint(7, size=len(recs))
        recs_["order_id"] = [f"OID-{i:03}"] * len(recs)
        start_date = np.random.choice(["2021-12", "2022-01", "2022-02", "2022-03"])
        recs_["ob_date"] = (np.datetime64(start_date, "M")
                            + np.arange(len(recs)).astype("m8[M]")
                            + np.timedelta64(10, "D"))

        recs_L.append(recs_)

    return pd.concat(recs_L)


# %%
def test_roll_from_ob_records():
    recs = ob_records()
    rolls = roll_from_ob_records(recs, 1)
    rolls = roll_from_ob_records(recs, 1, 4)
    assert rolls["ever_ovdp"].shape == (8, 8)

    rolls = roll_from_ob_records(recs, "2022-03-11")
    rolls = roll_from_ob_records(recs, "2022-03-11", "2022-04-11")
    assert rolls["ever_ovdp"].shape == (8, 8)

    mrolls = roll_from_ob_records(recs, "2022-03")
    mrolls = roll_from_ob_records(recs, "2022-03", "2022-04-11")
    assert np.all(mrolls["ever_ovdp"] == rolls["ever_ovdp"])


# %%
def test_pivot_ob_records():
    rec_one = test_addup_ob_records()
    N = 5
    recs = pd.concat([rec_one,] * N,
                     keys=[f"OID-{i:03}" for i in range(1, N+1)],
                     names=["order_id", "rid"])
    recs = recs.reset_index(level=0)
    precs = pivot_ob_records(recs)

    tags = pd.Series({
        "OID-001": "2021-12-11",
        "OID-002": "2021-12-11",
        "OID-003": "2022-01-11",
        "OID-004": "2022-01-11",
        "OID-005": "2021-12-11",
    })
    tprecs = pivot_ob_records(recs, tags)
    assert np.all(precs.select_dtypes(int) == tprecs.select_dtypes(int))

    vin_m1p = tprecs.groupby("rectag").apply(lambda x: (x > 1).sum(axis=0))

    return vin_m1p
