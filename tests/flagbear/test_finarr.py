#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finarr.py
#   Author: xyy15926
#   Created: 2024-04-11 09:11:58
#   Updated: 2024-05-08 22:09:27
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from flagbear import finarr
    reload(finarr)

from flagbear.finarr import pivot_tags
from flagbear.finarr import ovdd_from_duepay_records, month_date


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
def test_month_date():
    recs = pd.DataFrame([
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
    due_date = np.asarray(recs["due_date"], dtype="datetime64[D]")

    next_due = month_date(due_date, "nextdue")
    assert np.all(next_due[:-1] == due_date[1:])
    assert next_due[-1] - due_date[-1] == np.timedelta64(30, "D")

    next_due_noend = month_date(due_date, "nextdue_noend")
    assert np.all(next_due_noend[:-1] == due_date[1:])
    assert next_due_noend[-1] - due_date[-1] > np.timedelta64(30, "D")

    month_end = month_date(due_date, "monthend")
    assert np.all(month_end - month_end.astype("M8[M]").astype("M8[D]")
                  >= np.timedelta64(27, "D"))

    fixed_date = month_date(due_date, 28)
    assert np.all(fixed_date - fixed_date.astype("M8[M]").astype("M8[D]")
                  == np.timedelta64(27, "D"))
    assert np.all(fixed_date > due_date)

    fixed_date = month_date(due_date, 1, False)
    assert np.all(fixed_date - fixed_date.astype("M8[M]")
                  == np.timedelta64(0, "D"))

    fixed_date = month_date(due_date, 1, True)
    assert np.all(fixed_date > due_date)
    fixed_date_ = fixed_date - np.timedelta64(30, "D")
    assert np.all(fixed_date_ - fixed_date_.astype("M8[M]")
                  == np.timedelta64(0, "D"))


# %%
def test_ovdd_from_duepay_records():
    recs = pd.DataFrame([
        ("2021-12-11"   , None  , 0     , 2200),
        ("2022-01-11"   , 0     , 0     , 2200),
        # Repay date overpassing the observation date within the next duepay date.
        ("2022-02-11"   , 23    , 100   , 2100),
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
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    def check(ob_date):
        (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp,
         ever_duea, stop_duea, ever_rema, stop_rema) = ovdd_from_duepay_records(
            due_date, ovd_days, ob_date, due_amt, rem_amt)
        assert np.all(ever_ovdd >= stop_ovdd)
        assert ever_ovdd.max() == ovd_days.max()
        # The calculation `(x+30)//31` is used to estimate periods only
        # works when periods is month.
        assert np.all(((ever_ovdd + 30) // 31 == ever_ovdp)
                      [(ob_date - due_date) <= np.timedelta64(31, "D")])
        assert np.all(((stop_ovdd + 30) // 31 == stop_ovdp)
                      [(ob_date - due_date) <= np.timedelta64(31, "D")])
        assert np.all((ever_ovdp * 100) == ever_duea)
        assert np.all((stop_ovdp * 100) == stop_duea)

        repay_date = due_date + ovd_days
        oidx = np.concatenate([[False], (repay_date > ob_date)[:-1]
                              & (repay_date[:-1] <= due_date[1:])])
        assert np.all(((ever_rema - ever_duea) == rem_amt)[~oidx])
        assert np.all(((ever_rema - ever_duea) == rem_amt + 100)[oidx])
        assert np.all((stop_rema - stop_duea) == rem_amt)

        recs_b = recs.copy()
        recs_b["repay_date"] = due_date + ovd_days
        # recs_b["ob_date"] = ob_date
        recs_b["ever_ovdd"] = ever_ovdd
        recs_b["ever_ovdp"] = ever_ovdp
        recs_b["ever_duea"] = ever_duea
        recs_b["ever_rema"] = ever_rema
        # recs_b["rem-due"] = ever_rema - ever_duea

        return (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp,
                ever_duea, stop_duea, ever_rema, stop_rema)

    ob_date = month_date(due_date, "nextdue_noend")
    check(ob_date)

    # assert check(ob_date) == check(None)
    ob_date = month_date(due_date, "monthend")
    check(ob_date)

    # ob-dates must be strictly later than due-dates.
    with pytest.raises(AssertionError):
        ob_date = month_date(due_date, 11)
        check(ob_date)

    ob_date = np.array(["2099-12"] * len(recs), dtype="datetime64[M]")
    (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp,
     ever_duea, stop_duea, ever_rema, stop_rema) = ovdd_from_duepay_records(
        due_date, ovd_days, ob_date, due_amt, rem_amt)
    check(ob_date)
    assert np.all(ever_ovdd == np.asarray(ovd_days, "timedelta64[D]"))
    assert np.all(stop_ovdd == np.timedelta64(0, "D"))
    assert np.all((ever_ovdd >= stop_ovdd)[stop_ovdd > np.timedelta64(0, "D")])
