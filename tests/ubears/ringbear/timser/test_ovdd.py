#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finarr.py
#   Author: xyy15926
#   Created: 2024-04-11 09:11:58
#   Updated: 2025-01-18 18:55:50
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear.timser import ovdd
    reload(ovdd)

from ubears.ringbear.timser.ovdd import month_date, ovdd_from_duepay_records, snap_ovd


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
    recs["due_date"] = recs["due_date"].astype("M8[D]")

    return recs


# %%
def test_month_date():
    recs = ovd_recs()
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
    assert np.all(fixed_date - fixed_date.astype("M8[M]")
                  == np.timedelta64(0, "D"))

    nm_fixed_date = month_date(due_date, 101)
    assert np.all(nm_fixed_date == fixed_date)

    fixed_date = month_date(due_date, 28, True)
    nm_fixed_date = month_date(due_date, 128, True)
    assert np.all(nm_fixed_date - fixed_date.astype("M8[M]")
                  > np.timedelta64(27, "D"))


# %%
def test_snap_ovd_precisely():
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    ob_date = month_date(due_date, "monthend")
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    monthend_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [20 , 1 , 20    , 1 , 2300  , 100   , 100   , 2300  , 100   , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2100  , 0     , 0],
        [20 , 1 , 20    , 1 , 2100  , 100   , 100   , 2100  , 100   , 100],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 0],
        [20 , 1 , 20    , 1 , 1900  , 100   , 100   , 1900  , 100   , 100],
        [35 , 2 , 19    , 1 , 1900  , 200   , 200   , 1800  , 100   , 100],
        [50 , 2 , 50    , 2 , 1800  , 200   , 200   , 1800  , 200   , 200],
        [75 , 3 , 51    , 2 , 1800  , 300   , 300   , 1700  , 200   , 200],
        [81 , 3 , 81    , 3 , 1700  , 300   , 300   , 1700  , 300   , 300],
        [98 , 4 , 20    , 1 , 1700  , 400   , 400   , 1400  , 100   , 100],
        [0  , 0 , 0     , 0 , 1300  , 0     , 100   , 1200  , 0     , 0],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 0],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 0],
        [17 , 1 , 17    , 1 , 1000  , 100   , 100   , 1000  , 100   , 100],
        [28 , 1 , 20    , 1 , 1000  , 100   , 200   , 900   , 100   , 100],
        [31 , 1 , 0     , 0 , 900   , 100   , 200   , 700   , 0     , 0],
        [20 , 1 , 20    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
    ])
    movdt, movda = monthend_ret[:, :4], monthend_ret[:, 4:]
    assert np.all(ovdt == movdt)
    assert np.all(ovda == movda)

    ob_date = month_date(due_date, "nextdue")
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    nextdue_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 100   , 2300  , 0     , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2200  , 0     , 100],
        [0  , 0 , 0     , 0 , 2100  , 0     , 100   , 2100  , 0     , 100],
        [31 , 1 , 31    , 1 , 2100  , 100   , 200   , 2100  , 100   , 200],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 100],
        [31 , 1 , 31    , 1 , 1900  , 100   , 200   , 1900  , 100   , 200],
        [35 , 2 , 30    , 1 , 1900  , 200   , 200   , 1800  , 100   , 200],
        [61 , 2 , 61    , 2 , 1800  , 200   , 300   , 1800  , 200   , 300],
        [75 , 3 , 62    , 2 , 1800  , 300   , 300   , 1700  , 200   , 300],
        [92 , 3 , 92    , 3 , 1700  , 300   , 400   , 1700  , 300   , 400],
        [98 , 4 , 0     , 0 , 1700  , 400   , 400   , 1300  , 0     , 100],
        [0  , 0 , 0     , 0 , 1200  , 0     , 100   , 1200  , 0     , 100],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 100],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 100],
        [28 , 1 , 28    , 1 , 1000  , 100   , 200   , 1000  , 100   , 200],
        [31 , 1 , 31    , 1 , 900   , 100   , 200   , 900   , 100   , 200],
        [0  , 0 , 0     , 0 , 700   , 0     , 100   , 700   , 0     , 100],
        [30 , 1 , 30    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
    ])
    novdt, novda = nextdue_ret[:, :4], nextdue_ret[:, 4:]
    assert np.all(ovdt == novdt)
    assert np.all(ovda == novda)


# %%
def test_snap_ovd_precisely_noovdend():
    recs = ovd_recs().iloc[:-1]
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    ob_date = month_date(due_date, "monthend")
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    monthend_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [20 , 1 , 20    , 1 , 2300  , 100   , 100   , 2300  , 100   , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2100  , 0     , 0],
        [20 , 1 , 20    , 1 , 2100  , 100   , 100   , 2100  , 100   , 100],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 0],
        [20 , 1 , 20    , 1 , 1900  , 100   , 100   , 1900  , 100   , 100],
        [35 , 2 , 19    , 1 , 1900  , 200   , 200   , 1800  , 100   , 100],
        [50 , 2 , 50    , 2 , 1800  , 200   , 200   , 1800  , 200   , 200],
        [75 , 3 , 51    , 2 , 1800  , 300   , 300   , 1700  , 200   , 200],
        [81 , 3 , 81    , 3 , 1700  , 300   , 300   , 1700  , 300   , 300],
        [98 , 4 , 20    , 1 , 1700  , 400   , 400   , 1400  , 100   , 100],
        [0  , 0 , 0     , 0 , 1300  , 0     , 100   , 1200  , 0     , 0],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 0],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 0],
        [17 , 1 , 17    , 1 , 1000  , 100   , 100   , 1000  , 100   , 100],
        [28 , 1 , 20    , 1 , 1000  , 100   , 200   , 900   , 100   , 100],
        [31 , 1 , 0     , 0 , 900   , 100   , 200   , 700   , 0     , 0],
    ])
    movdt, movda = monthend_ret[:, :4], monthend_ret[:, 4:]
    assert np.all(ovdt == movdt)
    assert np.all(ovda == movda)

    ob_date = month_date(due_date, "nextdue")
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    nextdue_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 100   , 2300  , 0     , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2200  , 0     , 100],
        [0  , 0 , 0     , 0 , 2100  , 0     , 100   , 2100  , 0     , 100],
        [31 , 1 , 31    , 1 , 2100  , 100   , 200   , 2100  , 100   , 200],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 100],
        [31 , 1 , 31    , 1 , 1900  , 100   , 200   , 1900  , 100   , 200],
        [35 , 2 , 30    , 1 , 1900  , 200   , 200   , 1800  , 100   , 200],
        [61 , 2 , 61    , 2 , 1800  , 200   , 300   , 1800  , 200   , 300],
        [75 , 3 , 62    , 2 , 1800  , 300   , 300   , 1700  , 200   , 300],
        [92 , 3 , 92    , 3 , 1700  , 300   , 400   , 1700  , 300   , 400],
        [98 , 4 , 0     , 0 , 1700  , 400   , 400   , 1300  , 0     , 100],
        [0  , 0 , 0     , 0 , 1200  , 0     , 100   , 1200  , 0     , 100],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 100],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 100],
        [28 , 1 , 28    , 1 , 1000  , 100   , 200   , 1000  , 100   , 200],
        [31 , 1 , 31    , 1 , 900   , 100   , 200   , 900   , 100   , 200],
        [0  , 0 , 0     , 0 , 700   , 0     , 0     , 700   , 0     , 0],
    ])
    novdt, novda = nextdue_ret[:, :4], nextdue_ret[:, 4:]
    assert np.all(ovdt == novdt)
    assert np.all(ovda == novda)


# %%
def test_snap_ovd_precisely_additional_obdate():
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    ob_date = month_date(due_date, "monthend")
    ob_date = np.asarray([ob_date[0] - 30, *ob_date,
                          ob_date[-1] + 30, ob_date[-1] + 60])
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    monthend_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [20 , 1 , 20    , 1 , 2300  , 100   , 100   , 2300  , 100   , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2100  , 0     , 0],
        [20 , 1 , 20    , 1 , 2100  , 100   , 100   , 2100  , 100   , 100],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 0],
        [20 , 1 , 20    , 1 , 1900  , 100   , 100   , 1900  , 100   , 100],
        [35 , 2 , 19    , 1 , 1900  , 200   , 200   , 1800  , 100   , 100],
        [50 , 2 , 50    , 2 , 1800  , 200   , 200   , 1800  , 200   , 200],
        [75 , 3 , 51    , 2 , 1800  , 300   , 300   , 1700  , 200   , 200],
        [81 , 3 , 81    , 3 , 1700  , 300   , 300   , 1700  , 300   , 300],
        [98 , 4 , 20    , 1 , 1700  , 400   , 400   , 1400  , 100   , 100],
        [0  , 0 , 0     , 0 , 1300  , 0     , 100   , 1200  , 0     , 0],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 0],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 0],
        [17 , 1 , 17    , 1 , 1000  , 100   , 100   , 1000  , 100   , 100],
        [28 , 1 , 20    , 1 , 1000  , 100   , 200   , 900   , 100   , 100],
        [31 , 1 , 0     , 0 , 900   , 100   , 200   , 700   , 0     , 0],
        [20 , 1 , 20    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
        [34 , 1 , 0     , 0 , 700   , 100   , 100   , 600   , 0     , 0],
        [0  , 0 , 0     , 0 , 600   , 0     , 0     , 600   , 0     , 0],
    ])
    movdt, movda = monthend_ret[:, :4], monthend_ret[:, 4:]
    assert np.all(ovdt == movdt)
    assert np.all(ovda == movda)

    ob_date = month_date(due_date, "nextdue")
    ob_date = np.asarray([ob_date[0] - 30, *ob_date,
                          ob_date[-1] + 30, ob_date[-1] + 60])
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    nextdue_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [0  , 0 , 0     , 0 , 2300  , 0     , 100   , 2300  , 0     , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2200  , 0     , 100],
        [0  , 0 , 0     , 0 , 2100  , 0     , 100   , 2100  , 0     , 100],
        [31 , 1 , 31    , 1 , 2100  , 100   , 200   , 2100  , 100   , 200],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 100],
        [31 , 1 , 31    , 1 , 1900  , 100   , 200   , 1900  , 100   , 200],
        [35 , 2 , 30    , 1 , 1900  , 200   , 200   , 1800  , 100   , 200],
        [61 , 2 , 61    , 2 , 1800  , 200   , 300   , 1800  , 200   , 300],
        [75 , 3 , 62    , 2 , 1800  , 300   , 300   , 1700  , 200   , 300],
        [92 , 3 , 92    , 3 , 1700  , 300   , 400   , 1700  , 300   , 400],
        [98 , 4 , 0     , 0 , 1700  , 400   , 400   , 1300  , 0     , 100],
        [0  , 0 , 0     , 0 , 1200  , 0     , 100   , 1200  , 0     , 100],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 100],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 100],
        [28 , 1 , 28    , 1 , 1000  , 100   , 200   , 1000  , 100   , 200],
        [31 , 1 , 31    , 1 , 900   , 100   , 200   , 900   , 100   , 200],
        [0  , 0 , 0     , 0 , 700   , 0     , 100   , 700   , 0     , 100],
        [30 , 1 , 30    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
        [34 , 1 , 0     , 0 , 700   , 100   , 100   , 600   , 0     , 0],
        [0  , 0 , 0     , 0 , 600   , 0     , 0     , 600   , 0     , 0],
    ])
    novdt, novda = nextdue_ret[:, :4], nextdue_ret[:, 4:]
    assert np.all(ovdt == novdt)
    assert np.all(ovda == novda)


# %%
def test_snap_ovd_precisely_repeated_obdate():
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    ob_date = month_date(due_date, "monthend")
    ob_date[5] = ob_date[4]
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    monthend_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 0     , 2300  , 0     , 0],
        [20 , 1 , 20    , 1 , 2300  , 100   , 100   , 2300  , 100   , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2100  , 0     , 0],
        [20 , 1 , 20    , 1 , 2100  , 100   , 100   , 2100  , 100   , 100],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 0],
        [0  , 0 , 0     , 0 , 1900  , 0     , 0     , 1900  , 0     , 0],
        [35 , 2 , 19    , 1 , 1900  , 200   , 200   , 1800  , 100   , 100],
        [50 , 2 , 50    , 2 , 1800  , 200   , 200   , 1800  , 200   , 200],
        [75 , 3 , 51    , 2 , 1800  , 300   , 300   , 1700  , 200   , 200],
        [81 , 3 , 81    , 3 , 1700  , 300   , 300   , 1700  , 300   , 300],
        [98 , 4 , 20    , 1 , 1700  , 400   , 400   , 1400  , 100   , 100],
        [0  , 0 , 0     , 0 , 1300  , 0     , 100   , 1200  , 0     , 0],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 0],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 0],
        [17 , 1 , 17    , 1 , 1000  , 100   , 100   , 1000  , 100   , 100],
        [28 , 1 , 20    , 1 , 1000  , 100   , 200   , 900   , 100   , 100],
        [31 , 1 , 0     , 0 , 900   , 100   , 200   , 700   , 0     , 0],
        [20 , 1 , 20    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
    ])
    movdt, movda = monthend_ret[:, :4], monthend_ret[:, 4:]
    assert np.all(ovdt == movdt)
    assert np.all(ovda == movda)

    ob_date = month_date(due_date, "nextdue")
    # Move one day forward to check the stop.
    ob_date[-2] = ob_date[-3] + 1
    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date, due_amt, rem_amt)
    nextdue_ret = np.array([
        [0  , 0 , 0     , 0 , 2300  , 0     , 100   , 2300  , 0     , 100],
        [23 , 1 , 0     , 0 , 2300  , 100   , 100   , 2200  , 0     , 100],
        [0  , 0 , 0     , 0 , 2100  , 0     , 100   , 2100  , 0     , 100],
        [31 , 1 , 31    , 1 , 2100  , 100   , 200   , 2100  , 100   , 200],
        [37 , 2 , 0     , 0 , 2100  , 200   , 200   , 1900  , 0     , 100],
        [31 , 1 , 31    , 1 , 1900  , 100   , 200   , 1900  , 100   , 200],
        [35 , 2 , 30    , 1 , 1900  , 200   , 200   , 1800  , 100   , 200],
        [61 , 2 , 61    , 2 , 1800  , 200   , 300   , 1800  , 200   , 300],
        [75 , 3 , 62    , 2 , 1800  , 300   , 300   , 1700  , 200   , 300],
        [92 , 3 , 92    , 3 , 1700  , 300   , 400   , 1700  , 300   , 400],
        [98 , 4 , 0     , 0 , 1700  , 400   , 400   , 1300  , 0     , 100],
        [0  , 0 , 0     , 0 , 1200  , 0     , 100   , 1200  , 0     , 100],
        [17 , 1 , 0     , 0 , 1200  , 100   , 100   , 1100  , 0     , 100],
        [17 , 1 , 0     , 0 , 1100  , 100   , 100   , 1000  , 0     , 100],
        [28 , 1 , 28    , 1 , 1000  , 100   , 200   , 1000  , 100   , 200],
        [31 , 1 , 31    , 1 , 900   , 100   , 200   , 900   , 100   , 200],
        [0  , 0 , 0     , 0 , 700   , 0     , 0     , 700   , 0     , 0],
        [30 , 1 , 30    , 1 , 700   , 100   , 100   , 700   , 100   , 100],
    ])
    novdt, novda = nextdue_ret[:, :4], nextdue_ret[:, 4:]
    assert np.all(ovdt == novdt)
    assert np.all(ovda == novda)


# %%
def check_snap_ovd(ob_date):
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    ovdt, ovda = snap_ovd(due_date, rep_date, None, ob_date,
                          due_amt, rem_amt)
    ever_ovdd, ever_ovdp, stop_ovdd, stop_ovdp = ovdt.T
    ever_rema, ever_ovda, ever_duea, stop_rema, stop_ovda, stop_duea = ovda.T

    assert np.all(ever_ovdd >= stop_ovdd)
    assert ever_ovdd.max() == ovd_days.max()
    # The calculation `(x+30)//31` is used to estimate periods only
    # works when periods is month.
    assert np.all(((ever_ovdd + 30) // 31 == ever_ovdp)
                  [(ob_date - due_date) <= np.timedelta64(31, "D")])
    assert np.all(((stop_ovdd + 30) // 31 == stop_ovdp)
                  [(ob_date - due_date) <= np.timedelta64(31, "D")])
    assert np.all((ever_ovdp * 100) == ever_ovda)
    assert np.all((stop_ovdp * 100) == stop_ovda)
    assert np.all(ever_duea >= ever_ovda)

    return (ever_ovdd, ever_ovdp, stop_ovdd, stop_ovdp,
            ever_rema, ever_ovda, ever_duea,
            stop_rema, stop_ovda, stop_duea)


def check_ovdd_from_duepay_records(ob_date):
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")
    ovd_days = np.asarray(recs["ovd_days"], dtype="timedelta64[D]")
    rep_date = due_date + ovd_days
    due_amt = recs["due_amt"]
    rem_amt = recs["rem_amt"]

    (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp, ever_ovda, stop_ovda,
     ever_rema, stop_rema, ever_duea, stop_duea) = ovdd_from_duepay_records(
        due_date, ovd_days, ob_date, due_amt, rem_amt)
    assert np.all(ever_ovdd >= stop_ovdd)
    assert ever_ovdd.max() == ovd_days.max()
    # The calculation `(x+30)//31` is used to estimate periods only
    # works when periods is month.
    assert np.all(((ever_ovdd + 30) // 31 == ever_ovdp)
                  [(ob_date - due_date) <= np.timedelta64(31, "D")])
    assert np.all(((stop_ovdd + 30) // 31 == stop_ovdp)
                  [(ob_date - due_date) <= np.timedelta64(31, "D")])
    assert np.all((ever_ovdp * 100) == ever_ovda)
    assert np.all((stop_ovdp * 100) == stop_ovda)
    assert np.all(ever_duea >= ever_ovda)

    return (ever_ovdd, ever_ovdp, stop_ovdd, stop_ovdp,
            ever_rema, ever_ovda, ever_duea,
            stop_rema, stop_ovda, stop_duea)


# %%
def test_snap_ovdd_compared():
    recs = ovd_recs()
    due_date = np.asarray(pd.to_datetime(recs["due_date"]), dtype="datetime64[D]")

    # Nextdue
    ob_date = month_date(due_date, "nextdue")
    ret1 = check_snap_ovd(ob_date)
    ret2 = check_ovdd_from_duepay_records(ob_date)
    for a, b in zip(ret1[:4], ret2[:4]):
        assert np.all(a == b)

    # Monthend
    ob_date = month_date(due_date, "monthend")
    ret1 = check_snap_ovd(ob_date)
    ret2 = check_ovdd_from_duepay_records(ob_date)
    for a, b in zip(ret1[:2], ret2[:2]):
        assert np.all(a == b)
    # `ovdd_from_duepay_records` is wrong in MOB-10.
    for a, b in zip(ret1[2:4], ret2[2:4]):
        assert np.sum(a == b) == len(recs) - 1

    # Next 11-th.
    ob_date = month_date(due_date, 111, False)
    ret1 = check_snap_ovd(ob_date)
    ret2 = check_ovdd_from_duepay_records(ob_date)
    for a, b in zip(ret1[:4], ret2[:4]):
        assert np.all(a == b)
