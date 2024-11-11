#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: repayment.py
#   Author: xyy15926
#   Created: 2023-10-07 14:46:51
#   Updated: 2024-05-27 17:08:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar
from collections.abc import Callable, Iterator
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from IPython.core.debugger import set_trace

import numpy as np
import pandas as pd
from ringbear.tsarr import ovdd_from_duepay_records, month_date

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

DUM_OVDD = 181
DUM_OVDP = 7


# %%
def addup_ob_records(
    records: pd.DataFrame,
    start_date: pd.Timestamp | pd.Period = None,
    ob_date: list | np.ndarray | str | int = "nextdue_noend",
    dum_mon: pd.Timestamp | pd.Period | str | int = None,
    dum_ovdd: int = DUM_OVDD,
    dum_ovdp: int = DUM_OVDP,
) -> pd.DataFrame:
    """Add MOB and overdue status for given observation dates.

    1. This should be applied only upon one account's repayment records. Use
      `groupby(order_id).apply(addup_ob_records)` for records will multiple
      accounts.

    Params:
    ------------------------
    records: DataFrame[due_date, ovd_days, dum_amt, rem_amt]
      due_date: Datetime64 convertable sequence representing the duepayment
        dates.
      ovd_days: Day past due for each duepay dates.
      due_amt: Duepay amount.
      rem_amt: Remaining amount.
    start_date: Start date of the account.
      The first duepay dates will be used as the start date if not provided.
    ob_date: Observation dates.
      Sequence: Observation date for each duepayment point.
      str | int: Rule to generate the observations dates.
    dum_mon: Date from when the account is treated as the dummy account.
      str | Timestamp: Use the duepayment dates as the time tag.
        Dummy date will be casted into "M8[M]".
      int: Use the MOB as the time tag.
    dum_ovdd: The overdue days filled for the periods after being dummy.
    dum_ovdp: The overdue periods filled for the periods after being dummy.

    Return:
    ------------------------
    DataFrame with MOB, ever/stop overdue days/periods and duepay/remaining
      amounts.
    """
    due_date = np.asarray(records["due_date"], dtype="datetime64[D]")
    sort_order = np.argsort(due_date)
    records = records.iloc[sort_order]
    due_date = due_date[sort_order]

    ovd_days = np.asarray(records["ovd_days"])
    due_amt = np.asarray(records["due_amt"]) if "due_amt" in records else None
    rem_amt = np.asarray(records["rem_amt"]) if "rem_amt" in records else None

    # Set observation date.
    if isinstance(ob_date, (int, str)):
        ob_date = month_date(due_date, ob_date)

    # Set MOBs.
    if "MOB" not in records:
        if start_date is None:
            start_date = np.datetime64(due_date.min(), "M")
        else:
            start_date = np.datetime64(start_date, "M")
        mobs = (due_date.astype("M8[M]") - start_date).astype(np.int_)
    else:
        mobs = records["MOB"]
    # set_trace()

    # Calculate overdue index.
    (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp,
     ever_duea, stop_duea, ever_rema, stop_rema) = ovdd_from_duepay_records(
        due_date, ovd_days, ob_date, due_amt, rem_amt)

    # Fill overdue days and overdue periods for peroids after dummy date.
    if dum_mon is not None:
        if isinstance(dum_mon, int):
            ever_ovdd[mobs >= dum_mon] = dum_ovdd
            stop_ovdd[mobs >= dum_mon] = dum_ovdd
            ever_ovdp[mobs >= dum_mon] = dum_ovdp
            stop_ovdp[mobs >= dum_mon] = dum_ovdp
        else:
            dum_mon = np.datetime64(dum_mon, "M")
            ever_ovdd[due_date >= dum_mon] = dum_ovdd
            stop_ovdd[due_date >= dum_mon] = dum_ovdd
            ever_ovdp[due_date >= dum_mon] = dum_ovdp
            stop_ovdp[due_date >= dum_mon] = dum_ovdp

    df = pd.DataFrame({
        "ob_date": ob_date,
        "MOB": mobs,
        "ever_ovdd": ever_ovdd,
        "stop_ovdd": stop_ovdd,
        "ever_ovdp": ever_ovdp,
        "stop_ovdp": stop_ovdp,
        "ever_duea": ever_duea,
        "stop_duea": stop_duea,
        "ever_rema": ever_rema,
        "stop_rema": stop_rema,
    })

    return df


# %%
# TODO: annotations
def roll_from_ob_records(
    recs: pd.DataFrame,
    start: pd.Timestamp | int | str,
    end: pd.Timestamp | int | str = None,
) -> dict[str, pd.DataFrame]:
    """Calculate roll from observation records.

    1. If no `end` provided, the next MOB or `ob_date` will be used as default
      to calculate the roll or migration precisely.

    Params:
    ---------------------------
    recs: DataFrame[ob_date or MOB, ever_ovdp, stop_ovdp]
    start: The time tag for the start of the performance.
      str | Timestamp: Use the observation dates as the time tag.
        Both the observation dates and the start tag will be casted into
          "M8[M]" to the determining the records.
      int: Use the MOB as the time tag.
    end: The time tag for the start of the performance.
      The next MOB or `ob_date` will be used as default.

    Return:
    ---------------------------
    Dict[ovdp-type, roll-count]
    """
    # Select begin and end records for roll rate.
    if isinstance(start, int):
        if end is None:
            end = start + 1
        start_recs = recs[recs["MOB"] == start]
        end_recs = recs[recs["MOB"] == end]
    else:
        start = np.datetime64(start, "M")
        ob_mon = np.asarray(recs["ob_date"], "M8[M]")
        if end is None:
            end = start + np.timedelta64(1, "M")
        else:
            end = np.datetime64(end, "M")
        start_recs = recs[ob_mon == start]
        end_recs = recs[ob_mon == end]

        assert(len(start_recs) == len(end_recs))

    cats = pd.CategoricalDtype([f"M{i}" for i in range(8)])
    rolls = {}
    for ovdp in ["ever_ovdp", "stop_ovdp"]:
        if ovdp not in start_recs.columns:
            continue
        # Convert dtype to categorical so to keep null rows and columns.
        start_ovdp = start_recs[ovdp].apply(lambda x: f"M{x}").astype(cats)
        end_ovdp = end_recs[ovdp].apply(lambda x: f"M{x}").astype(cats)
        rolls[ovdp] = pd.crosstab(start_ovdp.values,
                                  end_ovdp.values,
                                  dropna=False)

    return rolls


# %%
def pivot_ob_records(
    recs: pd.DataFrame,
    rectag: pd.Series = None,
    ovdtag: str = "ever_ovdp",
) -> pd.DataFrame:
    """Pivot the observation records to align the MOB.

    Params:
    ---------------------------
    recs: DataFrame[order_id, due_date, MOB, <OVDTAG>]
    rectag: Series[order_id, tag]
      Mapping `order_id` to some group tags for following caluclations.
      Vintag, the month of the `due_date`, will be used as default if not
        provided.
    ovdtag: Field name in `recs`, `ever_ovdp` for example.
      Indicating the values filled in the result.

    Return:
    ---------------------------
    DataFrame with Index[order_id] and Column[MOB].
    """
    # Construct vintage tags with the first duepayment dates if record tag
    # not provided.
    if rectag is None:
        rectag = recs.groupby("order_id").agg(rectag=("due_date", "min"))
    rectag.name = "rectag"

    # Pivot to get the Dataframe with order and MOB as index and columns.
    mob_recs = pd.pivot(recs, index="order_id", columns="MOB", values=ovdtag)
    mob_recs = mob_recs.join(rectag, on="order_id")

    return mob_recs
