#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: finarr.py
#   Author: xyy15926
#   Created: 2024-03-12 11:02:29
#   Updated: 2024-04-28 15:11:37
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar
from collections import deque
from collections.abc import Callable, Iterator, Sequence
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from IPython.core.debugger import set_trace

import numpy as np
import pandas as pd

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
def pivot_tags(tags: pd.Series, sep: str = ",") -> pd.DataFrame:
    """Pivot Series with joined tags into DataFrame.

    Split values in `tags` with `seps`, and then count tag frequncies for each
    tag in each record.

    Params:
    tags: Series with values of tags seperated by `sep`.
    sep: Seperator.

    Return:
    DataFrame with tags as columns and counts of tags as value.
                tag1    tag2    ...
        idx1    1       0       ...
        idx2    0       2       ...
        ...
    """
    tags = tags.fillna("").astype(str).str.strip(sep).str.split(sep, expand=False)
    tag_counts = (
        pd.DataFrame(
            {
                "id": tags.index.repeat(tags.apply(len)),
                "tags": np.concatenate(tags.values),
                "ones": np.ones(np.add.reduce(tags.apply(len)), dtype=np.int_),
            }
        )
        .replace("", "NULL")
        .groupby(["id", "tags"])["ones"]
        .agg(sum)
        .unstack()
        .fillna(0)
        .astype(np.int_)
    )
    return tag_counts


# %%
def ovdd_from_duepay_records(
    due_date: list | np.ndarray,
    ovd_days: list | np.ndarray,
    ob_date: list | np.ndarray = None,
    due_amt: list | np.ndarray = None,
    rem_amt: list | np.ndarray = None,
) -> tuple[pd.DataFrame]:
    """Calculate overdue days from duepay records.

    1. Mostly, the repayment records will be with format like:

        | ORDER_ID | PAYMENT_N | DUEPAY_DATE | OVERDUE_DAYS |
        |----------+-----------+-------------+--------------|

      That will be easy to get the day past due for each duepayment, but not
      convenient to get the overdue days at some specific time point, A.K.A.
      observation point.
      Here, `due_date`, with other arrays, is assumed to be sorted in ascending
      order.

    2. There are 2 kind of overdue days for a period of time:

      2.1 Ever: The maximum overdue days ever occuring during the period
        of time between observation points.
      2.2 Stop: The overdue days at the observation point.

      Take the following time line for example:
        * due day: 2022-01-11
        * repayment day: 2022-01-28
        * observation: 2022-01-28
      The perception above could be calculated to:
        * ever overdue days: 17D
        * stop overdue days: 0D
        * ever overdue period: 1
        * stop overdue period: 0
      Explanation:
        At 2022-01-28, the repayment is done, so the whole overdue days is
        determined to be 17D.
        And when looking at the very end of 2022-01-28, the account isn't
        overdued any longer, and the whole overdue thing could be totally
        observed at 2022-01-28 and won't effect the following repayments.
        (Looking back from T+1 seems good.)

    3. A queue is used to maintain the periods that the responsible overdue
      days overpassed the former observation.
      The following assumptions are necessary for "ever"s:

      3.1 The observation should be strictly later than the duepay date.
        The overdue days is provided at each period during the iteration, and
        it will be used calculated the repayment date and overdue status. But
        if the observation is the duepay day, the infomation from overdue
        days shouldn't be used, which will bring the uncompatiablilty of
        overdue days and overdue periods.
      3.2 The gaps between repayments and the gaps between the observations
        should be (independently) close to equal. Or if the overdue periods
        overlap partly, the oldest period may not be the longest.
      3.3 The duepay amount and the remaining amount 

    4. Take the following timeline as a example:

        | date        | dpd | last_dpd | due_gap | ob_gap | od_gap | ever | stop |
        |-------------+-----+----------+---------+--------+--------+------+------|
        | 2022-01-11  | 0   | 0        | 31      | 0*     |        |      |      |
        | 2022-01-28* |     |          |         | 0*     | 17     | 0    | 0    |
        | 2022-02-11  | 3   | 0        | 28      | 31     | 31     | 0    | 0    |
        | 2022-02-28* |     |          |         | 31     | 17     | 3    | 0    |
        | 2022-03-11  | 4   | 0        | 31      | 28     | 28     | 3    | 0    |
        | 2022-03-28* |     |          |         | 28     | 17     | 4    | 0    |
        | 2022-04-11  | 37  | 0        | 30      | 31     | 31     | 4    | 0    |
        | 2022-04-28* |     |          |         | 31     | 17     | 17   | 17   |
        | 2022-05-11  | 4   | 37       | 31      | 30     | 30     | 30   | 30   |
        | 2022-05-28* |     |          |         | 30     | 17     | 37   | 0    |
        | 2022-06-11  | 0   | 0        | 30      | 31     | 31     | 37   | 0    |
        | 2022-06-28* |     |          |         | 31     | 17     | 0    | 0    |
        | 2022-07-11  | 35  | 0        | 31      | 30     | 30     | 0    | 0    |
        | 2022-07-28* |     |          |         | 30     | 17     | 17   | 17   |
        | 2022-08-11  | 75  | 35       | 31      | 31     | 31     | 31   | 31   |
        | 2022-08-28* |     |          |         | 31     | 17     | 35   | 17   |
        | 2022-09-11  | 46  | 75       | 30      | 31     | 31     | 35   | 31   |
        | 2022-09-28* |     |          |         | 31     | 17     | 48   | 48   |
        | 2022-10-11  | 20  | 75       | 31      | 30     | 30     | 61   | 61   |
        | 2022-10-28* |     |          |         | 30     | 17     | 75   | 17   |
        | 2022-11-11  | 0   | 0        | 30*     | 31     | 31     | 75   | 0    |
        | 2022-11-28* |     |          |         | 31     | 17     | 20   | 0    |
        | inf*        |     |          |         | inf    | inf    | 0    | 0    |

      Note:
      * last_dpd: Former overdue days that effects current period.
      * due_gap: Gap of days between two adjacent duepayment date.
      * ob_gap: Gap of days between two adjacent observation date.
      * od_gap: Gap of days between duepayment date and responsible observation
        date.
      * *: Assumption for observation date or some setting for convenience.

    Params:
    ----------------------
    due_date: Sequence of duepayment date, which should be datetime64 or
      string that could be casted into datetime64 by NumPy.
    ovd_days: Sequence of overdue days of each repayments.
    ob_date: Sequence of observation date for each duepayment point.
      If no argument passed, this will be the `due_date` after shifting
      out the first duepay date and including a faraway date.
    due_amt: Sequence of duepay amount.
    rem_amt: Sequence of remaining amount.

    Return:
    ----------------------
    ever_ovdd: NDArray of maximum of overdue days ever occured during before
      responsible point of observation.
    stop_ovdd: NAArray of overdue days at the point of observation.
    ever_ovdp: The number periods counted here will be more precise as the
      days of each of month is not the same.
    stop_ovdp: Ditto, the same below.
    ever_duea:
    stop_duea:
    ever_rema:
    stop_rema:
    """
    dueds = np.asarray(due_date, dtype="datetime64[D]")
    if ob_date is None:
        obds = np.concatenate(
            [dueds[1:], np.array(["2999-12-31"], dtype="datetime64[D]")])
    else:
        obds = np.asarray(ob_date, dtype="datetime64[D]")
    ovdds = np.asarray(ovd_days, dtype="timedelta64[D]")
    ovdds[np.isnat(ovdds)] = np.timedelta64(0, "D")
    das = [0] * len(ovd_days) if due_amt is None else due_amt
    ras = [0] * len(ovd_days) if rem_amt is None else rem_amt

    OTD = np.timedelta64(0, "D")
    rec_N = len(due_date)

    ever_ovdd = [0] * rec_N     # Number of overdue days ever occuring during the period.
    stop_ovdd = [0] * rec_N     # Number of overdue days at the end of the period.
    ever_ovdp = [0] * rec_N     # Number of overdue periods ever occuring during the period.
    stop_ovdp = [0] * rec_N     # Number of overdue periods at the end of the period.
    ovd_Q = deque()
    ever_duea = [0] * rec_N
    stop_duea = [0] * rec_N
    ever_rema = [0] * rec_N
    stop_rema = [0] * rec_N

    for idx, dued, ovdd, obd, duea, rema in zip(range(rec_N), dueds, ovdds,
                                                obds, das, ras):
        if dued > obd:
            logger.warning(f"Invalid observation date in records at {dued}.")

        # if dued == np.datetime64("2022-04-11", "D"):
        #     set_trace()
        repd = dued + ovdd
        if len(ovd_Q) == 0:
            if repd <= obd:
                ever_ovdd[idx] = ovdd
                stop_ovdd[idx] = OTD
                ever_ovdp[idx] = 1 if ovdd > 0 else 0
                stop_ovdp[idx] = 0

                ever_duea[idx] = duea if ovdd > 0 else 0
                stop_duea[idx] = 0
                ever_rema[idx] = rema + ever_duea[idx]
                stop_rema[idx] = rema
            else:
                # Append current repayment status iff repayment date overpass the
                # observation point strictly, which representes the rest of the
                # overdue days that can only be observed later.
                ovd_Q.append((dued, repd, duea, rema))

                ever_ovdd[idx] = obd - dued
                stop_ovdd[idx] = obd - dued
                ever_ovdp[idx] = 1 if ever_ovdd[idx] > 0 else 0
                stop_ovdp[idx] = 1 if stop_ovdd[idx] > 0 else 0

                ever_duea[idx] = duea if ovdd > 0 else 0
                stop_duea[idx] = 0 if ovdd == OTD else duea
                ever_rema[idx] = rema + ever_duea[idx]
                stop_rema[idx] = rema + stop_duea[idx]
        else:
            last_dued, last_repd, last_duea, last_rema = ovd_Q[0]
            if last_repd > repd:
                logger.warning(f"Invalid overdue day in records at {dued}.")

            if last_repd > obd:
                ovd_Q.append((dued, repd, duea, rema))

                ever_ovdd[idx] = obd - last_dued
                stop_ovdd[idx] = obd - last_dued
                ever_ovdp[idx] = len(ovd_Q)
                stop_ovdp[idx] = len(ovd_Q)

                ever_duea[idx] = sum([ele[2] for ele in ovd_Q])
                stop_duea[idx] = sum([ele[2] for ele in ovd_Q])
                # If `oas[idx] + rema[idx] == rema[idx-1]`, the following
                # assignment must lead to the same result:
                #   ever_rema[idx] = rema + ever_duea[idx]
                #   stop_rema[idx] = rema + stop_duea[idx]
                # Else the perception of remain amount isn't defined well.
                ever_rema[idx] = last_rema + last_duea
                stop_rema[idx] = last_rema + last_duea
            else:
                # `ever_ovdd` is assigned directly here only under the
                # assumption that both the gaps between repayments and the gaps
                # between the observations are independently equal. Or the gaps
                # between the last repayment and duepayment may not the largest
                # ever during current period.
                ever_ovdd[idx] = last_repd - last_dued
                ever_rema[idx] = last_rema + last_duea
                # `>` represent that the duepay amount won't be included at the
                # duepay day.
                if last_repd > dued:
                    ever_ovdp[idx] = len(ovd_Q) + 1
                    ever_duea[idx] = sum([ele[2] for ele in ovd_Q]) + duea
                else:
                    ever_ovdp[idx] = len(ovd_Q)
                    ever_duea[idx] = sum([ele[2] for ele in ovd_Q])

                if repd > obd:
                    ovd_Q.append((dued, repd, duea, rema))

                while len(ovd_Q) > 0:
                    last_dued, last_repd, last_duea, last_rema = ovd_Q[0]
                    if last_repd > obd:
                        stop_ovdd[idx] = obd - last_dued
                        stop_ovdp[idx] = len(ovd_Q)
                        stop_duea[idx] = sum([ele[2] for ele in ovd_Q])
                        stop_rema[idx] = last_rema + last_duea
                        break
                    else:
                        ovd_Q.popleft()
                else:
                    stop_ovdd[idx] = OTD
                    stop_ovdp[idx] = 0
                    stop_duea[idx] = 0
                    stop_rema[idx] = rema

    return (np.asarray(ever_ovdd), np.asarray(stop_ovdd),
            np.asarray(ever_ovdp), np.asarray(stop_ovdp),
            np.asarray(ever_duea), np.asarray(stop_duea),
            np.asarray(ever_rema), np.asarray(stop_rema))


# %%
def month_date(
    dates: pd.Series | list | np.ndarray,
    rule: str | int = 28,
    forced: bool = True,
) -> np.ndarray:
    """Generate dates for given date sequences.

    Mostly, this is called to generate observation date according to some
    predefined rules.

    Params:
    ----------------------
    due_date: Sequence of datetime64 or values could be casted into datetime64.
    rule: Rule to set observation date.
      nextdue: Next due date as the observation date, the last observation date
        will be 30days after the last due date.
      nextdue_noend: Ditto, but the last observation date will be 2099-12-31.
      monthend: The end of month for each due date.
      int: The fixed date of month for each due date.
    forced: If to moved 30 days forward to ensure all the dates in result
      succeed the corresponding given dates.

    Return:
    ----------------------
    np.darray of Datetime64.
    """
    due_date = np.asarray(dates, dtype="M8[D]")

    if rule == "nextdue":
        stop_date = due_date.max() + np.timedelta64(30, "D")
        ob_date = np.concatenate(
            [due_date[1:], np.asarray([stop_date], dtype="M8[D]")])
    elif rule == "nextdue_noend":
        ob_date = np.concatenate(
            [due_date[1:], np.array(["2099-12-31"], dtype="M8[D]")])
    elif rule == "monthend":
        ob_date = (due_date.astype("M8[M]") + np.timedelta64(1, "M")
                   - np.timedelta64(1, "D"))
    elif isinstance(rule , int) and 1 <= rule <= 28:
        ob_date = due_date.astype("M8[M]") + np.timedelta64(rule - 1, "D")
        if np.any(ob_date < due_date):
            invalid = due_date[ob_date < due_date]
            logger.warning(
                f"Result dates {invalid} may precedes the given dates.")
            if forced:
                ob_date += np.timedelta64(30, "D")
    else:
        raise ValueError(f"Invalid observeration date setting: {rule}.")

    return ob_date
