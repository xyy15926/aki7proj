#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pdler.py
#   Author: xyy15926
#   Created: 2023-07-24 16:03:48
#   Updated: 2023-07-24 16:13:34
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd
from ringbear.pdler import (
    autotype_ser,
    parse_sort_keys,
    drop_records,
    drop_fields,
    fill_numeric_ser,
)


# %%
# @pytest.mark.skip(reason="Uncharted function")
def test_autotype() -> None:
    test_data = [1, 2, 2, 4, 4, 6, 7, 9, 10] * 4
    test_data.extend(["3", "ab", float("nan")])
    ser = pd.Series(test_data)
    ser_numeric = pd.to_numeric(ser, errors="coerce")
    ser_converted_1 = autotype_ser(ser, autotype_thresh=0.6)
    assert (ser_converted_1 == ser_numeric).sum() == 37
    ser_converted_2 = autotype_ser(ser.astype("string"), autotype_thresh=0.6)
    assert (ser_converted_2 == ser_numeric).sum() == 37
    assert (ser_converted_2 == ser_converted_1).sum() == 37
    test_data[2] = 3.5
    assert(autotype_ser(test_data).iloc[-3] == 3)
    assert(np.isnan(autotype_ser(test_data).iloc[-2]))
    test_data[-2] = "ab99999999"
    assert(autotype_ser(test_data).iloc[-2] == 99999999)
    test_data = pd.Series([1, 2, "3"] * 5)
    test_data[2] = None
    assert(fill_numeric_ser(test_data)[2] == -999999)


# %%
def test_df_sort_key() -> None:
    sort_keys = ">af"
    assert(parse_sort_keys(sort_keys) == ([">af"], [True]))
    sort_keys = ["af", "<"]
    assert(parse_sort_keys(sort_keys) == (["af"], [True]))
    sort_keys = [["af", "<"]]
    assert(parse_sort_keys(sort_keys) == (["af"], [True]))
    sort_keys = {"af": ">"}
    assert(parse_sort_keys(sort_keys) == (["af"], [False]))


def test_df_drop() -> None:
    test_data = pd.DataFrame([["a", "a", "b", "b"],
                              [1, 2, 3, 4],
                              [1, 2, 3, 4],
                              [9, 8, 7, 6]]).T
    assert((drop_records(
        test_data, unique_keys=0,
        keep="last")[1] == [2, 4]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep="first")[1] == [1, 3]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep={1: "first", 2: "last"})[2] == [2, 4]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep={1: "first", 2: "last"})[1] == [1, 3]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep={1: "first", 2: "last"},
        sort_by=(3, 1))[1] == [4, 2]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep={1: "first", 2: "last"},
        sort_by=(3, 1))[2] == [3, 1]).sum() == 2)
    assert((drop_records(
        test_data, unique_keys=0,
        keep={1: "concat"})[1] == ["1:2", "3:4"]).sum() == 2)
