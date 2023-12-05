#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dtyper.py
#   Author: xyy15926
#   Created: 2023-07-24 14:17:05
#   Updated: 2023-10-11 22:09:13
#   Description:
# ---------------------------------------------------------

# %%
import re

import numpy as np
import pandas as pd
import pytest
from ringbear.dtyper import (STR_CASTER, TYPE_REGEX, concat_interval,
                             detect_dtype_anyway, detect_str_dtype,
                             infer_dtype, infer_major_dtype,
                             intervals_from_list, max_key, min_key,
                             regex_caster, tidy_up)


# %%
def test_type_regex() -> None:
    assert re.match(TYPE_REGEX["integer"], "2342") is not None
    assert re.match(TYPE_REGEX["integer"], "2,342") is not None
    assert re.match(TYPE_REGEX["integer"], "23,42") is None
    assert re.match(TYPE_REGEX["integer"], "2342.23") is None
    assert re.match(TYPE_REGEX["integer"], "2,342.23") is None
    assert re.match(TYPE_REGEX["integer"], "2022-11-11 12:23") is None
    assert re.match(TYPE_REGEX["floating"], "2342.23") is not None
    assert re.match(TYPE_REGEX["floating"], "2342") is not None
    assert re.match(TYPE_REGEX["datetime"], "2022-02-28") is not None
    assert re.match(TYPE_REGEX["datetime"], "2022/02/28") is not None
    assert re.match(TYPE_REGEX["datetime"], "2022-02-29") is None
    assert re.match(TYPE_REGEX["datetime"], "2022-02-28 12:12:12").span()[1] == 19
    assert re.match(TYPE_REGEX["datetime"], "2022-02-28 12:12") is not None
    assert re.match(TYPE_REGEX["datetime"], "2022-02-28 12: 12:12").span()[1] == 10
    assert re.match(TYPE_REGEX["interval"], "(12, 13]") is not None
    assert re.match(TYPE_REGEX["interval"], "(12,13]") is not None
    assert re.match(TYPE_REGEX["interval"], "[12,13]") is not None
    assert re.match(TYPE_REGEX["interval"], "[12,13)") is not None
    assert re.match(TYPE_REGEX["fset"], "{12,13}") is not None
    assert re.match(TYPE_REGEX["fset"], "{12,  13, 14}") is not None
    assert re.match(TYPE_REGEX["fset"], "{ 12,  13, 14}") is not None
    assert re.match(TYPE_REGEX["na"], "nAn") is not None


# %%
def test_str_caster() -> None:
    assert STR_CASTER["integer"]("2342") == 2342
    assert STR_CASTER["integer"]("2,342") == 2342
    assert STR_CASTER["floating"]("2342.23") == 2342.23
    assert STR_CASTER["floating"]("2,342.23") == 2342.23
    assert STR_CASTER["datetime"]("2022-02-28") == pd.Timestamp("2022/02/28 00:00:00")
    assert 13 in STR_CASTER["interval"]("(12, 13]")
    assert 12 not in STR_CASTER["interval"]("(12, 13]")
    assert 12 in STR_CASTER["interval"]("[12, 13]")
    assert 12 in STR_CASTER["fset"]("{12, 13,14}")
    assert pd.isna(STR_CASTER["na"]("nan"))


# %%
def test_sort_key() -> None:
    a = [1, 2, 3, pd.Interval(1, 2), frozenset([2, 3, 4])]
    assert sorted(a, key=min_key) == [1, pd.Interval(1, 2), 2, frozenset([2, 3, 4]), 3]
    assert sorted(a, key=max_key) == [1, 2, pd.Interval(1, 2), 3, frozenset([2, 3, 4])]


# %%
def test_interval_operations() -> None:
    assert intervals_from_list([1, 2, 5, 6]) == [
        pd.Interval(1, 2, "both"),
        pd.Interval(2, 5, "right"),
        pd.Interval(5, 6, "right"),
    ]
    assert intervals_from_list([float("-inf"), 2, 5, float("inf")]) == [
        pd.Interval(float("-inf"), 2, "both"),
        pd.Interval(2, 5, "right"),
        pd.Interval(5, float("inf"), "right"),
    ]
    intervals = [
        pd.Interval(2, 3),
        pd.Interval(3, 4),
        pd.Interval(4, 5),
        pd.Interval(2.4, 10),
    ]
    assert concat_interval(intervals[0], intervals[1]) == pd.Interval(2, 4)
    assert concat_interval(intervals[0], intervals[2]) is None
    assert concat_interval(intervals[0], intervals[3]) == pd.Interval(2, 10)
    assert tidy_up(intervals) == [pd.Interval(2, 10)]
    intervals.extend([frozenset([1, 2, 3, 4, 5]), 100, 45, 2])
    assert tidy_up(intervals) == [1, pd.Interval(2, 10, closed="both"), 45, 100]


# %%
def test_infer_dtype_scalar() -> None:
    assert infer_dtype(pd.Interval(3, 4)) == "interval"
    assert infer_dtype(3) == "integer"
    assert infer_dtype(3, True) == "floating"
    assert infer_dtype(3.3) == "floating"
    assert infer_dtype(str) == "string"
    assert infer_dtype(np.arange(10))[0] == "integer"
    assert infer_dtype(pd.RangeIndex(10))[0] == "integer"


# %%
def test_detect_str_dtype() -> None:
    assert detect_str_dtype("2323") == "integer"
    assert detect_str_dtype("2323.23") == "floating"
    assert detect_str_dtype("addddd2323.23", match_ratio=0.1) == "floating"
    assert detect_str_dtype("addddd2323.23", match_ratio=0.6) == "unknown"
    assert detect_str_dtype("a2323.23", match_ratio=0.8) == "floating"
    assert detect_str_dtype("a2323.23", how="match") == "unknown"
    assert detect_str_dtype("a2323.23", how="fullmatch") == "unknown"
    assert detect_str_dtype("2323.23", how="fullmatch") == "floating"
    assert detect_str_dtype("(1,2)", match_ratio=0.6) == "interval"
    assert detect_str_dtype("2022-10-20", match_ratio=0.6) == "datetime"
    assert detect_dtype_anyway("addddd2323.23", match_ratio=0.6) == "unknown"


# %%
def test_regex_caster() -> None:
    assert regex_caster("23,223d", match_ratio=0.8) == 23223
    assert regex_caster("232,23d", match_ratio=0.8) == "232,23d"
    assert regex_caster(222) == 222
    assert regex_caster("232,23d", target="integer", match_ratio=0.8) is None
    assert regex_caster("23,223d", target="integer", match_ratio=0.8) == 23223
    assert regex_caster("23,223d", target="integer", match_ratio=1) is None
    assert regex_caster("232,23d", target="floating", match_ratio=0.8) is None
    assert regex_caster("2022-10-20", target="integer") is None
    assert regex_caster("2022-10-20", target="datetime") == pd.Timestamp("2022-10-20")


# %%
def test_infer_dtype_series() -> None:
    test_data = [1, 2, 2, 4, 4, 6, 7, 9, 10] * 4
    test_data.extend(["3", "ab", float("nan")])
    ser = pd.Series(test_data)
    assert infer_dtype(ser)[0] == "mixed-integer"
    # infer major dtype in series.
    assert infer_major_dtype(test_data)[0] == "integer"
    test_data[2] = 3.5
    assert infer_major_dtype(test_data, to_float=True)[0] == "floating-integer"
    assert infer_major_dtype(test_data)[0] == "integer"
