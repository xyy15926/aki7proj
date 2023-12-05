#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dtyper.py
#   Author: xyy15926
#   Created at: 2020-09-07 18:56:18
#   Updated: 2023-10-12 19:38:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
import re
from collections import ChainMap, OrderedDict
from collections.abc import Iterable
from typing import Any, Union

import numpy as np
import pandas as pd

if __name__ == "__main__":
    pass
else:
    pass


# %%
EPSILON = 1e-6
SAMPLE_N = 1000
REGEX_MATCH_RATIO = 1.0
REGEX_HOW = "search"


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
TYPE_REGEX = OrderedDict(
    {
        # 1. `(?<![\.\d])` and `(?![\.\d])` exclude the integer in a float
        # 2. As `(?<![\-\./:])` and `(?![\-\./:])` exclude integer in datetime
        "integer": r"(?<![\.\d\-/:\[\{\(])[+-]?"
        r"(\d+|\d{1,3}(?:,\d{3})*)"
        r"(?![,\.\d\-/:\]\}\)])",
        "floating": r"(?<![\.\d\-/:\[\{\(])[+-]?"
        r"(\d+|\d{1,3}(?:,\d{3})*)"
        r"(?:\.\d*)?(?![,\.\d\-/:\]\}\)])",
        "na": r"[nN][aA][nNTt]?",
        "idcard": r"[1-9]\d{5}(?:18|19|20)\d{2}(?:(?:0[1-9])|(?:1[0-2]))"
        r"(?:(?:[0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]",
        "mobile": r"1(?:3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])"
        r"\d{8}",
        # It must be so complicated to handle days in months: 28, 29, 30, 31
        "datetime": r"(?!0000)[0-9]{4}[-/]"
        r"(?:(?:0[1-9]|1[0-2])[-/]"
        r"(?:0[1-9]|1[0-9]|2[0-8])|(?:0[13-9]|1[0-2])"
        r"-(?:29|30)|(?:0[13578]|1[02])-31)"
        r"(?: \d{1,2}:\d{1,2}(:\d{1,2})?)?",
        "interval": r"[\(\[] *[+-]?\d+(?:\.\d*)?, *[+-]?\d+(?:\.\d*)? *[\)\]]",
        # Frozenset is hashable
        # "fset": r"\{ *[+-]?\d+(?:\.\d*)?(?:, *[+-]?\d+(?:\.\d*)?)* *\}",
        "fset": r"\{ *[\w_+-\\\.\(\)\[\]]+(?: *, *[\w_+-\\\.\(\)\[\]]+)* *\}",
    }
)


TYPE_ENUMS = {
    "integer": (int, np.int8, np.int16, np.int32, np.int64),
    "floating": (float, np.float16, np.float32, np.float64),
    "datetime": (pd.Timestamp,),
    "interval": (pd.Interval,),
    "fset": (frozenset, set),
    "na": (type(None), pd._libs.missing.NAType, float,),
    "string": (str,),
}


def _str2_interval(s: str) -> pd.Interval:
    left, right = map(lambda _x: float(_x.strip()), s[1:-1].split(","))
    if s[0] == "(" and s[-1] == ")":
        closed = "neither"
    elif s[0] == "[" and s[-1] == ")":
        closed = "left"
    elif s[0] == "(" and s[-1] == "]":
        closed = "right"
    else:
        closed = "both"
    return pd.Interval(left, right, closed)


STR_CASTER = {
    "integer": lambda x: int("".join(x.split(","))),
    "floating": lambda x: float("".join(x.split(","))),
    "datetime": pd.Timestamp,
    "interval": _str2_interval,
    "fset": lambda x: frozenset(map(regex_caster, x.strip()[1:-1].split(","))),
    "na": float,
}


VALID_CASTER = {
    "integer-floating": float,
    "floating-integer": int,
    "datetime-integer": lambda x: int(pd.Timestamp(x).asm8() / 86400 + 25567),
    "integer-datetime": lambda x: pd.Timestamp(x - 25567, unit="D"),
}


TYPE_MIXIN = frozenset([*TYPE_ENUMS["interval"], *TYPE_ENUMS["fset"],])
TYPE_SINGLE = frozenset(
    [*TYPE_ENUMS["integer"], *TYPE_ENUMS["floating"], *TYPE_ENUMS["datetime"],]
)
TYPE_ORDERED = frozenset(
    [
        *TYPE_ENUMS["integer"],
        *TYPE_ENUMS["floating"],
        *TYPE_ENUMS["datetime"],
        *TYPE_ENUMS["interval"],
        *TYPE_ENUMS["fset"],
    ]
)
STR_TYPE_ORDERED = frozenset(["integer", "floating", "datetime",])


# %%
def min_key(ele: Any) -> Any:
    """
    Description:
    Minimum key function for `sort`
    1. interval will be represented by its left edge
       (doesn't consider that the left edge isn't not included)
    2. frozenset will be represented by its minimum elements

    Params:
    ele: ele in array-like to be sorted

    Return:
    element which is comparable
    """
    if type(ele) in TYPE_ENUMS["interval"]:
        return ele.left + EPSILON
    if type(ele) in TYPE_ENUMS["fset"]:
        return min(ele)
    return ele


def max_key(ele: Any) -> Any:
    """
    Description:
    Maximum key function for `sort`
    1. interval will be represented by its right edge
    2. frozenset will be represented by its maximum elements

    Params:
    ele: ele in array-like to be sorted

    Return:
    element which is comparable
    """
    if type(ele) in TYPE_ENUMS["interval"]:
        return ele.right
    if type(ele) in TYPE_ENUMS["fset"]:
        return max(ele)
    return ele


# %%
def is_overlapped(seq: list, *, sorted_: bool = False) -> bool:
    """
    Description:
    1. Check if elements in `seq` overlap

    Params:
    seq:
    sorted_: if seq is ordered
inv
    Return:
    """
    if not sorted_:
        _seq = seq.copy()
        sorted(_seq, key=min_key)
    else:
        seq_ = seq

    for ele_before, ele_later in zip(seq_[:-1], seq[1:]):
        if max_key(ele_before) >= min_key(ele_later):
            return True
    return False


# %%
def intervals_from_list(
    seq: list, closed: str = "right", edges: str = "both"
) -> list:
    """
    Description:
    Contruct intervals from list

    Params:
    seq: list of numbers determines intevals
    closed: whether each interval is closed on left, right, both or neither
    edges: whether first interval is closed on left and last interval is closed
        on right. This will override the behavior of `closed`.

    Return:
    list of intervals
    """
    intervals = [
        pd.Interval(le, re, closed) for le, re in zip(seq[:-1], seq[1:])
    ]
    if edges == "both" or edges == "left":
        intervals[0] = pd.Interval(
            intervals[0].left, intervals[0].right, "both"
        )
    if edges == "neither":
        intervals[-1] = pd.Interval(
            intervals[-1].left, intervals[-1].right, "neither"
        )
    return intervals


# %%
def concat_interval(il: pd.Interval, ir: pd.Interval) -> pd.Interval:
    """
    Description:
    Return concatenated interval if two ordered are overlapped or
    adjacent intervals, else return None.

    Params:
    il: smaller interval
    ir: larger interval

    Return:
    None or pd.Interval
    """
    # Return `None` if two intervals can't be concatenated
    if not (il.overlaps(ir) or il.left == ir.right or il.right == ir.left):
        return None

    # Determine left edge
    if il.left < ir.left:
        ll, lclosed = il.left, il.closed_left
    elif il.left > ir.left:
        ll, lclosed = ir.left, ir.closed_left
    else:
        ll, lclosed = il.left, il.closed_left or ir.closed_left

    # Determine right edge
    if il.right < ir.right:
        rr, rclosed = ir.right, ir.closed_right
    elif il.right > ir.right:
        rr, rclosed = il.right, il.closed_right
    else:
        rr, rclosed = ir.right, il.closed_right or ir.closed_right

    # Determine `closed`
    if lclosed and rclosed:
        closed = "both"
    elif lclosed and not rclosed:
        closed = "left"
    elif not lclosed and rclosed:
        closed = "right"
    else:
        closed = "neither"

    return pd.Interval(ll, rr, closed)


def _point_interval(tgt: pd.Interval, p: Union[int, float]) -> pd.Interval:
    """
    Description:
    Concatenate interval and point.
    Note: interval will be returned directly if `p` is not in `tgt`.

    Params:
    tgt: interval
    p: point

    Return:
    pd.Interval
    """
    if tgt.left == p:
        closed = "left"
        if tgt.closed_right is True:
            closed = "both"
        return pd.Interval(tgt.left, tgt.right, closed)
    elif tgt.right == p:
        closed = "right"
        if tgt.closed_left is True:
            closed = "both"
        return pd.Interval(tgt.left, tgt.right, closed)
    return tgt


# %%
def tidy_up(seq: list) -> Any:
    """
    Description:
    Tidy the elements in `seq` up, including:
    1. Concatenate intervals
    2. Remove duplicated point
    3. Full interval's edge with points

    Params:
    seq: list of intervals, int, float and frozenset

    Return:
    List
    """
    no_set = []
    # Fronzenset will flatened at first to be sorted
    for ele in seq:
        if type(ele) in TYPE_ENUMS["fset"]:
            no_set.extend(ele)
        else:
            no_set.append(ele)
    # Sort sequences with `min_key` for conviniences
    no_set = sorted(no_set, key=min_key)
    flattened = [no_set[0]]
    for ele in no_set[1:]:
        # If last element is interval, check if merged can be done
        if type(flattened[-1]) in TYPE_ENUMS["interval"]:
            if type(ele) in TYPE_SINGLE:
                # Put the latest point in last interval if could
                flattened[-1] = _point_interval(flattened[-1], ele)
                if ele not in flattened[-1]:
                    flattened.append(ele)
            elif type(ele) in TYPE_ENUMS["fset"]:
                flattened.extend(
                    [sub_ele for sub_ele in ele if ele not in flattened[-1]]
                )
            elif type(ele) in TYPE_ENUMS["interval"]:
                if (
                    ele.overlaps(flattened[-1])
                    or ele.left == flattened[-1].right
                ):
                    flattened[-1] = concat_interval(flattened[-1], ele)
                else:
                    flattened.append(ele)
            else:
                logger.warning(f"Invalid sequence element: {ele}")
        # If last element is single, check if merged can be done
        else:
            # Put the last point in the latest interval if could.
            # Then replace the last point with latest interval and
            #   skip `append`
            if type(ele) in TYPE_ENUMS["interval"]:
                ele = _point_interval(ele, flattened[-1])
                if flattened[-1] in ele:
                    flattened[-1] = ele
                    continue
            # Append if latest element is different from last point, no matter
            #   what the type of the latest element is.
            if ele != flattened[-1]:
                flattened.append(ele)

    return flattened


# %% ------------------------------------------------------------------------
#       * * * * Type Detecter * * * *
# ---------------------------------------------------------------------------
def infer_dtype(val: type | Any, to_float: bool = False) -> str | tuple:
    """
    Description:
    Infer the data type of `val` or the elements in `val`, representing
    with a string.
    1. For iterables, list, Series, np.ndarray, for eaxmple,
        `pd.api.types.infer_dtype` is used.
    2. Else, enumerate `TYPE_ENUMS` if `val` is a type.
    and most scalar, except pd.Interval,

    Params:
    val:
    to_float: if unify `integer` to `floating`

    Return:
    iterable val: dtype-string, type(val)
    scalar val: dtype-string
    """
    # Check data type in order.
    if isinstance(val, Iterable) and not isinstance(val, str):
        type_str = pd.api.types.infer_dtype(val), type(val)
    # Enumerate `TYPE_ENUMS` to check data type.
    elif isinstance(val, type):
        for type_str, type_clss in TYPE_ENUMS.items():
            if issubclass(val, type_clss):
                break
    else:
        for type_str, type_clss in TYPE_ENUMS.items():
            if isinstance(val, type_clss):
                break

    # Unify string representing data type.
    if to_float and type_str == "integer":
        type_str = "floating"
    if type_str == "datetime64":
        type_str == "datetime"

    return type_str


def detect_str_dtype(
    val: str,
    regex: dict | None = None,
    how: str = REGEX_HOW,
    keep_type: bool = True,
    match_ratio: float = REGEX_MATCH_RATIO,
    to_float: bool = False,
    return_match: bool = False,
) -> str:
    """
    Description:
    Detect the probably proper data type of a string with regex, as almost
    all kinds of objects could be serielized and stored as a string.

    Params:
    val: string to be detected
    how: how the regex works while detecting
    regex: the data type regex for detecting. `TYPE_REGEX` will be used
        in default if none passed.
    keep_type: determining if merging passed `regex` with `TYPE_REGEX`
    to_float: determining if cast `int` to `float`
    match_ratio: the threshold for regex matching, which is evaluated by
        the ratios of the matched characters in the whole string. This only
        works when how is not `fullmatch`.

    Return:
    string representing data type.
    """
    if keep_type:
        regex = (
            ChainMap(regex, TYPE_REGEX) if regex is not None else TYPE_REGEX
        )
    match_ratio = min(match_ratio, 1)
    if match_ratio == 1:
        how = "fullmatch"

    # Use `re.search`、`match`、`fullmatch` to find the **first** matched
    # substring in `s`, and then cast the matched substring to given type.
    # set_trace()
    type_str, matched_part = "unknown", None
    for _type_str, ptn in regex.items():
        matched = getattr(re, how)(ptn, val)
        if not matched:
            continue

        # `match_ratio` could be calculated for `fullmatch` which must be 1.
        start, end = matched.span()
        # Note that `val` doesn't need to be stripped when `search` is passed
        # as `how` is some cases. But we set `match_ratio` here, which may be
        # effected by surplus blanks.
        if (end - start) / len(val.strip()) >= match_ratio:
            type_str = _type_str
            matched_part = matched.group()
            if to_float and type_str == "integer":
                type_str = "floating"
            break

    if return_match:
        return type_str, matched_part
    else:
        return type_str


def detect_dtype_anyway(
    val: Any,
    regex: dict | None = None,
    how: str = REGEX_HOW,
    keep_type: bool = True,
    match_ratio: float = REGEX_MATCH_RATIO,
    to_float: bool = False,
) -> str:
    """
    Description:
    Detect the possible data type of `val`, representing with a string.
    1. If `val` is a string, detect the probably proper data type of a string
        with regex with `detect_str_type`
    2. Else try `infer_dtype`

    Params:
    val:
    how:
    regex:
    keep_type:
    match_ratio:
    to_float:

    Return:
    """
    if isinstance(val, str):
        return detect_str_dtype(
            val, regex, how, keep_type, match_ratio, to_float, False
        )
    else:
        return infer_dtype(val, to_float)


# %%
def infer_major_dtype(
    ser: list | tuple | pd.Series | np.ndarray,
    sample_n: int = SAMPLE_N,
    detect_str: bool = True,
    to_float: bool = False,
    return_freqs: bool = False,
    **kwargs,
) -> tuple:
    """
    Description:
    Infer the major data type in `ser`, while string in `ser` will be tried
    to be casted to proper data type if `regex` is not None.

    Params:
    ser:
    sample_n:
    detect_str: if detecting possible data type for strings in `ser`
    kwargs: keyword arguments passed `detect_dtype_anyway`, among which
        elements should starts with `detect_str_`

    Return:
    major_dtype, major_dtype frequency
    """
    # Sample from `ser`, in case of its length.
    ser_nona = pd.Series(ser).dropna()
    # In case `ser` if full of na
    if len(ser_nona) == 0:
        sample = ser.sample(min(ser.size, sample_n))
    else:
        sample = ser_nona.sample(min(ser_nona.size, sample_n))

    # Check data types in sample
    if detect_str:
        detect_kwargs = {
            k[11:]: v for k, v in kwargs.items() if k.startswith("detect_str_")
        }
        dtype_ratios = sample.apply(
            detect_dtype_anyway, **detect_kwargs
        ).value_counts(sort=True, normalize=True)
    else:
        dtype_ratios = pd.value_counts(
            sample.apply(infer_dtype, to_float=False),
            sort=True,
            normalize=True,
        )

    # Add an item with floating and integer together.
    if to_float and "floating" in dtype_ratios and "integer" in dtype_ratios:
        dtype_ratios["floating-integer"] = (
            dtype_ratios["floating"] + dtype_ratios["integer"]
        )
        logger.info(
            "Combine floating(%s) and integer(%s) together.",
            dtype_ratios["floating"],
            dtype_ratios["integer"],
        )
        dtype_ratios = dtype_ratios.sort_values(ascending=False)

    if not return_freqs:
        return dtype_ratios.index[0], dtype_ratios.values[0]
    else:
        return dtype_ratios.index[0], dtype_ratios.values[0], dtype_ratios


# %%
# ----------------------------------------------------------------------------
#           * * * Typer Caster * * *
# ----------------------------------------------------------------------------
def regex_caster(
    val: Any,
    target: str | None = None,
    regex_how: str = REGEX_HOW,
    match_ratio: float = REGEX_MATCH_RATIO,
    type_regex: list | dict | None = None,
    str_caster: list | dict | None = None,
) -> int:
    """
    Description:
    Cast `val` to data type `target`.
    Mostly, this is used to cast string to another data type, vice versa,
    since it's not common for a value to be casted to abitrary data type,
    but some exceptions:
    1. datetime <-> integer
    2. integer <-> floating

    Params:
    val:
    target: target data type represented with string
    match_ratio:
    type_regex:
    str_caster:
    kwargs: keyword arguments for `detect_str_dtype`

    Return:
    """
    # Skip na and return None directly.
    # If `val` is iterable, `pd.isna` returns Series of bool, which can't be
    #   booled simply.
    if not isinstance(val, Iterable) and pd.isna(val):
        return None
    # Call `str` directly if target is `"string"`.
    if target == "string":
        return repr(val)

    match_ratio = min(match_ratio, 1)
    if match_ratio == 1:
        regex_how = "fullmatch"

    # set_trace()
    type_str = infer_dtype(val)
    # Cast `val` to `target` with regex search if `val` is a string.
    if type_str == "string":
        type_regex = (
            ChainMap(type_regex, TYPE_REGEX)
            if type_regex is not None
            else TYPE_REGEX
        )
        if target is not None:
            type_regex = {target: type_regex[target]}

        target_, matched_part = detect_str_dtype(
            val,
            regex=type_regex,
            how=regex_how,
            keep_type=False,
            match_ratio=match_ratio,
            to_float=False,
            return_match=True,
        )

        # Return None if dtype is unknown.
        if target_ not in type_regex or matched_part is None:
            # If `target` is specified, return None,
            # Else return `val`.
            return val if target is None else None

        str_caster = (
            ChainMap(str_caster, STR_CASTER)
            if str_caster is not None
            else STR_CASTER
        )
        if target_ not in str_caster:
            logger.warning(
                "Can't fetch valid string caster for dtype: %s", target_
            )
            return val if target is None else None
        else:
            return str_caster[target_](matched_part)
    # Return directly if no need to cast.
    elif type_str == target:
        return val
    # Else only valid type conversion pairs make sense.
    elif f"{type_str}-{target}" in VALID_CASTER:
        return VALID_CASTER[f"{type_str}-{target}"](val)
    else:
        return val if target is None else None


# %%
def keep_dtype_caster(val: Any, dtype: str, default: Any = None) -> Any:
    """
    Description:
    Return `val` directly if its dtype fits with `dtype`, else return
    `default`.

    Params:
    val:
    dtype:
    default:

    Return:
    """
    return val if infer_dtype(val) == dtype else default
