#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pdler.py
#   Author: xyy15926
#   Created: 2023-03-28 20:56:32
#   Updated: 2023-08-09 10:27:39
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import re
import logging
from typing import Any
import pandas as pd
import numpy as np
from collections import ChainMap
from ringbear import unifier, dtyper, executor, npam
# from importlib import reload
# reload(unifier)
# reload(dtyper)
# reload(executor)
# reload(npam)
from IPython.core.debugger import set_trace
from ringbear.dtyper import infer_major_dtype, regex_caster, infer_dtype
from ringbear.executor import AGG_MAPPER, STR_CHECKER
from ringbear.npam import get_outliers_with_sigma


# %%
NUMERIC_DTYPE_STR = ("integer", "floating", "datetime")
SAMPLE_N = 1000
CAT_TRHESH = 100
FACTOR_MAX = 30
POS_FLAG = 999999
NEG_FLAG = -999999
NA_FLAG = "NA"
NUMERIC_NAFILL = np.nanmean


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def parse_sort_keys(
    sort_keys: list | tuple | str | dict,
) -> pd.DataFrame:
    """
    Description:
    Parse sort-keys with following formats into `by` and `ascending` for
    `df.sort_values`:
    1. [KEY>, KEY<, ...]
    2. [(KEY, 1), (KEY, True), (KEY, ">"), ...]
    3. {KEY: ASCENDING, ...}

    Params:
    sort_keys:

    Return:
    by, ascending
    """
    # Prepare the `sort-keys` with different formats.
    # Here only str and int typed field names taked into consideration.
    if isinstance(sort_keys, (str, int)):
        sort_keys = [sort_keys, ]
    if isinstance(sort_keys, (tuple, list)) and \
            len(sort_keys) == 2 and \
            sort_keys[1] in [0, 1, True, False, "<", ">"]:
        sort_keys = [sort_keys, ]

    # Convert `sort_keys` for `sort_values`
    if isinstance(sort_keys, (tuple, list)):
        sort_by, sort_ascending = [], []
        for key in sort_keys:
            if isinstance(key, (tuple, list)):
                by, ascending = key
            elif key[-1] in ("<", ">"):
                by, ascending = key[:-1], key[-1]
            else:
                by, ascending = key, True
            sort_by.append(by)
            sort_ascending.append(ascending)
    elif isinstance(sort_keys, (dict, pd.Series)):
        sort_by, sort_ascending = list(zip(*sort_keys.items()))
    else:
        logger.warning("Unsupported %s for parameter `sort_keys`.",
                       sort_keys)

    return list(sort_by), [i in (True, "<", 1) for i in sort_ascending]


# %%
def drop_records(
    df: pd.DataFrame,
    unique_keys: list | tuple | str | None = None,
    keep: list | tuple | str | dict | None = None,
    sort_by: list | tuple | str | None = None,
    sort_ascending: list | tuple | bool | None = None,
    keep_columns: str | bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Description:
    Drop invalid records(rows) in `df` according to:

    Params:
    df:
    unique_keys:
    keep: aggragating key and responsible aggregation
    sort_by: sort-key for `sort_by` in `df.sort_values` or `parse_sort_keys`
    sort_ascending: sort-key-ascending for `ascending` in `df.sort_values`
    keep_columns: keep unmentioned columns
    kwargs: kwargs for `df.sort_values`

    Return:
    Dataframe with duplicated keys dropped.
    """
    # Sorting for each group seperately is increditablely slow, which may
    #   resulting from the assignment?
    # So, pre-sorting for the whole df is done and `RANGE_INDEX` is added
    #   to store the original index to restore order.
    if sort_by is not None:
        if sort_ascending is None:
            sort_by, sort_ascending = parse_sort_keys(sort_by)
        df = df.sort_values(by=sort_by, ascending=sort_ascending, **kwargs)

    # Drop duplicated directly if no further requirements.
    if keep is None or isinstance(keep, str):
        keep = keep or "first"
        result = df.drop_duplicates(unique_keys, keep=keep)
    else:
        if keep_columns:
            keep_actions = {i: "first" for i in df.columns}
            keep = ChainMap(keep, keep_actions)
        for key in keep:
            if isinstance(keep[key], str) and not hasattr(
                    pd.core.groupby.generic.SeriesGroupBy, keep[key]):
                keep[key] = AGG_MAPPER[keep[key]]
        result = df.groupby(unique_keys, sort=False) \
            .aggregate(keep)

    return result


# %%
def drop_fields(
    df: pd.DataFrame,
    drop_na: bool | float = True,
    drop_flat: bool | float = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Description:
    Drop invalid fields(columns) in dataframe, according to:

    Params:
    drop_na:
        bool: if keep columns full filled with `nan`
        float: threshold of so-called `full`
    drop_flat:
        bool: if keep columns full filled with the same value
        float: threshold of so-called `full`
    kwargs: key-word arguments, which should be an instance of:
        list | str | None, default None,
    prefix: invalid prefix for field names
    suffix: invalid suffix for field names
    contains: invalid contains for field names
    fullmatch: invalid fullmatch-regex pattern for field names
    search: ditto
    match: ditto

    Return:
    dataframe
    """

    # Drop na-columns and flat-columns.
    if drop_na or drop_flat:
        field_indices = []
        for idx, field in enumerate(df.columns):
            append_flag = True
            value_freq = df.iloc[:, idx].value_counts(
                sort=True, dropna=False, normalize=True)

            # Drop columns with to many nas.
            na_freq = value_freq[value_freq.isna()].sum()
            if drop_na and na_freq >= float(drop_na):
                logger.info("Column %s are dropped for being filled with na "
                            "up to %s.", field, na_freq)
                append_flag = False

            # Drop flat columns, aka columns with full filled with the same
            # value.
            if drop_flat and value_freq.iloc[0] >= float(drop_flat):
                logger.info("Column %s are dropped for being filled with the "
                            "%s up to %s.", field, value_freq.index[0],
                            value_freq.iloc[0])
                append_flag = False

            if append_flag:
                field_indices.append(idx)
        df = df.iloc[:, field_indices]

    # Checkers for each kind of invalid field name pattern.
    if kwargs:
        enum_fields = list(enumerate(df.columns.tolist()))
        for key, val in kwargs.items():
            if key in STR_CHECKER:
                checker = STR_CHECKER[key]
                if isinstance(val, str):
                    val = (val, )
                if isinstance(val, (list, tuple)):
                    for i in val:
                        enum_fields = filter(lambda x: not checker(x[1], i),
                                             enum_fields)
            else:
                logger.warning("Invalid keyword parameter %s with value %s.",
                               key, val)
        # Note that duplicated fields-name will result columns grows rapidly,
        # if `iloc` is not used.
        df = df.iloc[:, list(list(zip(*enum_fields))[0])]

    return df


# %%
def autotype_ser(
    ser: pd.Series,
    targets: None | list = None,
    caster: list | tuple | dict | None = None,
    autotype_thresh: float = 0.95,
    sample_n: int = SAMPLE_N,
    detect_str: bool = True,
    sketch: str | bool = False,
    **kwargs,
) -> pd.Series:
    """
    Description:

    Params:
    ser:
    targets:
    caster: callable with one argument `target`, representing the target
        data type trying to casting to.
    autotype_thresh:
    sample_n:
    detect_str:
    kwargs: keyword arguments passed `detect_dtype_anyway`, called by
        `infer_major_dtype`, among which elements should starts with
        `detect_str_`

    Return:
    """
    skt = {}
    # Return directly if null Series or string Series passed-in.
    # NOTE: As `pd.api.types.is_string_dtype` only check the `dtype` if
    #   `Series` if provided and returns `True` when `Series` is of both
    #   `object` dtype and `string` dtype, no matter what the real values are,
    #   it's fine to just return the original `Series` if `True` is returned
    #   by `pd.api.types.is_string_dtype`.
    ser = pd.Series(ser)
    if not pd.api.types.is_string_dtype(ser) or ser.isna().all():
        return ser

    targets = NUMERIC_DTYPE_STR if targets is None else \
        set([*NUMERIC_DTYPE_STR, *targets])
    # `caster` must be passed and updated responsibly to `regex`.
    major_dtype, major_freq, dtype_freqs = infer_major_dtype(
        ser, sample_n, detect_str, True, True, **kwargs)
    if major_dtype == "floating-integer":
        major_dtype = "floating"

    skt["major_dtype"], skt["major_freq"] = major_dtype, major_freq
    skt["dtype_freqs"] = dtype_freqs.to_dict()

    if major_freq >= autotype_thresh and major_dtype in targets:
        logger.info("Try to unify series data type to %s with about %s of "
                    "total, which should be safe.", major_dtype, major_freq)
        if caster is None:
            match_ratio = kwargs.get("detect_str_match_ratio", 0.8)
            ser = ser.apply(regex_caster,
                            target=major_dtype,
                            regex_how="search",
                            match_ratio=match_ratio)
        else:
            ser = ser.apply(caster, args=(major_dtype,))
    else:
        logger.info("Can't unify %s to any data type with major dtype %s "
                    "with proportion of about %s.",
                    ser.name, major_dtype, major_freq)

    if sketch is not False:
        return ser, {f"{sketch}{k}": v for k, v in skt.items()}
    else:
        return ser


# %%
def cut_numeric_with_outliers(
    ser: pd.Series,
    cuts_n: int = FACTOR_MAX,
    outlier_with: np.ndarray | pd.Series | None = None,
    lowiers: np.ndarray | None = None,
    highiers: np.ndarray | None = None,
    outliers_cuts_n: int = FACTOR_MAX // 2,
    reorder: bool = True,
) -> tuple:
    """
    Description:
    Cut numeric after excluding the outliers.
    NOTE that `.[original-index]` is used at last to restore the `ser`'s
    order.

    Params:
    ser:
    cuts_n:
    outlier_with:
    lowiers:
    highers:
    outliers_cuts_n:
    reorder:

    Return:
    cut_bins, lowiers, highiers
    """
    # Check `outlier_with` and get `lowiers` and `highiers` if not provided.
    ser_index = ser.index
    outlier_with = np.unique(ser) if outlier_with is None else outlier_with
    if lowiers is None and highiers is None:
        lowiers, highiers, outliers_map = \
            get_outliers_with_sigma(outlier_with)

    ser_in, ser_lowiers, ser_highiers = \
        filter_outliers(ser, lowiers, highiers)

    # Cut.
    uniques = np.unique(ser)
    in_cuts_n = min(cuts_n, len(uniques) - len(lowiers) - len(highiers))
    in_cut = pd.cut(ser_in, in_cuts_n)
    if len(lowiers) > 0:
        lowiers_cuts_n = min(outliers_cuts_n, len(lowiers))
        lowiers_cut = pd.cut(ser_lowiers, lowiers_cuts_n)
        in_cut = pd.concat([lowiers_cut, in_cut], axis=0)
    if len(highiers) > 0:
        highiers_cuts_n = min(outliers_cuts_n, len(highiers))
        highiers_cut = pd.cut(ser_highiers, highiers_cuts_n)
        in_cut = pd.concat([in_cut, highiers_cut], axis=0)

    return in_cut[ser_index], lowiers, highiers


def filter_outliers(
    ser: pd.Series,
    lowiers: np.ndarray,
    highiers: np.ndarray,
) -> tuple:
    """
    Description:
    Filter `ser` into `ser_in`, `ser_lowiers` and `ser_highiers`

    Params:

    Return:
    """
    ser_in, ser_lowiers, ser_highiers = \
        ser, pd.Series(dtype="f"), pd.Series(dtype="f")
    if len(lowiers) > 0 and len(highiers) > 0:
        ser_in = ser[(ser > lowiers.max()) & (ser < highiers.min())]
        ser_lowiers = ser[ser <= lowiers.max()]
        ser_highiers = ser[ser >= highiers.min()]
    elif len(lowiers) > 0:
        ser_in = ser[ser > lowiers.max()]
        ser_lowiers = ser[ser <= lowiers.max()]
    elif len(highiers) > 0:
        ser_in = ser[ser < highiers.min()]
        ser_highiers = ser[ser >= highiers.min()]
    else:
        ser_in = ser

    return ser_in, ser_lowiers, ser_highiers


# %%
def fill_numeric_ser(
    ser: pd.Series,
    valid_ser: pd.Series | None = None,
    mapper: dict | pd.Series | None = None,
    return_mapper: bool = False,
) -> pd.Series:
    """
    Descripion:
    Fill numeric `valid_ser` with map original `ser` to fill values. If mapper
    isn't provided explicily, mapper will be built by comparing two series.

    Params:

    Return:
    filled series, mapper
    """
    valid_ser = autotype_ser(ser) if valid_ser is None else valid_ser

    # Build `mapper` by comparing two series
    if mapper is None:
        na_ser = ser[valid_ser.isna()]
        codes, factors = pd.factorize(na_ser, na_sentinel=None)
        codes += NEG_FLAG
        mapper = {k: v for k, v in zip(factors, codes)}
    ser = valid_ser.combine_first(ser.replace(mapper))

    if not return_mapper:
        return ser
    else:
        return ser, mapper


def build_marker(
    ser: pd.Series,
    valid_ser: pd.Series,
) -> pd.DataFrame:
    """
    Description:
    Mark na and invalid positions in `ser` by comparing `ser` and `valid_ser`.

    Params:
    ser:
    valid_ser:

    Return:
    marker dataframe
    """
    na_flags = valid_ser.isna()
    ser_ = ser.copy()
    # Replace valid values for convenience for removing after `get_dummies`
    ser_[~na_flags] = "NOTNA_FLAG"
    marker = pd.get_dummies(ser_, prefix=ser.name, dummy_na=True)
    if ser.name is None:
        marker = marker.drop("NOTNA_FLAG", axis=1)
    else:
        marker = marker.drop(f"{ser.name}_NOTNA_FLAG", axis=1)

    return marker


# %%
def categorize_ser(
    ser: pd.Series,
    cat_thresh: int = 8
) -> pd.Series:
    pass


# # %%
# def fill_numeric_with_valid(
#     valid_ser: pd.Series,
#     cat_thresh: int = FACTOR_MAX,
#     cat_fillna: callable | int | float = NEG_FLAG,
#     num_fillna: callable | int | float = NUMERIC_NAFILL,
# ) -> pd.Series:
#     """
#     Description:
#     Fill `valid_ser` according to its factors.
#     1. If factors is less than the threshold, `valid_ser` may be more fit
#         to be treated as categorical, which `cat_fillna` will be used to
#         fill NA.
#     2. Else `num_fillna` will be used to fill NA.

#     Params:
#     valid_ser: Series with only valid values
#     cat_thresh:
#     cat_fillna:
#     num_fillna:

#     Return:
#     """
#     unis = np.unique(valid_ser)
#     # If there are only few unique values in `valid_ser`, it will be treated
#     # as a categorical Series, and `valid_ser`-independent `cat_fillna` will
#     # be used to fill na.
#     if len(unis) < cat_thresh:
#         if callable(cat_fillna):
#             cat_fillna = cat_fillna(valid_ser)
#         return valid_ser.fillna(cat_fillna)
#     # Else, `valid_ser` depentdent `num_fillna` will be call be determine the
#     # na-fill value.
#     else:
#         if callable(num_fillna):
#             num_fillna = num_fillna(valid_ser)
#         return valid_ser.fillna(num_fillna)


# def fill_numeric_with_outliers(
#     ser: pd.Series,
#     valid_ser: pd.Series,
#     nafill: callable | int | float = NEG_FLAG,
#     lowiers: callable | list | dict | None = None,
#     highiers: callable | list | dict | None = None,
# ) -> pd.Series:
#     """
#     Description:
#     Fill `ser` according to `nafill`、`lowiers`、`highiers`.

#     Params:
#     ser:
#     valid_ser: Series with only valid values
#     nafill: value to fill NA
#     lowiers: values that should be mapped to be smaller than valid values
#     highiers: values that should be mapped to be larger than valid values

#     Return:
#     """
#     # Fill NA first.
#     if callable(nafill):
#         nafill = nafill(valid_ser)
#     ser = ser.fillna(nafill)

#     outlier_mapper = {}
#     if lowiers is not None:
#         # Call `lowiers` with `ser`, `valid_ser` to determine `lowiers`.
#         if callable(lowiers):
#             lowiers = lowiers(ser, valid_ser)
#         # Build replacement mapper for `lowiers`.
#         if isinstance(lowiers, (tuple, list)):
#             outlier_mapper.update(
#                     {k: v for k, v in
#                      zip(lowiers, range(NEG_FLAG, NEG_FLAG + len(lowiers)))})
#         elif isinstance(lowiers, (dict, pd.Series)):
#             outlier_mapper.update(lowiers)
#         else:
#             outlier_mapper[lowiers] = NEG_FLAG
#     if highiers is not None:
#         if callable(highiers):
#             highiers = highiers(ser, valid_ser)
#         if isinstance(highiers, (tuple, list)):
#             outlier_mapper.update(
#                     {k: v for k, v in
#                      zip(highiers, range(POS_FLAG - len(highiers), POS_FLAG))})
#         elif isinstance(highiers, (dict, pd.Series)):
#             outlier_mapper.update(highiers)
#         else:
#             outlier_mapper[highiers] = NEG_FLAG
#     ser = ser.replace(outlier_mapper)

#     return ser


# def fill_ser(
#     ser: pd.Series,
#     valid_ser: pd.Series | None = None,
#     nafill: callable | int | float = NEG_FLAG,
#     lowiers: callable | list | dict | None = None,
#     highiers: callable | list | dict | None = None,
#     cat_thresh: int = FACTOR_MAX,
#     cat_fillna: callable | int | float = NEG_FLAG,
#     num_fillna: callable | int | float = NUMERIC_NAFILL,
#     return_marker: bool = True
# ) -> tuple | pd.Series:
#     """
#     Decription:
#     Map invalid values in `ser`, including nan and outliers. And then
#     return both mapped `ser` and a dataframe marking the position of invalid
#     values series.
#     In fact, it can't be easy to handle invalid value. But some rules could
#     be followed as some common senses:
#     1. For categorical usage, just `fillna` will be fine.
#     2. For ordering usage, check the inner meaning of na and outliers, then
#         determine the mapper with the consideration of the normal values.
#     3. For numeric usage, map na and outlier to mean or mode is a convinient
#         way, with marker dataframe recovering the lost information.

#     Params:
#     ser:
#     valid_ser:
#     nafill:
#     lowiers:
#     highiers:
#     cat_thresh:
#     cat_fillna:
#     num_fillna:
#     return_marker:

#     Return:
#     """
#     valid_ser = autotype_ser(ser) if valid_ser is None else valid_ser
#     dtype = infer_dtype(valid_ser)
#     if dtype[0] in NUMERIC_DTYPE_STR:
#         if lowiers is not None or highiers is not None:
#             result = fill_numeric_with_outliers(ser, valid_ser, nafill,
#                                                 lowiers, highiers)
#         else:
#             result = fill_numeric_with_valid(valid_ser, cat_thresh,
#                                              cat_fillna, num_fillna)
#             na_flag = valid_ser.isna()
#             if return_marker and na_flag.sum() > 0:
#                 ser_ = ser.copy()
#                 ser_[~na_flag] = "NOTNA_FLAG"
#                 marker = pd.get_dummies(ser_, prefix=ser.name, dummy_na=True)
#                 if ser.name is None:
#                     marker = marker.drop("NOTNA_FLAG", axis=1)
#                 else:
#                     marker = marker.drop(f"{ser.name}_NOTNA_FLAG", axis=1)

#                 return result, marker
#     elif dtype[0] == "categorical":
#         result = ser.cat.add_categories(NA_FLAG).fillna(NA_FLAG)
#     else:
#         result = ser.fillna(NA_FLAG)

#     return result

