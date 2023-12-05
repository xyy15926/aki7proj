#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pdchain.py
#   Author: xyy15926
#   Created: 2023-04-13 10:24:08
#   Updated: 2023-04-24 12:25:34
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
import logging
import ringbear.dtyper
import ringbear.pdler
import ringbear.npam
import ringbear.biclf
# from importlib import reload
# reload(ringbear.dtyper)
# reload(ringbear.pdler)
# reload(ringbear.npam)
# reload(ringbear.biclf)
from ringbear.dtyper import infer_major_dtype, regex_caster
from ringbear.pdler import (
        autotype_ser, fill_numeric_ser,
        cut_numeric_with_outliers, filter_outliers)
from ringbear.npam import get_outliers_with_sigma
from ringbear.biclf import lift_ordered, woe_ordered

# %%
FACTOR_MAX = 10
NEG_FLAG = -999999
NA_FLAG = "NA"
NUMERIC_DTYPE_STR = ("integer", "floating", "datetime")
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def sketch_handler(
    ser: pd.Series,
    label: pd.Series,
    sketch_logger: dict | None = None,
    marker_logger: dict | None = None,
    factors_n: int = FACTOR_MAX,
    autotype_thresh: float = 0.95,
) -> pd.Series:
    """
    Description:
    Preprocessing `ser` along with sketching in following steps:
    1. Sketch orignial `ser` basicly
    2. Cast `ser` to proper dtype and then sketch the result basicly
    3. Fill `ser` of unified dtype.
    3.1 For numeric `ser`, `fill_numeric_ser` is called, and then check
        uniques' number to determine if cut will be applied.
    3.2 For categorical `ser`, fill na with `NA` directly.

    Params:
    ser:
    label:
    sketch_logger: dict for storing sketches
    marker_logger: dict for storing markers
    factors_n: the maximum number of uniques for sketching and half of the
        number of the uniques for cutting.
    autotype_thresh:

    Return:
    preprocessed series
    """
    skt = {}
    skt.update(sketch_series_basicly(
        ser, factors_n=factors_n, sketch_prefix="ori_"))
    major_dtype, major_freq = \
        skt["ori_major_dtype"], skt["ori_major_freq"]
    unique_n = skt["ori_unique_n"]

    # Autotype and then sketch.
    # Note that `major_freq` will change after `detect_str`.
    if major_dtype not in NUMERIC_DTYPE_STR or major_freq < 1:
        typed_ser = autotype_ser(ser, caster=regex_caster, detect_str=True)
        skt.update(sketch_series_basicly(
            typed_ser, factors_n=factors_n, sketch_prefix="typed_"))
        major_dtype, major_freq = \
            skt["typed_major_dtype"], skt["typed_major_freq"]
        unique_n = skt["typed_unique_n"]
    else:
        typed_ser = ser

    # Fill series and then sketch
    # Note that:
    # In fact, it can't be easy to handle invalid value. But some rules could
    # be followed as some common senses:
    # 1. For categorical usage, just `fillna` will be fine.
    # 2. For ordering usage, check the inner meaning of na and outliers, then
    #     determine the mapper with the consideration of the normal values.
    # 3. For numeric usage, map na and outlier to mean or mode is a convinient
    #     way, with marker dataframe recovering the lost information.
    if major_dtype in NUMERIC_DTYPE_STR and major_freq == 1:
        # Fill numeric and return fill-mapper
        filled_ser, mapper = fill_numeric_ser(ser, typed_ser,
                                              return_mapper=True)
        skt["filled_mapper"] = mapper

        # Convert dtypes, mainly from floating to integer if possible.
        filled_ser = filled_ser.convert_dtypes()
        skt.update(sketch_series_basicly(
            filled_ser, factors_n=factors_n, sketch_prefix="filled_"))

        # Sketch with label
        skt.update(sketch_ordered_with_label(
            filled_ser, label, factors_n=factors_n, sketch_prefix="filled_"))

        # Cut `ser` into bins if need.
        if unique_n > 2 * factors_n:
            bin_ser, lowiers, highiers = cut_numeric_with_outliers(
                filled_ser, cuts_n=factors_n)
            skt.update(sketch_categorical_alone(
                bin_ser, factors_n=factors_n, sketch_prefix="cut_"))
            skt.update(sketch_ordered_with_label(
                bin_ser, label, factors_n=factors_n, sketch_prefix="cut_"))
    else:
        # Fill categorical series with `NA_FLAG` directly.
        filled_ser = typed_ser.fillna(NA_FLAG)
        skt.update(sketch_categorical_with_label(
            filled_ser, label, factors_n=factors_n, sketch_prefix="filled_"))

    logger.info("Sketch %s and return filled-series with dtype of %s.",
                ser.name, major_dtype)

    if sketch_logger is not None:
        sketch_logger[ser.name] = skt
    return filled_ser


# %%
def sketch_categorical_with_label(
    ser: pd.Series,
    label: pd.Series,
    factors_n: int = FACTOR_MAX,
    key_order: list | None = None,
    sketch_prefix: str = "",
) -> dict:
    """
    Description:
    Sketch `ser` with `label` for:
    1. lifts, accelerating lifts
    2. kendell correlations and p-value
    3. accumulating keys
    Note that `sketch_ordered_with_label` is called here after categorizing
    `ser`.
    Note that `OrdinalEncoder` may be good for this.

    Params:
    ser:
    label:
    factors_n:
    key_order: categorizing order
    sketch_prefix:

    Return:
    """
    skt = {}

    # Categorize to encode series ordinally.
    key_order = pd.Index(pd.unique(ser)).dropna() \
        if key_order is None else pd.Index(key_order)
    cat_ser = ser.astype(pd.CategoricalDtype(key_order, ordered=True))
    skt["key_order"] = key_order.tolist()

    # Used `.cat.codes` instead of orignal series for sketch with label.
    code_skt = sketch_ordered_with_label(
        cat_ser.cat.codes, label,
        factors_n=factors_n, sketch_prefix="")

    # Map codes back to orignal values.
    for key in code_skt:
        if isinstance(code_skt[key], dict):
            val = {key_order[kk]: vv for kk, vv in code_skt[key].items()}
            code_skt[key] = val
        elif "uniques" in key:
            code_skt[key] = [key_order[ele] for ele in code_skt[key]]
        elif "unique" in key:
            code_skt[key] = key_order[code_skt[key]]
    skt.update(code_skt)

    return {f"{sketch_prefix}{k}": v for k, v in skt.items()}


def sketch_ordered_with_label(
    ser: pd.Series,
    label: pd.Series,
    factors_n: int = FACTOR_MAX,
    acc_keys: list | None = None,
    sketch_prefix: str = "",
) -> dict:
    """
    Description:
    Sketch `ser` with `label` for:
    1. lifts, accelerating lifts
    2. kendell correlations and p-value
    3. accumulating keys

    Params:
    ser:
    label:
    factors_n:
    sketch_prefix:

    Return:
    sketch-dict
    """
    skt = {}

    # Caculate woes and ives and then sketch.
    skt["iv"] = np.sum(woe_ordered(ser, label)[2])

    # Calculate the lifts and register in sketch.
    uniques, lifts, acc_keys, acc_lifts, acc_ascending, corr_ken, pv = \
        lift_ordered(ser, label, acc_keys=acc_keys,
                     return_counts=False, return_cor=True)

    # Sort `lifts` and `acc_lifts`
    lift_topk = np.argsort(lifts)[:-factors_n-1:-1]
    acc_lift_topk = np.argsort(acc_lifts)[:-factors_n-1:-1]

    # Sketch top k wihth `XXX_topk` before, which is limited by `factors_n`.
    skt["lift_topk"] = {k: v for k, v in
                        zip(uniques[lift_topk], lifts[lift_topk])}
    skt["acc_lift_topk"] = {k: v for k, v in
                            zip(acc_keys[acc_lift_topk],
                                acc_lifts[acc_lift_topk])}

    # Sketch stratified samples if not all lifts or acc_lifts are sketched.
    if len(uniques) <= factors_n:
        unique_sample = \
            np.linspace(0, len(uniques)-1, factors_n-1, dtype=np.int32)
        skt["lift_sample"] = \
            {k: v for k, v in
             zip(uniques[unique_sample], lifts[unique_sample])}
    if len(acc_keys) <= factors_n:
        acc_sample = \
            np.linspace(0, len(acc_keys)-1, factors_n-1, dtype=np.int32)
        # Sketch stratified sample
        skt["acc_lift_sample"] = {k: v for k, v in
                                  zip(acc_keys[acc_sample],
                                      acc_lifts[acc_sample])}

    # Sketch top1 lifts and acc_lifts.
    skt["lift_top1_unique"], skt["lift_top1"] = \
        uniques[lift_topk[0]], lifts[lift_topk[0]]
    skt["acc_lift_top1_unique"], skt["acc_lift_top1"] = \
        uniques[acc_lift_topk[0]], acc_lifts[acc_lift_topk[0]]

    # Sketch kendell correlation and p-value.
    skt["lift_corr_ken"], skt["lift_p_value"] = corr_ken, pv
    skt["lift_acc_keys"] = acc_keys.tolist()

    return {f"{sketch_prefix}{k}": v for k, v in skt.items()}


# %%
def sketch_series_basicly(
    ser: pd.Series,
    factors_n: int = FACTOR_MAX,
    sketch_prefix: str = "",
) -> dict:
    """
    Description:
    Sketch Series basicly for:
    1. na-counts
    2. dtypes' frequencies
    3. major dtype
    4. major dtype's frequences
    5. number of uniques'
    And then determine whether and how to sketch further.

    Params:
    ser:
    sketch_prefix:

    Return:
    sketch-dict
    """
    skt = {}

    # Get dtypes and their frequencies.
    major_dtype, major_freq, dtype_freqs = infer_major_dtype(
        ser, detect_str=False, to_float=True, return_freqs=True)
    skt["major_dtype"], skt["major_freq"] = major_dtype, major_freq
    skt["dtype_freqs"] = dtype_freqs.to_dict()

    # Get unqiues' number and na-ratios.
    # `dropna=False` will set `pd.NA` or `np.nan` as index, so don't use it.
    value_counts = pd.value_counts(ser, dropna=True, normalize=False)
    skt["na_freqs"] = pd.isna(ser).sum() / len(ser)
    skt["nna_freqs"] = 1 - skt["na_freqs"]
    skt["unique_n"] = len(value_counts)

    if skt["na_freqs"] == 1:
        pass
    elif skt["major_dtype"] in NUMERIC_DTYPE_STR and skt["major_freq"] == 1:
        skt.update(sketch_numeric_alone(ser, value_counts, factors_n))
    else:
        skt.update(sketch_categorical_alone(ser, value_counts, factors_n))

    return {f"{sketch_prefix}{k}": v for k, v in skt.items()}


# %%
def sketch_categorical_alone(
    ser: pd.Series,
    value_counts: pd.Series | None = None,
    factors_n: int = FACTOR_MAX,
    sketch_prefix: str = "",
) -> dict:
    """
    Decription:
    Sketch numeric Series for:
    1. lowiers, highiers
    2. unique values or bins
    3. numbers of each unique values or bins
    4. skew, kurtosis, mean, std

    Params:

    Return:
    """
    skt = {}
    value_counts = ser.value_counts(dropna=True, normalize=False, sort=True) \
        if value_counts is None else value_counts

    if len(value_counts) > factors_n:
        # Sketch major unqiues
        value_counts_in = value_counts.iloc[:factors_n]
        skt["major_unqiues"] = value_counts_in.index.tolist()
        skt["minor_unique_n"] = len(value_counts) - factors_n
        value_counts_in["minor_others"] = \
            value_counts.iloc[-skt["minor_unique_n"]:].sum()
        skt["unique_counts"] = value_counts_in.to_dict()

        # Sketch stratified sample uniques
        sample = np.linspace(0, len(value_counts)-1, factors_n, dtype=np.int32)
        value_counts_sample = value_counts.iloc[sample]
        skt["unique_samples"] = value_counts_sample.index.tolist()
        skt["unique_sample_counts"] = value_counts_sample.to_dict()
    else:
        skt["uniques"] = value_counts.index.tolist()
        skt["unique_counts"] = value_counts.to_dict()

    return {f"{sketch_prefix}{k}": v for k, v in skt.items()}


# %%
def sketch_numeric_alone(
    ser: pd.Series,
    value_counts: pd.Series | None = None,
    cuts_n: int = FACTOR_MAX,
    sketch_prefix: str = "",
) -> dict:
    """
    Decription:
    Sketch numeric Series for:
    1. lowiers, highiers
    2. unique values or bins
    3. numbers of each unique values or bins
    4. skew, kurtosis, mean, std

    Params:

    Return:
    """
    skt = {}
    # Return null dict if `ser` is full na
    if np.all(pd.isna(ser)):
        return skt
    value_counts = pd.value_counts(
            ser, dropna=True, normalize=False, sort=True) \
        if value_counts is None else value_counts

    # skt.update({f"desc_{k}": v for k, v in ser.describe().items()})
    # skt["skew"], skt["kurtosis"] = ser.skew(), ser.kurtosis()

    skt.update(sketch_categorical_alone(ser, value_counts, cuts_n))

    # Get lowiers, highers and outlier_map
    # `[::-1]` is called to reverse `.index`, so that short index with
    # outliers, which should be less frequent, can be striped out, which is
    # determined by the routines of `get_outliers_with_sigma`.
    lowiers, highiers, outliers_map = \
        get_outliers_with_sigma(value_counts.index[::-1])

    # Filter values into `ser_in`, `ser_lowiers`, `ser_highiers`
    ser_in, ser_lowiers, ser_highiers = \
        filter_outliers(ser, lowiers, highiers)

    # Add value ranges, mean, std, skew and kurtosis
    skt["lowiers_n"] = len(lowiers)
    skt["lowiers_range"] = pd.Interval(
        lowiers.min(), lowiers.max(), closed="both") \
        if len(lowiers) > 0 else None
    skt["highiers_n"] = len(highiers)
    skt["highiers_range"] = pd.Interval(
        highiers.min(), highiers.max(), closed="both") \
        if len(highiers) > 0 else None
    skt["nor_unique_n"] = len(value_counts) - len(lowiers) - len(highiers)
    skt["nor_range"] = pd.Interval(
        min(ser_in), max(ser_in), closed="both")
    # skt.update({f"nor_{k}": v for k, v in ser_in.describe().items()})
    # skt["nor_skew"], skt["nor_kurtosis"] = ser_in.skew(), ser_in.kurtosis()

    return {f"{sketch_prefix}{k}": v for k, v in skt.items()}


