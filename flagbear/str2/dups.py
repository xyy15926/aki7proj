#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dups.py
#   Author: xyy15926
#   Created: 2024-11-11 09:25:08
#   Updated: 2024-11-11 09:26:12
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Iterable, Callable, Mapping

import logging

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def rename_duplicated(ori: list[str | int | float]) -> list:
    """Rename duplicated elements inorder.

    Params:
    ---------------------------
    ori: List that may containing duplicated elements.

    Return:
    ---------------------------
    List with duplicated values renamed with `_<N>`.
    """
    cnts = {}
    new = []

    for ele in ori:
        if ele in cnts:
            cnts[ele] += 1
            times = cnts[ele]
            new.append(f"{ele}_{times}")
        else:
            cnts[ele] = 1
            new.append(ele)

    return new


# %%
def rename_overlaped(
    ori: list[Iterable],
    suffixs: Iterable = None,
) -> list[list]:
    """Rename overlaped elements inorder.

    1. Elements of list in `ori` must be unique.
    2. The elements of first list in `ori` won't be changed all the time.

    Params:
    ----------------------
    ori: Lists that may containes overlaped elements.

    Return:
    ----------------------
    Lists with overlaped elements renamed with `_<N>`.
    """
    rets = []
    cnts = set()
    suffixs = range(1, len(ori) + 1) if suffixs is None else suffixs
    for idx, ll in zip(suffixs, ori):
        ret = []
        for ele in ll:
            if ele in cnts:
                ret.append(f"{ele}_{idx}")
            else:
                ret.append(ele)
                cnts.add(ele)
        rets.append(ret)

    return rets
