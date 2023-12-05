#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: unifier.py
#   Author: xyy15926
#   Created: 2023-04-01 17:41:41
#   Updated: 2023-11-10 14:55:00
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import json
import logging
from collections import OrderedDict, defaultdict, deque
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd

from ringbear.dtyper import regex_caster

# from importlib import reload
# reload(ringbear.executor)
# reload(ringbear.dtyper)
from ringbear.executor import exec_aggstr, exec_expstr

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def update_ordered_seq(
    param: Any, default: list | dict | tuple, *, keepna: bool = False
) -> list | dict:
    """
    Description:
    Update `default` with `param`, with keep relative order, as `param`
    should be itered early than `default`. The following formats are
    supported:
    1. [val, ].extend
    2. [val, ].append
    3. dict(Series).update
    4. dict(Series).add
    5. [(key, val), ] <-> dict(Series)

    Params:
    param:
    default:
    keepna: skip param and return default directly when param is None

    Return:
    """
    if not keepna and param is None:
        return default

    # Both list and tuple keep order, so just "prepend" will be fine.
    if isinstance(default, (tuple, list)):
        if isinstance(param, (tuple, list)):
            updated = [*param, *default]
        elif isinstance(param, (dict, pd.Series)):
            updated = [*zip(*(param.items())), *default]
        else:
            updated = [param, *default]
    # Use `OrderedDict` to keep iteration order, while dict can't keep
    #   iteration order.
    elif isinstance(default, (dict, pd.Series)):
        if isinstance(param, (tuple, list)):
            if isinstance(param[0], (tuple, list)):
                assert len(param[0]) == 2
                updated = OrderedDict({k: v for k, v in param})
            # If only one k, v is provided, `param` could be (k,v) instead of
            # [(k,v), ]
            else:
                assert len(param) == 2
                updated = OrderedDict({param[0]: param[1]})
        else:
            updated = OrderedDict(param)
        # Iterate by self to ensure priority and order.
        for k, v in default.items():
            if k not in updated:
                updated[k] = v
    else:
        logger.warning("Unsupported default sequences: %s.", default)
        return default

    return updated


# %%
def unify_shape22(*arrs: np.ndarray, default: float | int = 1) -> list:
    """
    Description:
    Unify shape, type of array in `arrs` to 2D-ndarray, with length of
    arrs[0].

    Params:
    arrs:
    default: default value to fill in ndarray, if `None` is passed.

    Return:
    list of np.ndarray.
    """
    length = len(arrs[0])
    rets = []
    for arr in arrs:
        if arr is None:
            arr = np.ones(length, dtype=np.int32) * default
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        assert 1 <= arr.ndim <= 2
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        assert length == arr.shape[0]
        rets.append(arr)
    return rets


# %%
def parse_rules(rules: list) -> dict:
    """
    Description:
    Iterate rule from `rules` and build a map-tree to describe the dependency
    relationships of `rules`.

    Params:
    rules: [{key, from, deps, ...}, ...]
        key: string, name of newly parsed field
        from: string | None, refering to previous `key` or the original dict
        deps: words seperated by `,`, refering to previous `key`

    Return:
    rule_dict
    """
    rule_q, rule_tmpq, rule_deps = deque(), deque(), set()

    # Here, it's assumed that the former rules should be parsed FIFO. Thus
    # there should be few rules pushed to `rule_tmpq`. Or this process won't
    # be time-efficient.
    # Attention: `chain` and `for` is used to iterate a mutable list.
    for rule in chain(rules, rule_tmpq):
        key, from_, deps = rule["key"], rule["from"], rule["deps"]
        if from_ is not None:
            deps.append(from_)
        if deps is None or np.all([i in rule_deps for i in deps]):
            rule_q.append(rule)
            rule_deps.append(key)
        # Push `rule` to `rule_tmpq` if its `nest` not handled yet.
        else:
            rule_tmpq.append(rule)

    return rule_q


def rebuild_dict(obj: dict, rules: list) -> dict:
    """
    Description:
    `extract_field` from `obj` with `rules` to rebuild.

    Params:
    obj:
    rules:

    Return:
    """
    rets = {}
    for rule in rules:
        key, from_, deps, steps = rule["key"], rule["from"], rule["deps"]
        steps, dtype = rule["steps"], rule["dtype"]

        # Assert that all dependencies have been parsed.
        assert (np.all([i in rets for i in deps]))

        # Extract field if `steps` is provided.
        cur_obj = rets.get(from_, obj)
        if steps is not None:
            if isinstance(cur_obj, list):
                rets[key] = [extract_field(ele, steps, dtype) for ele in cur_obj]
            else:
                rets[key] = extract_field(cur_obj, steps, dtype)

    return rets


# %%
def transform_field(rets: dict, trans: str, dtype: type = None) -> Any:
    pass


# %%
def extract_field(obj: dict, steps: str, dtype: type = None,) -> Any:
    """
    Description:
    Extract field in `obj` determined by the `steps`, which representes the
    path in `obj` after following rules:
    1. `:` seperates different steps for each level
    2. `&&` seperates destiny and conditions which is checked by `exec_expstr`
        determining if current object is valid.
    3. `step` for list object must be a aggregation.
    4. `{<AGG>}` represents apply aggregation AGG on object.values()

    Params:
    obj:
    steps:
    dtype:

    Return:
    """
    # Check parameters
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            logger.debug("Unrecognized json string: %s.\n %s", obj, e)
            if steps is not None:
                return None
    steps = steps.split(":") if isinstance(steps, str) else steps

    cur_obj = obj
    for idx, step in enumerate(steps):
        # Stop early.
        if not cur_obj:
            return None

        # Reload json if necessary.
        # `cur_obj`, for example, is a JSON-string.
        if isinstance(cur_obj, str):
            try:
                cur_obj = json.loads(cur_obj)
            except json.JSONDecodeError as e:
                logger.debug(e)
                if steps is not None:
                    return None

        # Check aggregation for dict.values().
        if_agg = False
        if step[0] == "{" and step[-1] == "}":
            if_agg = True
            step = step[1:-1]

        # Split destiny and conditions.
        dest, *conds = step.split("&&")
        if conds and not exec_expstr(conds[0], cur_obj):
            return None

        # Get field directly.
        if not if_agg and isinstance(cur_obj, dict):
            cur_obj = cur_obj.get(dest, None)
        # Handle aggregation.
        else:
            # Traverse the values if aggregation is specified.
            if isinstance(cur_obj, dict):
                cur_obj = cur_obj.values()
            filtered = []
            for list_item in cur_obj:
                ret = extract_field(list_item, steps[idx + 1 :], dtype)
                # All `ret` will be append to `filtered`, even if `None`.
                filtered.append(ret)
            return exec_aggstr(dest, filtered)

    return regex_caster(cur_obj, target=dtype)


# %%
def flat_dict(
    obj: dict,
    fields: list,
    prefix: str = "",
    suffix: str = "",
    zipflat: bool = False,
) -> dict:
    """
    Description:
    Flat `obj` to 1-depth dict with only wanted `fields` extracted by
    `extract_field`.

    Params:
    obj:
    fields: [(key, steps, dtype), ], [steps, ] or [(key, step), ]
        key: field name in return dict
        steps: parameter for `extract_field`
        dtype:
    prefix:
    suffix:
    zipflat: zip extracted lists

    Return:
    """
    rets = {}
    for field in fields:
        steps, key, dtype = None, None, None
        if isinstance(field, str):
            steps = field
        elif len(field) == 1:
            steps = field
        elif len(field) == 2:
            key, steps = field
        elif len(field) == 3:
            key, steps, dtype = field
        key = key or steps[steps.rfind(":") + 1 :]
        rets[f"{prefix}{key}{suffix}"] = extract_field(obj, steps, dtype)
    if zipflat:
        for key, value in rets.items():
            if not isinstance(value, list):
                break
        else:
            return [
                {k: v for k, v in zip(rets.keys(), vs)}
                for vs in zip(*(rets.values()))
            ]
    else:
        return rets
