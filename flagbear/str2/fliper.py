#   Name: fliper.py
#   Author: xyy15926
#   Created: 2023-12-07 10:41:26
#   Updated: 2023-12-07 15:17:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Iterable, Callable, Mapping

import logging
import json
from collections import deque

from flagbear.llp.parser import EnvParser
from flagbear.str2.dtyper import stype_spec, str_caster
from IPython.core.debugger import set_trace

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def extract_field(
    obj: dict | str,
    steps: str,
    envp: EnvParser | None = None,
    dtype: str = None,
    extended: bool = False,
    *,
    dfill: Any = None,
    dforced: bool = False,
) -> Any:
    """Extract field from dict.

    Extract field in `obj` indicated by the `steps`. String that can be
    converted to dict is also supported, but only JSON string currently.
    1. If no proper field found in `obj`, None will be returned.
    2. `steps` should be after following rules:
      1. `:`: Seperates different steps for each level.
      2. `&&`: Seperates target and conditions which will be checked by
        `exec_expstr` for validation.
      3. `[<AGG>]`: represents apply aggregation AGG on the results.
      4. `{<AGG>}`: represents apply aggregation AGG on the results of the the
        `values()`.
      5. Step for list object must be a aggregation.
    3. This will be called recursively to for each step in steps.
    4. Try type casting iff `dtype` is provided to cast dtype from `str` to
      other dtype, and nothing will be done if the value can't be converted
      properly unless `dforced` is set.

    Attention: `dfill` will be used as the return if `deforced` is set and the
    `dtype` is not defined in in `stype_spec`.

    Example:
    ----------------
    obj = {
      "a": {
        "c": {
          "d": [1, 2, 3],
        },
        "d": 1,
      },
    }
    steps = "a:c&&d=1:d:{sum}"  -> 6
    steps = "a:c&&d=2:d:{sum}"  -> None

    Params:
    ----------------
    obj: dict | str
      dict: Dict where fields will be found.
      str: JSON string, which dict will loaded from.
    steps: str | list
      str: String representing the path to extract field in `obj`.
      list: List of steps.
    envp: EvnParser to execute the conditions and the aggragation.
      EvnParser with default arguments will be used as default.
    dtype: str | AUTO | None
      AUTO: Call `regex_caster` to cast string to any proper dtype.
      str: Cast string to dtype defined in `stype_spec` by call `str_caster`.
        INT:
        FLOAT:
        DATE:
        DATETIME:
    extended: Import numpy or some other modules for convinience or
      representation string compatiability for dtype conversion.
    dfill: The default value after dtype casting fails.
      This will override the default values for the dtype if not None.
    dforced: If to rollback to used `dfill` if caster in `stype_spec` fails.

    Return:
    ----------------
    Any
    """
    envp = EnvParser() if envp is None else envp
    steps = steps.split(":") if isinstance(steps, str) else steps
    if isinstance(obj, str):
        try:
            cur_obj = json.loads(obj)
        # In case the `obj` is merely a string with no JSON structure.
        except json.JSONDecodeError:
            cur_obj = obj
    else:
        cur_obj = obj

    for idx, step in enumerate(steps):
        # Stop early.
        # `[]` or `{}` shouldn't stop early to keep behavior consistent while
        #   aggregating.
        if cur_obj is None:
            break

        if step[0] in "[{" and step[-1] in "]}":
            rets = []
            # Iterate over the values of `cur_obj`.
            if step[0] == "{":
                cur_obj = cur_obj.values()
            for obj_ in cur_obj:
                ret = extract_field(obj_, steps[idx + 1:], envp, dtype,
                                    dforced=dforced, dfill=dfill)
                rets.append(ret)
            agg_expr = step[1:-1]
            if agg_expr:
                rets = envp.bind_env(rets).parse(agg_expr)
            return rets
        else:
            # Split target and conditions.
            # 1. `*conds` ensures that it's always be successful to match the
            #   result of `steps.split`. And `conds[0]` will be the conditions
            #   if conditions exists.
            dest, *conds = step.split("&&")
            if conds and not envp.bind_env(cur_obj).parse(conds[0]):
                return None
            cur_obj = (cur_obj.get(dest, None) if isinstance(cur_obj, dict)
                       else None)

    # Try type casting iff `dtype` is provided.
    if dtype:
        # Call `str_caster` to cast dtype from str.
        if isinstance(cur_obj, str):
            try:
                cur_obj = str_caster(cur_obj,
                                     dtype=dtype,
                                     extended=extended,
                                     dfill=dfill,
                                     dforced=dforced)
            except ValueError:
                logger.warning(f"Can't cast to target dtype {dtype}.")
        # Set with default value if dtype is specified and dtype unfication
        # is forced.
        elif cur_obj is None and dforced:
            cur_obj = stype_spec(dtype, "default") if dfill is None else dfill

    return cur_obj


# %%
def reset_field(
    obj: dict | str,
    steps: str,
    val: Any,
    envp: EnvParser | None = None,
) -> Any:
    """Extract field from dict.

    Reset field's in `obj` indicated by the `steps`. String that can be
    converted to dict is also supported, but only JSON string currently.
    1. If no proper field found in `obj`, nothing will be changed.
    2. `steps` will use `:` as seperator for each level.
    3. This will be called recursively to for each step in steps.

    Example:
    ----------------
    obj = {
      "a": {
        "c": {
          "d": [1, 2, 3],
        },
        "d": 1,
      },
    }
    steps = "a:c&&d=1:c"  -> list
    steps = "a:c&&d=2:c"  -> None

    Params:
    ----------------
    obj: dict | str
      dict: Dict where fields will be found.
      str: JSON string, which dict will loaded from.
    steps: str | list
      str: String representing the path to extract field in `obj`.
      list: List of steps.
    val: Target value.
    envp: EvnParser to execute the conditions and the aggragation.
      EvnParser with default arguments will be used as default.

    Return:
    ----------------
    Modifed `obj`.
    """
    envp = EnvParser() if envp is None else envp
    steps = steps.split(":") if isinstance(steps, str) else steps
    if isinstance(obj, str):
        try:
            cur_obj = json.loads(obj)
        # In case the `obj` is merely a string with no JSON structure.
        except json.JSONDecodeError:
            cur_obj = obj
    else:
        cur_obj = obj

    for idx, step in enumerate(steps[:-1]):
        # Stop early.
        # `[]` or `{}` shouldn't stop early to keep behavior consistent while
        #   aggregating.
        if cur_obj is None:
            break

        if step[0] in "[{" and step[-1] in "]}":
            # Iterate over the values of `cur_obj`.
            if step[0] == "{":
                cur_obj = cur_obj.values()
            for obj_ in cur_obj:
                reset_field(obj_, steps[idx + 1:], val, envp)
        else:
            # Split target and conditions.
            # 1. `*conds` ensures that it's always be successful to match the
            #   result of `steps.split`. And `conds[0]` will be the conditions
            #   if conditions exists.
            dest, *conds = step.split("&&")
            if conds and not envp.bind_env(cur_obj).parse(conds[0]):
                return obj
            cur_obj = (cur_obj.get(dest, None) if isinstance(cur_obj, dict)
                       else None)
    else:
        # Set value.
        if isinstance(cur_obj, dict) and steps[-1] in cur_obj:
            if callable(val):
                cur_obj[steps[-1]] = val()
            else:
                cur_obj[steps[-1]] = val

    return obj


# %%
def rebuild_dict(
    obj: dict | str,
    rules: list,
    extended: bool = False,
    envp: EnvParser | None = None,
) -> dict:
    """Rebuild dict after extracting fields.

    Rebuild `obj` by calling `extract_field` for each item in `rules`.

    Params:
    ----------------
    obj: Dict from which to get the value according to the rules.
      dict: Dict where fields will be found.
      str: JSON string, from which dict will loaded.
    rules: [(key, from_, steps, dtype), ...]
      2-Tuple: [key, steps]
      3-Tuple: [key, steps, dtype]
      4-Tuple: [key, from_, steps, dtype]
      5-Tuple: [key, from_, steps, dtype, default]
      6-Tuple: [key, from_, steps, dtype, forced, default]
        key: Key in the new dict.
        from_: Dependency and source from which get the value and will be
          passed to `extract_field` as `obj.`
        steps: Steps passed to `extract_field` as `steps`.
        dtype: Dtype passed to `extract_field` as `dtype`.
        default: Default value passed to `extract_field` as `dfill`.
        forced: Forced-dtype conversion flag passed to `extract_field` as
          dforced.
    extended: Import numpy or some other modules for convinience or
      representation string compatiability for dtype conversion.
    envp: EvnParser to execute the conditions and the aggragation.
      EvnParser with default arguments will be used as default.

    Return:
    ----------------
    rets: dict
      Dict with new structure.
    """
    envp = EnvParser() if envp is None else envp
    rets = {}
    rule_Q = deque(rules)
    for rule in rules:
        if len(rule) == 2:
            key, steps = rule
            from_, dtype = None, None
            dforced, dfill = False, None
        elif len(rule) == 3:
            key, steps, dtype = rule
            from_ = None
            dforced, dfill = False, None
        elif len(rule) == 4:
            key, from_, steps, dtype = rule
            dforced, dfill = False, None
        elif len(rule) == 5:
            key, from_, steps, dtype, dfill = rule
            dforced = True
        elif len(rule) == 6:
            key, from_, steps, dtype, dfill, dforced = rule
        else:
            continue

        # Set `obj` as default data source.
        if from_ is None:
            cur_obj = obj
        else:
            # Append current item to the rear of the queue.
            # NOTE: This may time comsuming if few items in rules exchange
            # their positions and the rest of rules depend on them.
            if from_ not in rets:
                rule_Q.append((key, from_, steps, dtype))
                continue
            else:
                cur_obj = rets[from_]

        # Extract fields.
        rets[key] = extract_field(cur_obj, steps, envp, dtype, extended,
                                  dfill=dfill,
                                  dforced=dforced)

    return rets
