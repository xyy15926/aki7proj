#   Name: fliper.py
#   Author: xyy15926
#   Created: 2023-12-07 10:41:26
#   Updated: 2023-12-07 15:17:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Iterator, Callable, Mapping
from collections import deque
from IPython.core.debugger import set_trace

from flagbear.lex import Lexer
from flagbear.parser import EnvParser
from flagbear.patterns import REGEX_TOKEN_SPECS, LEX_ENDFLAG

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def rename_duplicated(ori: list):
    """Rename duplicated elements inorder."""
    duped = False
    cnts = {}
    new = []

    for ele in ori:
        if ele in cnts:
            duped = False
            cnts[ele] += 1
            times = cnts[ele]
            new.append(f"{ele}_{times}")
        else:
            cnts[ele] = 1
            new.append(ele)

    return new if duped else None


# %%
def regex_caster(
    words: str,
    lexer: Lexer | None = None,
    match_ratio: float = 0.8,
) -> tuple[Any, str]:
    """Caster string into possible dtype with regex.

    Params:
    ------------------
    words: Input string.
    lexer: Lexer to parse input string.
      Lexer inited with `REGEX_TOKEN_SPECS` will be used as default.

    Return:
    token.val: Any
    token.type: str
    """
    lexer = (Lexer(REGEX_TOKEN_SPECS, {}, set(), LEX_ENDFLAG)
             if lexer is None
             else lexer)
    words_len = len(words)
    for token in lexer.input(words):
        if token.len / words_len >= match_ratio:
            return token.val, token.type
    return None


# %%
def extract_field(
    obj: dict | str,
    steps: str,
    envp: EnvParser | None = None,
    dtype: str = None,
    *,
    dforced: bool = False,
    dfill: int | float | Mapping | Callable = None,
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

    Attention: `dtype` in `REGEX_TOKEN_SPECS` is necessary. Namely `dfill`
    should be used as some kind of hanlder for any DIY dtype.

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
      str: Casting string to indicating dtype in `REGEX_TOKEN_SPECS`:
        INT:
        FLOAT:
        DATE:
        DATETIME:
    dforced: If to rollback to used `dfill` if converters in
      `REGEX_TOKEN_SPECS` fails.
    dfill: How to convert values after failure.
      Mapping: Map invalid value.
      Callable: Accept invalid value and return result after conversion.

    Return:
    ----------------
    Any
    """
    envp = EnvParser() if envp is None else envp
    steps = steps.split(":") if isinstance(steps, str) else steps
    if isinstance(obj, str):
        cur_obj = json.loads(obj)
    else:
        cur_obj = obj

    for idx, step in enumerate(steps):
        # Stop early.
        # `[]` or `{}` shouldn't stop early to keep behavior consistent while
        #   aggregating.
        if cur_obj is None:
            return None

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
            cur_obj = cur_obj.get(dest, None) if isinstance(cur_obj, dict) else None

    # Try type casting iff `dtype` is provided to cast dtype from `str`.
    if isinstance(cur_obj, str) and dtype is not None:
        if dtype == "AUTO":
            ret = regex_caster(cur_obj)
            if ret is not None:
                cur_obj = ret[0]
        else:
            # `REGEX_TOKEN_SPECS` stores dtype with capital letters.
            if isinstance(dtype, str):
                dtype = dtype.upper()
            convers = REGEX_TOKEN_SPECS.get(dtype)
            if convers is not None:
                try:
                    cur_obj = convers[1](cur_obj)
                    # Reset `deforced` if `convers` succeeds.
                    dforced = False
                except ValueError as e:
                    logger.info(e)
                # Rollback to default value determined by `dfill` if
                # `deforced` is set.
                # Attention: `dtype` in `REGEX_TOKEN_SPECS` is necessary.
                #   Namely `dfill` should be used as some kind of hanlder
                #   for any DIY dtype.
                if dforced:
                    if isinstance(dfill, Callable):
                        cur_obj = dfill(cur_obj)
                    elif isinstance(dfill, Mapping):
                        cur_obj = dfill.get(cur_obj)
                    else:
                        cur_obj = dfill

    return cur_obj


# %%
def rebuild_dict(
    obj: dict | str,
    rules: list,
    envp: EnvParser | None = None,
) -> dict:
    """Rebuild dict after extracting fields.

    Rebuild `obj` by calling `extract_field` for each item in `rules`.

    Params:
    ----------------
    obj: dict | str
      dict: Dict where fields will be found.
      str: JSON string, which dict will loaded from.
    rules: [(key, from_, steps, dtype), ...]
      key: Key in the new dict.
      from_: Dependency and source from which get the value and will be passed
        to `extract_field` as `obj.`
      steps: Steps passed to `extract_field` as `steps`.
      dtype: Dtype passed to `extract_field` as `dtype`.
      default: Default value passed to `extract_field` as `dfill`.
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
        if len(rule) == 4:
            dforced, dfill = False, None
            key, from_, steps, dtype = rule
        else:
            dforced = True
            key, from_, steps, dtype, dfill = rule

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
        rets[key] = extract_field(cur_obj, steps, envp, dtype,
                                  dforced=dforced, dfill=dfill)

    return rets
