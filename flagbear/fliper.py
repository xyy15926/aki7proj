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
from collections.abc import Iterable, Callable, Mapping
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


# %%
def regex_caster(
    words: str,
    lexer: Lexer | None = None,
    match_ratio: float = 0.8,
) -> tuple[Any, str]:
    """Cast string into possible dtype with regex.

    Params:
    ------------------
    words: Input string.
    lexer: Lexer to parse input string.
      Lexer inited with `REGEX_TOKEN_SPECS` will be used as default.

    Return:
    ------------------
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
def str_caster(
    words: str,
    dtype: str = None,
    dforced: bool = False,
    dfill: Any = None,
    regex_specs: Mapping = REGEX_TOKEN_SPECS,
) -> Any:
    """Cast string into other dtype.

    Params:
    ---------------------
    words: String to be casted to other dtype.
    dtype: str | AUTO | None
      AUTO: Call `regex_caster` to cast string to any proper dtype.
      str: Casting string to indicating dtype in `REGEX_TOKEN_SPECS`(default)
        INT:
        FLOAT:
        TIME:
        DATE:
    dforced: If to rollback to used `dfill` if converters in
      `REGEX_TOKEN_SPECS` fails.
    dfill: The default value after dtype casting fails.
      This will override the default values in `regex_specs` if not None.
    regex_specs: Mapping[dtype, (regex, convert-function, default,...)]
      Mapping storing the dtype name and the handler.

    Return:
    ---------------------
    Any
    """
    ret = words
    if dtype is None or dtype == "AUTO":
        ret = regex_caster(ret)
        if ret is not None:
            ret = ret[0]
    else:
        # `REGEX_TOKEN_SPECS` stores dtype with capital letters.
        dtype = dtype.upper()
        convers = regex_specs.get(dtype)
        if convers is not None:
            try:
                ret = convers[1](ret)
            except Exception as e:
                logger.info(e)
                if dforced:
                    if dfill is None:
                        ret = convers[2]
                    else:
                        ret = dfill
        else:
            raise ValueError(f"Unrecognized dtype {dtype}.")

    return ret


# %%
def extract_field(
    obj: dict | str,
    steps: str,
    envp: EnvParser | None = None,
    dtype: str = None,
    *,
    dforced: bool = False,
    dfill: Any = None,
    regex_specs: Mapping = REGEX_TOKEN_SPECS,
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
      str: Casting string to indicating dtype in `REGEX_TOKEN_SPECS`(default)
        INT:
        FLOAT:
        TIME:
        DATE:
    dforced: If to rollback to used `dfill` if converters in
      `REGEX_TOKEN_SPECS` fails.
    dfill: The default value after dtype casting fails.
      This will override the default values in `regex_specs` if not None.
    regex_specs: Mapping[dtype, (regex, convert-function, default,...)]
      Mapping storing the dtype name and the handler.

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
                                    dforced=dforced, dfill=dfill,
                                    regex_specs=regex_specs)
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
    if isinstance(cur_obj, str) and (dtype == "AUTO" or dtype in regex_specs):
        cur_obj = str_caster(cur_obj, dtype=dtype, dforced=dforced,
                             dfill=dfill, regex_specs=regex_specs)

    return cur_obj


# %%
def rebuild_dict(
    obj: dict | str,
    rules: list,
    envp: EnvParser | None = None,
    regex_specs: Mapping = REGEX_TOKEN_SPECS,
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
      4-Tuple: [key, from_, steps, dtype]
      5-Tuple: [key, from_, steps, dtype, default]
        key: Key in the new dict.
        from_: Dependency and source from which get the value and will be
          passed to `extract_field` as `obj.`
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
        rets[key] = extract_field(cur_obj, steps, envp, dtype,
                                  dforced=dforced, dfill=dfill,
                                  regex_specs=regex_specs)

    return rets
