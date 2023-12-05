#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: rule_engine.py
#   Author: xyy15926
#   Created: 2023-05-09 10:27:57
#   Updated: 2023-06-25 10:11:35
#   Description:
# ---------------------------------------------------------

# %%
import re
from collections import deque
import numpy as np
import pandas as pd
from charted.executor import RPNize, exec_expstr, split_expression


# %%
def prepare_rule(
    var_names: str,
    logics: str,
) -> list:
    """
    Description:
    Combine variables and its thresholds together.
    1. Split variables.
    2. Replace some letter identifiers to punctuation.
    3. Combine variables and punctuations together.

    Param:
    var_names: string of variables
    logics: thresholds for variables

    Return:
    list of operands and operators
    """
    # Split variable names.
    var_names = deque(var_names.split())
    # Replace and split.
    logics = logics.replace("true", "== 1") \
        .replace("false", "== 0") \
        .replace("in", "@ ") \
        .replace("not in", "!@ ")
    logics = split_expression(logics)
    # Traverse to find the right position of the variables.
    ops = []
    for op in logics:
        # Only following operators for comparation are considered and used
        # as milestone for inserting variables.
        if op in ["==", "!=", ">", ">=", "<", "<=", "@", "!@"]:
            ops.extend((var_names.popleft(), op))
        elif re.fullmatch(op, r"[\w\{\},]+"):
            ops.extend((op, "$"))
        else:
            ops.append(op)

    return RPNize(ops)


def build_vars(
    var_names: list,
    var_logics: list,
    nest: pd.DataFrame,
) -> pd.DataFrame():
    """
    Desription:
    Build variables specified by `var_logics` from values in `nest`.

    Params:
    var_names: list storing variables names
    var_logics: list storing logics for building variables
    nest: dataframe storing original value for building variables

    Return:
    dataframe
    """
    vars_ = []
    for var_name, logic in zip(var_names, var_logics):
        var_ = exec_expstr(logic, nest)
        var_.name = var_name
        vars_.append(var_)
    vars_df = pd.concat(vars_, axis=1)
    return vars_df

