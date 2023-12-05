#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_rule_engine.py
#   Author: xyy15926
#   Created: 2023-05-09 11:32:59
#   Updated: 2023-07-13 14:46:28
#   Description:
# ---------------------------------------------------------

# %%
import numpy as np
import pandas as pd

# %%
def main(
    vars_: pd.DataFrame,
    rules: pd.DataFrame,
    vals_df: pd.DataFrame
) -> pd.DataFrame:
    nest = build_vars(
        vars_logic_df["var_name"],
        vars_logic_df["var_logic"],
        vals_df)
    rule_logics = rules.apply(
        lambda ser: prepare_rule(ser["var_name"], ser["rule_logic"]),
        axis=1)
    vars_df = build_vars(rules["rule_name"], rule_logics, nest)



