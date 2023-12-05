#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: context.py
#   Author: xyy15926
#   Created: 2022-11-23 18:15:56
#   Updated: 2023-04-23 14:08:12
#   Description:
# ---------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# %%
def make_data(length: int = 20, *,
              add_nan: bool = True,
              seed: int = 7777) -> tuple:
    np.random.seed(seed)
    X = pd.DataFrame({
        "cat_1": np.random.choice(list("AEUIO"), length),
        "cat_2": np.random.choice(list("YHGBN"), length),
        "ord_1": np.random.randint(1, 5, length),
        "ord_2": np.random.randint(1, 5, length),
        "num_1": np.random.randint(1, 100, length),
        "num_2": np.random.randint(1, 100, length),
        "num_3": np.random.randint(1, 100, length),
        "date": np.random.choice(
            [f"2023-01-0{D}" for D in range(1, 10)], length),
    })
    y = np.random.choice([0, 1], length)
    if add_nan:
        X.iloc[np.random.choice(range(length), 2), 0] = np.nan
        X.iloc[np.random.choice(range(length), 2), 5] = np.nan
        X.iloc[np.random.choice(range(length), 1)] = np.nan
    return X, y

# %%
