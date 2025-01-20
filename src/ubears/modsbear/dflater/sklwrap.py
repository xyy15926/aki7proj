#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: sklwrap.py
#   Author: xyy15926
#   Created: 2023-01-04 12:36:20
#   Updated: 2023-12-07 08:48:56
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import (FunctionTransformer, )

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %% ------------------------------------------------------------------------
#                   * * * * Transformer Modification * * * *
#
#   Some transformers will be inherited and modified here to fit in the whole
#   process.
# ---------------------------------------------------------------------------
class OneToOneFunctionTransformer(FunctionTransformer):
    """
    Description:
    This class inherits from FunctionTrasnformer with only attributes
    `faeture_names_out` and `n_features_in` add, which endues the class
    the feature `get_feature_names_out`.

    Attention:
    This implement is based on skikit-learn 1.1.2, this may need to be changed
    with the releases.
    """
    def fit(self, X, y=None):
        self.feature_names_out = "one-to-one"
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        return super().fit(X, y)
