#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: xgbflow.py
#   Author: xyy15926
#   Created: 2024-07-03 15:50:04
#   Updated: 2024-07-10 10:49:58
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, XGBRanker
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def inner_fit():
    """ Fit XGB in XGB-style.
    """
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    xgtrain = xgb.DMatrix(train_X, train_y)
    xgtest = xgb.DMatrix(test_X, test_y)
    params = {"max_depth": 5,
              "eta": 0.1,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "num_class": len(np.unique(y))}
    watchlist = [(xgtest, "eval"), (xgtrain, "train")]
    epochs = 10

    xgbt = xgb.train(params, xgtrain, epochs, watchlist)

    preds = xgbt.predict(xgtest)
    precision = (preds == test_y).sum() / len(test_y)
    logger.info(f"Precision: {precision * 100:0.2f}%.")


# %%
def skl_fit():
    """ Fit XGB in sklearn-style.
    """
    X, y = load_iris(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    xgbclf = XGBClassifier(n_estimators=10,
                           silent=1,
                           max_depth=4,
                           learning_rate=0.1,
                           subsample=0.7,
                           colsample_bytree=0.7,
                           eval_metric="error")

    xgbclf.fit(train_X, train_y)

    preds = xgbclf.predict(test_X)
    precision = (preds == test_y).sum() / len(test_y)
    logger.info(f"Precision: {precision * 100:0.2f}%.")


