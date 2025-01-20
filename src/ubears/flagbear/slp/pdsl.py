#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pdsl.py
#   Author: xyy15926
#   Created: 2024-11-11 10:30:14
#   Updated: 2024-12-10 14:15:21
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Iterable

import logging
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from ubears.flagbear.slp.finer import tmp_file
from ubears.flagbear.slp.db import tmp_table, DB_COLUMN_MAX

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

EXCEL_COLUMN_MAX = 10000


# %%
def jsonfy_df(df: pd.DataFrame) -> pd.DataFrame:
    """JSONfy DB-uncompatiable columns in DataFrame.

    Value in DataFrame may not be compatiable with MYSQL's dtype. So it may
    be necessary to convert `mixed` columns into simple string.

    Params:
    -----------------------
    df: DataFrame with DB-uncompatable columns.

    Returns:
    -----------------------
    DataFrame with some columns JSONfied.
    """
    import pandas as pd
    import json

    dfc = df.copy()
    for coln, colv in dfc.iteritems():
        if pd.api.types.infer_dtype(colv) in ["mixed", "unknown-array"]:
            dfc[coln] = colv.apply(json.dumps)
    return dfc


# %%
def save_with_excel(
    dfs: dict[str, pd.DataFrame] | pd.DataFrame,
    fname: str,
    with_pickle: bool = False,
    hidden: bool = True,
) -> Path:
    """Write dict of DataFrame into Excel.

    The final Excel file will be in TMP dir with date and order tag
    `TMP/fname_<DATE>_<ORDER>.xlsx`.
    And the final pickle file will be in TMP dir with date and the order tag
    `TMP/fname_<DATE>_<ORDER>.pkl`.

    Params:
    ----------------------
    dfs: DataFrame or dict of DataFrames to be saved.
    fname: The identical part of the Excel file name relative the TMP dir.
    with_pickle: If save the DataFrames with pickle for the convenience to
      recover the whole dict later.
    hidden: Skip items with key starting with `_`.

    Return:
    ----------------------
    The final absolute pathlib.Path of the Excel file.
    """
    import pandas as pd
    import pickle

    # Add extname if not provided or not proper.
    tfname = tmp_file(fname)
    if tfname.suffix.lower() not in [".xls", ".xlsx"]:
        tfname = tfname.with_suffix(".xlsx")

    # Prepare the `dfs`.
    if isinstance(dfs, pd.DataFrame):
        dfs = {fname: dfs}

    # Save the DataFrames with Excel.
    xlw = pd.ExcelWriter(tfname)
    for part, df in dfs.items():
        if df.empty or (isinstance(part, str) and part.startswith("_")):
            continue
        elif df.shape[1] <= EXCEL_COLUMN_MAX:
            # In case the sheetname may not be str type.
            df.to_excel(xlw, sheet_name=f"{part}")
        else:
            stop = 0
            while stop * EXCEL_COLUMN_MAX < df.shape[1]:
                start = stop * EXCEL_COLUMN_MAX
                end = (stop + 1) * EXCEL_COLUMN_MAX
                df.iloc[:, start: end].to_excel(
                    xlw, sheet_name=f"{part}_part{stop}")
                stop += 1
    xlw.close()
    logger.info(f"Data saved at {tfname}.")

    # Save the DataFrames with pickle.
    if with_pickle:
        pickle.dump(dfs, open(tfname.with_suffix(".pkl"), "wb"))

    return tfname


# %%
def save_with_pickle(
    inst: Any,
    fname: str,
) -> Path:
    """Save the picklable instance.

    Params:
    ----------------------------
    inst: Any pickable instance.
    fname: The identical part of the file name.

    Return:
    ----------------------------
    The final absolute pathlib.Path of the Excel file.
    """
    import pickle

    # Add extname if not provided or not proper.
    tfname = tmp_file(fname).with_suffix(".pkl")
    pickle.dump(inst, open(tfname, "wb"))

    return tfname


# %%
def load_from_pickle(
    fname: str,
) -> Any:
    """Load from accompanyed pickle file.

    Load from the last version of pickle file with name of format like
     `TMP/fname_<DATE>_<ORDER>.pkl` generated from `tmp_file`.

    Params:
    ----------------------------
    fname: The identical part of the file name.

    Return:
    ----------------------------
    Instance loaded from the pickle.
    """
    import pickle

    tfname = tmp_file(fname, None, 0).with_suffix(".pkl")
    with open(tfname, "rb") as fp:
        ret = pickle.load(fp)

    return ret


# %%
def save_with_db(
    dfs: dict[str, pd.DataFrame] | pd.DataFrame,
    fdb: str | sa.engine.Engine = "tmp.db",
    dtype: dict = None,
) -> sa.engine.Engine:
    """Write dict of DataFrame into Database.

    1. Sqlite will be used if `fdb` is string, and the database file will be
       in TMP dir with date and order tag `TMP/fdb_<DATE>_<ORDER>.db`,
       determined by `tmp_file`.
    2. Or `sa.engine.Engine` instance should be passed to save the DataFrame
       with the table name `<KEY>_<DATE>_<ORDER>`, determined by `tmp_table`.

    Params:
    ----------------------
    dfs: DataFrame or dict of DataFrames to be saved.
    fdb: Sqlite-DB file name or sa.engine.Engine instance.

    Return:
    ----------------------
    sa.engine.Engine of the DB.
    """
    import pandas as pd

    if isinstance(dfs, pd.DataFrame):
        dfs = {fdb: dfs}

    # 1. Unique index, namely Index.unique is True, will be set as key
    #    automatically, which is not compatiable with `TEXT` dtype, which
    #    is the default dtype for object.
    if isinstance(fdb, str):
        dbname = tmp_file(fdb)
        fdb = sa.create_engine(f"sqlite:///{dbname}")
        for part, df in dfs.items():
            df.reset_index().to_sql(name=part,
                                    con=fdb,
                                    index=False,
                                    if_exists="fail")
    else:
        for part, df in dfs.items():
            ftbl = tmp_table(fdb, part)
            df.reset_index().to_sql(name=ftbl,
                                    con=fdb,
                                    index=False,
                                    if_exists="fail")

    logger.info(f"Data saved at {fdb.url} successfully.")
    return fdb
