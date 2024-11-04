#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: finer.py
#   Author: xyy15926
#   Created: 2024-06-24 14:04:14
#   Updated: 2024-11-03 21:03:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Iterable
if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as sa
    from IPython.core.debugger import set_trace

import logging
import os
import re
from pathlib import Path
from functools import lru_cache
from datetime import datetime, date


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

# %%
ROOT_FLAG = {".root", ".git", "makefile", ".svn", ".vscode"}
EXCEL_COLUMN_MAX = 10000
DB_COLUMN_MAX = 1000


# %%
@lru_cache
def get_root_path() -> Path:
    """Get the absolute path of the root of the project.
    """
    if __name__ == "__main__":
        path = Path(".").absolute()
    else:
        path = Path(__file__).absolute().parent

    cur = path
    while cur != path.root:
        for ele in cur.iterdir():
            if ele.name in ROOT_FLAG:
                return cur
        cur = path.parent
    else:
        return path


@lru_cache
def get_tmp_path() -> Path:
    """Get the absolute path of the tmp of the project.
    """
    return get_root_path() / "tmp"


@lru_cache
def get_assets_path() -> Path:
    """Get the absolute path of the assets of the project.
    """
    return get_root_path() / "assets"


# %%
def date_order_mark(
    kstr: str,
    alts: Iterable[str] = None,
    rdmark: str | None = "today",
    incr: int = 1,
) -> str:
    """Generate mark string with date and order.

    Params:
    --------------------------
    kstr: Key mark
    alts: Alternative to get the start point of the order.
    rdmark: The date mark for the file with format `%Y%m%d`.
      "today": Today will be used as default.
      None: All proper date string with format `%Y%m%d` will be fine enough,
        and the latest date will be used.
    incr: The increasement for the order mark.
      `1` will be used as default, namely the next proper mark will be
        returned.
      And `0` will return the latest existing mark among the `alts`.

    Return:
    --------------------------
    Next or the largest mark.
    """
    if rdmark == "today":
        today = date.today().isoformat().replace("-", "")
        max_mark = int(today + "0000")
        rdmark = f"({today})"
    elif rdmark is None:
        rdmark = r"(20\d{2}[01]\d[0123]\d)"
        max_mark = int("20000101" + "0000")
    else:
        max_mark = int(rdmark + "0000")
        rdmark = f"({rdmark})"
    idx = r"(\d{4})"
    fnreg = f"{kstr}_{rdmark}_{idx}"

    alts = [] if alts is None else alts
    for alt in alts:
        mret = re.match(fnreg, alt)
        if mret:
            cdate, cid = mret.groups()
            max_mark = max(max_mark, int(cdate + cid))
    max_mark += incr
    fmark = str(max_mark)
    fdate, foid = fmark[:8], fmark[8:]

    return f"{kstr}_{fdate}_{foid}"


# %%
def tmp_file(
    fname: str = "tmp.tmp",
    dmark: str | None = "today",
    incr: int = 1
) -> Path:
    """Generate file absolute path in TMP dir.

    Params:
    ----------------------
    fname: The identical part of the file name.
    dmark: The date mark for the file with format `%Y%m%d`.
      "today": Today will be used as default.
      None: All proper date string with format `%Y%m%d` will be fine enough,
        and the latest date will be used.
    incr: The increasement for the order mark
      `1` will be used as default, namely the next proper filename will be
        returned.
      And `0` will return the latest existing filename.

    Return:
    ----------------------
    Absolute file path with date and order tag `fname_<DATE>_<ORDER>.ext`.
    """
    # Mkdir if necessary.
    tfname = get_tmp_path() / fname
    if not tfname.parent.exists():
        tfname.parent.mkdir()
    elif not tfname.parent.is_dir():
        newfd = tfname.parent.with_name(tfname.parent.name + "_bak")
        newfd.mkdir()
        tfname = newfd / tfname.name

    # Split basename and extname.
    basename, extname = os.path.splitext(tfname.name)
    tbname = date_order_mark(basename,
                             (fd.name for fd in tfname.parent.iterdir()),
                             dmark,
                             incr)
    tfname = tfname.with_name(tbname + extname)

    return tfname


# %%
def tmp_table(
    engine: sa.engine.Engine,
    tblname: str = "tmp",
    dmark: str | None = "today",
    incr: int = 1
) -> str:
    """Generate table name in given database.

    Params:
    ----------------------
    engine: sa.engine.Engine instance.
    tblname: Table name format.

    Return:
    ----------------------
    Table name with format `tblname_<DATE>_<ORDER>`.
    """
    import sqlalchemy as sa

    ftbl = date_order_mark(tblname,
                           sa.inspect(engine).get_table_names(),
                           dmark,
                           incr)
    return ftbl


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
        if df.empty:
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
    import sqlalchemy as sa

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
