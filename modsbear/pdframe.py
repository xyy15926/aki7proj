#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pdframe.py
#   Author: xyy15926
#   Created: 2024-08-24 16:07:38
#   Updated: 2024-08-25 16:45:54
#   Description:
#     Ref: <https://github.com/jsvine/pdfplumber>
# ---------------------------------------------------------

# %%
from __future__ import annotations
import os
import logging
from typing import Any, TypeVar, Tuple
from collections.abc import Callable, Iterator
from collections import ChainMap
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from IPython.core.debugger import set_trace

import re
import numpy as np
import pandas as pd
import pdfplumber
from flagbear.fliper import str_caster
from suitbear.finer import get_tmp_path, get_assets_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def extract_tables(
    file: str | os.PathLike,
) -> list[pd.DataFrame]:
    """Extract tables from PDF file.

    All tables will be extracted from the PDF file and tables with the same
    number of columns will be concatenated together.

    Params:
    ---------------------
    file: PDF file with tables.

    Return:
    ---------------------
    [DataFrame with RangeIndex as Index and Column].
    """
    table_D = {}
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for table in page.find_tables():
                table_D.setdefault(len(table.columns),
                                   []).append(pd.DataFrame(table.extract()))
    rets = []
    for col_N, tables in table_D.items():
        rets.append(pd.concat(tables).reset_index(drop=True))

    return rets


# %%
def format_table(
    table: pd.DataFrame,
    columns: list | tuple = None,
    drop_chars: str | dict = "\n \t",
    dtypes: dict = None,
) -> (pd.DataFrame, pd.DataFrame):
    """Format the table extracted from PDF file.

    Params:
    ------------------------------
    table: Dataframe extracted by `extract_tables` with
      1. RangeIndex as Index and Column.
      2. String as the elements' dtype.
    columns: Column names.
      The original RangeIndex will be kept if not provided.
    drop_chars: Characters to be removed in table elements.
    dtypes: {Column-Name: Dtype-Param} Specify the dtype of each columns.
        Dtype-Params: Dtype specified in `pattern.REGEX_TOKEN_SPECS`.
        Dtype-Params: Tuple that will be unpacked and passed to `str_caster`
          directly.

    Return:
    ------------------------------
    desc: DataFrame with description of the table.
    table: DataFrame of the main content.
    """
    desc = None
    # Find the rows the head of the column.
    if columns is not None:
        col_head_idxs = []
        columns = tuple(columns)
        for idx, row in enumerate(table.iloc[:3].itertuples(index=False)):
            if row == columns:
                col_head_idxs.append(idx)
        # Reset the head the of the columns if ColumnHeadRows exists.
        # 1. Reserve only the first ColumnHeadRow and remove the rest, which
        #   is necessary for the tables across multiple pages.
        # 2. Remove the rows before the first ColumnHeadRow so to skip the
        #   description of the table.
        if len(col_head_idxs) == 0:
            logger.warning("Can't find the head of the columns in the table.")
        else:
            _first, *_rest = col_head_idxs
            desc = table.iloc[:_first].copy()
            table = (table.drop(_rest + list(range(_first + 1)), axis=0)
                     .reset_index(drop=True))
            table.columns = columns

    # Remove invalid chars for display only.
    if drop_chars is not None:
        ptn = re.compile("|".join(list(drop_chars)))
        if isinstance(drop_chars, str):
            table = table.applymap(lambda x: re.sub(ptn, "", x)
                                   if isinstance(x, str) else x)

    # Apply dtype transformation.
    if dtypes is not None:
        for col, _dtype in dtypes.items():
            if np.isscalar(_dtype):
                table[col] = table[col].apply(lambda x: str_caster(x, _dtype))
            else:
                table[col] = table[col].apply(lambda x: str_caster(x, *_dtype))

    return table, desc






