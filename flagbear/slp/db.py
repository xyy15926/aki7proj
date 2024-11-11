#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: db.py
#   Author: xyy15926
#   Created: 2024-11-11 10:25:29
#   Updated: 2024-11-11 10:30:24
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Iterable

import logging
import sqlalchemy as sa

from flagbear.slp.finer import date_order_mark, tmp_file

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

DB_COLUMN_MAX = 1000


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
