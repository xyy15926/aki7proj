#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: tag_analysis.py
#   Author: xyy15926
#   Created: 2023-10-07 14:46:51
#   Updated: 2023-10-12 14:36:02
#   Description:
# ---------------------------------------------------------

import logging

# %%
import numpy as np
import pandas as pd

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def pivot_tags(tags: pd.Series, seps: str = ",",) -> pd.DataFrame:
    """
    Description:
    Split values in `tags` with `seps`, and then count tag frequncies for each
    tag in each record.

    Params:
    tags:

    Return:
    P
    """
    tags = tags.fillna("").astype(str).str.strip(seps).str.split(seps, expand=False)
    tag_counts = (
        pd.DataFrame(
            {
                "id": tags.index.repeat(tags.apply(len)),
                "tags": np.concatenate(tags.values),
                "ones": np.ones(np.add.reduce(tags.apply(len)), dtype=np.int_),
            }
        )
        .replace("", "NULL")
        .groupby(["id", "tags"])["ones"]
        .agg(sum)
        .unstack()
        .fillna(0)
        .astype(np.int_)
    )
    return tag_counts
