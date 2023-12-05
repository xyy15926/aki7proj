#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_io.py
#   Author: xyy15926
#   Created: 2022-11-23 16:00:42
#   Updated: 2023-05-05 10:39:01
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from ringbear.dtyper import (
    TYPE_REGEX,
    STR_CASTER,
    min_key,
    max_key,
    tidy_up,
    concat_interval,
    intervals_from_list,
    infer_dtype,
    detect_str_dtype,
    detect_dtype_anyway,
    infer_major_dtype,
    regex_caster,
)
from ringbear.pdler import (
    autotype_ser,
    parse_sort_keys,
    drop_records,
    drop_fields,
    fill_numeric_ser,
)
from ringbear.executor import (
    exec_expstr,
    aggregate_with_key,
)
from ringbear.unifier import (
    extract_field,
)
from ringbear.biclf import (
    lift_ordered,
    woe_ordered,
)
from ringbear.pdchain import (
    sketch_series_basicly,
    sketch_categorical_alone,
    sketch_numeric_alone,
    sketch_categorical_with_label,
    sketch_ordered_with_label,
    sketch_handler,
)
import re
import string
import numpy as np
import pandas as pd
import os
ASSETS = os.path.join(os.path.dirname(__file__), "../assets")


# %%
from importlib import reload
import ringbear.dtyper
import ringbear.pdler
import ringbear.npam
import ringbear.executor
reload(ringbear.dtyper)




def test_sketch() -> None:
    a = [1] * 20 + [2] * 20 + [3] * 20
    b = [0] * 60
    b[:10], b[20:23], b[40:42] = [1]*10, [1]*3, [1]*2
    a, b = pd.Series(a), pd.Series(b)
    assert sketch_series_basicly(a)
    assert sketch_ordered_with_label(a, b)
    assert sketch_categorical_with_label(a, b)
    assert np.all(sketch_handler(a, b).notna())



