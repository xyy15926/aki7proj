#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_data_preprocessing.py
#   Author: xyy15926
#   Created: 2023-07-24 15:58:10
#   Updated: 2023-07-24 16:04:20
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd
from ringbear.biclf import (
    lift_ordered,
    woe_ordered,
)


# %%
def test_lift_ordered() -> None:
    a = [1] * 20 + [2] * 20 + [3] * 20
    aa = [3] * 20 + [2] * 20 + [1] * 20
    b = [0] * 60
    b[:10], b[20:23], b[40:42] = [1]*10, [1]*3, [1]*2
    assert np.all(lift_ordered(a, b)[2] == [1, 2, 3])
    assert np.all(lift_ordered(aa, b)[2] == [3, 2, 1])
    assert np.all(lift_ordered(a, b, acc_keys=[2, 3])[2] == [2, 3])
    assert np.all(lift_ordered(a, b, acc_keys=[3, 2])[2] == [2, 3])
    assert np.all(lift_ordered(aa, b, acc_keys=[2, 3])[2] == [3, 2])
    assert np.all(lift_ordered(aa, b, acc_keys=[3, 2])[2] == [3, 2])


