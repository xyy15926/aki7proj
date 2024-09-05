#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_stats.py
#   Author: xyy15926
#   Created: 2024-09-04 15:32:10
#   Updated: 2024-09-04 15:41:32
#   Description:
# ---------------------------------------------------------

# %%
import numpy as np
import pytest

if __name__ == "__main__":
    from importlib import reload
    import suitbear
    reload(suitbear)

from suitbear.stats import AHP_get_RCI, AHP_get_RCI_n, AHP_solve


# %%
def test_AHP_get_RCI():
    np.random.seed(7777)
    rci = [AHP_get_RCI_n(n) for n in range(2, 16)]
    RCI = AHP_get_RCI()[1:]
    assert np.all(np.isclose(rci, RCI, 5 * 1e-2))


def test_ahq_solve():
    a1 = np.array([
        [1, 1, 1, 4, 1, 1 / 2],
        [1, 1, 2, 4, 1, 1 / 2],
        [1, 1 / 2, 1, 5, 3, 1 / 2],
        [1 / 4, 1 / 4, 1 / 5, 1, 1 / 3, 1 / 3],
        [1, 1, 1 / 3, 3, 1, 1],
        [2, 2, 2, 3, 3, 1],
    ])

    b1 = np.array([[1, 1 / 4, 1 / 2], [4, 1, 3], [2, 1 / 3, 1]])
    b2 = np.array([[1, 1 / 4, 1 / 5], [4, 1, 1 / 2], [5, 2, 1]])
    b3 = np.array([[1, 3, 1 / 3], [1 / 3, 1, 1 / 7], [3, 7, 1]])
    b4 = np.array([[1, 1 / 3, 5], [3, 1, 7], [1 / 5, 1 / 7, 1]])
    b5 = np.array([[1, 1, 7], [1, 1, 7], [1 / 7, 1 / 7, 1]])
    b6 = np.array([[1, 7, 9], [1 / 7, 1, 1], [1 / 9, 1, 1]])

    hmats = [[a1], [b1, b2, b3, b4, b5, b6]]

    ws = AHP_solve(hmats)
    assert np.all(np.isclose(ws, [0.3952, 0.2996, 0.3052], 1e-3))
