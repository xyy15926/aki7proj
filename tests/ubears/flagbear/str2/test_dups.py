#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dups.py
#   Author: xyy15926
#   Created: 2024-11-11 11:58:40
#   Updated: 2024-11-11 14:11:49
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.str2 import dups
    reload(dups)

from ubears.flagbear.str2.dups import rename_duplicated, rename_overlaped


# %%
def test_rename_duplicated():
    ll = [1, 2, 3, 3, 2, 4, 5]
    nodup = rename_duplicated(ll)
    assert all([x == y if isinstance(y, int) else 1
                for x, y in zip(ll, nodup)])

    unil = [1, 2, 3, 4, 5]
    nodup = rename_overlaped([unil, unil])
    assert all([f"{x}_2" == y for x, y in zip(*nodup)])

    nodup = rename_overlaped([unil, unil], ["a", "b"])
    assert all([f"{x}_b" == f"{y}" for x, y in zip(*nodup)])
