#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ans_bits.py
#   Author: xyy15926
#   Created: 2024-01-14 16:40:31
#   Updated: 2024-01-15 11:33:14
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from answers import ans_bits
    reload(ans_bits)

import math

from answers.ans_bits import (count_one, count_one_iter1,
                              count_one_table, count_one_division,
                              count_one_division_v2)
from answers.ans_bits import (count_one_odd,
                              count_one_odd_division,
                              count_one_odd_division_tbl)
from answers.ans_bits import (even_u8, odd_u8,
                              even_u8_mod, odd_u8_mod)
from answers.ans_bits import (reverse_bits, reverse_bits_mod)
from answers.ans_bits import (count_prepending_zeros,
                              count_prepending_zeros_mod)
from answers.ans_bits import (div3, sqrt)


# %%
def test_count_one():
    x = 1231421431
    ans = count_one(x)
    assert ans == count_one_iter1(x)
    assert ans == count_one_table(x)
    assert ans == count_one_division(x)
    assert ans == count_one_division_v2(x)


def test_cound_one_odd():
    x = 234213516767
    ans = count_one_odd(x)
    assert ans == count_one_odd_division(x)
    assert ans == count_one_odd_division_tbl(x)


def test_even_one():
    for x in range(127):
        even_ans = even_u8(x)
        odd_ans = odd_u8(x)
        assert even_ans == even_u8_mod(x)
        assert odd_ans == odd_u8_mod(x)


def test_reverse_one():
    for x in range(0x3f):
        ans = reverse_bits(x)
        assert ans == reverse_bits_mod(x)


def test_count_prepending_zeros():
    x = 234251783
    ans = count_prepending_zeros(x)
    assert ans == count_prepending_zeros_mod(x)


# %%
def test_div3():
    x = 23415
    ans = x // 3
    assert ans == div3(x)


def test_sqrt():
    x = 341.15
    ans = 1 / math.sqrt(x)
    sqrt(x)
    assert math.isclose(ans, sqrt(x), abs_tol=0.001)

