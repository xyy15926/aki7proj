#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: scalar.py
#   Author: xyy15926
#   Created: 2024-01-12 14:21:00
#   Updated: 2024-01-12 15:45:33
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable
import logging

NUM = TypeVar("number")

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def euclid_gcd(x: int, y: int) -> int:
    """Euclid GCD.

    gcd(x, y) = gcd(y, mod(x, y))
    """
    while x != 0 and y != 0:
        tmp = x % y
        x = y
        y = tmp
    return x or y


def euclid_lcm(x: int, y: int) -> int:
    """Euclid LCM.

    lcm(x, y) = x * y / gcd(x, y)
    """
    return x * y / euclid_gcd(x, y)


# %%
def poly_eval(p: list[NUM], x: NUM) -> NUM:
    """Polynomial evaluation.

      p0 + p1 * x + ... + p4 * x^4 + p5 * x^5
    = p0 + p1 * x + ... + (p4 + p5 * x) * x^4
    = ...
    """
    ans = 0
    for i in p[::-1]:
        ans = ans * x + i
    return ans


# %%
def bmultiply(x: int, y: int):
    pass


# %%
def russian_mul(x: int, y: int):
    """Russian mutilply.

    n       m
    ------------------------------------------
    50      65
    25      130         130
    12      260
     6      520
     3      1040        1040
     1      2080        2080
    ------------------------------------------
    sum=130+1040+2080=3205
    """
    ans = 0
    while x > 1:
        if x & 1 == 1:
            ans += y
        x >>= 1
        y <<= 1
    return ans


# %%
def qpower(x: int, n: int):
    """Quick power.

    1. Assuming n = 0b10101
    2. x^n = x^(0b10000 + 0b100 + 0b1)
           = x^(0b10000) * x^(0b100) * x^(0b1)
    3. x^(10000) = x^(0b1000)^2
    Namely, muliply the power result of corresponsible `1` in `n`.
    """
    ans = 1
    while n > 0:
        if n & 1 == 1:
            ans *= x
        x *= x
        n >>= 1
    return ans
