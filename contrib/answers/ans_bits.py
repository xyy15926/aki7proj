#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ans_bits.py
#   Author: xyy15926
#   Created: 2024-01-14 16:23:36
#   Updated: 2024-01-15 11:31:59
#   Description:
# ---------------------------------------------------------

# %%
import struct

# %%
# Problem:
# Count 1 in unsigned bits.
def count_one(x: int) -> int:
    """Count `1` in 32-bits unsigned.
    """
    one_cnt = 0
    while x:
        if x & 0x1:
            one_cnt += 1
        x >>= 1
    return one_cnt


def count_one_iter1(x: int) -> int:
    """Count `1` in 32-bits unsigned.

    Only travese `1` in unsigned bits by:
    1. `x &= x - 1`: unset the rightmost 1 in bits.
    """
    one_cnt = 0
    while x:
        x &= x - 1
        one_cnt += 1
    return one_cnt


def count_one_table(x: int) -> int:
    """Count `1` in 32-bits unsigned.

    1. Init table of `1`s in unsigned bits from 0~255.
    2. Shift right to get `1`s for each 8bits group and then add up.
    """
    def init_one_table() -> list:
        one_tbl = [0] * 256
        for i in range(256):
            one_tbl[i] = one_tbl[i >> 1] + (i & 0x1)
        return one_tbl

    one_tbl = init_one_table()
    return (one_tbl[x & 0xff]
            + one_tbl[(x >> 8) & 0xff]
            + one_tbl[(x >> 16) & 0xff]
            + one_tbl[(x >> 24) & 0xff])


def count_one_division(x: int) -> int:
    """Count `1` in 32-bits unsigned.

    1. Add up `1`s for adjacent 1, 2, 4, 8, 16 bits.
    2. Result of `1`s of adjacent group will be stored in each group.
    """
    x = (x & 0x55555555) + ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f)
    x = (x & 0x00ff00ff) + ((x >> 8) & 0x00ff00ff)
    x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff)
    return x


def count_one_division_v2(x: int) -> int:
    """Count `1` in 32-bits unsigned.

    1. Add up `1`s for adjacent groups of 1, 2, 4, 8, 16 bits.
    2. Result of `1`s of adjacent group will be stored in each group.
    """
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3f


# %%
# Problem:
# Check the count of 1 in unsigned bits is odd or even.
def count_one_odd(x: int) -> int:
    """Check if the count of `1` in 32-bits unsigned is odd.

    1. Get the count of `1` first.
    """
    return count_one(x) % 2


def count_one_odd_division(x: int) -> int:
    """Check if the count if `1` in 32-bits unsigned is odd.

    1. Check the oddness for adjacent groups 1, 2, 4, 8, 16 bits.
    """
    x = x ^ (x >> 1)
    x = x ^ (x >> 2)
    x = x ^ (x >> 4)
    x = x ^ (x >> 8)
    x = x ^ (x >> 16)
    return x & 0x01


def count_one_odd_division_tbl(x: int) -> int:
    """Check if the count if `1` in 32-bits unsigned is odd.

    1. Check the oddness for adjacent 1, 2, 4, 8, 16 bits.
    2. Search in table `0x6996` of oddness from 0~0x0f.
    """
    x = x ^ (x >> 4)
    x = x ^ (x >> 8)
    x = x ^ (x >> 16)
    return (0x6996 >> (x & 0x0f)) & 0x01


# %%
# Problem:
# Fill the 1 or 0 in the highest bit to make the count of 1 in unsigned bits
# even or odd.
def even_u8(x: int) -> int:
    one_n = count_one(x)
    if one_n % 2 == 1:
        return x + 0x80
    else:
        return x


def odd_u8(x: int) -> int:
    one_n = count_one(x)
    if one_n % 2 == 1:
        return x
    else:
        return x + 0x80


def even_u8_mod(x: int) -> int:
    """Fill the highest bit to make count of `1` even for 0~127.

    Assuming x's bits is `abcdefg`.
    1. Repeat `x`'s bits 5 times, ignoring overflow bits.
    2. Bitwise and to get `d000 a000 e000 b000 f000 c000 gabc defg`.
    3. Mod with 1920 = 128 * 15 to get [X]abcdfg.
    """
    return (((x * 0x10204081) & 0x888888ff) % 1920) & 0xff


def odd_u8_mod(x: int) -> int:
    """Fill the highest bit to make count of `1` odd for 0~127.

    Assuming x's bits is `abcdefg`.
    1. Repeat `x`'s bits 5 times, ignoring overflow bits.
    2. Bitwise or to get some `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
    3. Mod with 1152 = 128 * 9 to get [X]abcdfg.
    """
    return (((x * 0x00204081) | 0x3db6db00) % 1152) & 0xff


# %%
# Problem:
# Reverse bits for 6bits unsigned.
def reverse_bits(x: int) -> int:
    """Reverse bits for 6bits unsigned.
    """
    return int(''.join(reversed(f"{x:06b}")), base=2)


def reverse_bits_mod(x: int) -> int:
    """Reverse bits for 6bits unsinged.
    """
    return ((x * 0x00082082) & 0x01122408) % 0xff


# %%
# Problem:
# Count the number of prepending 0s for 32bits unsigned.
def count_prepending_zeros(x: int) -> int:
    """Count the number of prepending 0s for 32bits unsigned.
    """
    ans = 0
    while x:
        x >>= 1
        ans += 1
    return 32 - ans


def count_prepending_zeros_mod(x: int) -> int:
    """Count the number of prepending 0s for 32bits unsigned.
    """
    tbl = [
        32, 31, 'u', 16, 'u', 30, 3, 'u', 15, 'u', 'u', 'u', 29,
        10, 2, 'u', 'u', 'u', 12, 14, 21, 'u', 19, 'u', 'u', 28,
        'u', 25, 'u', 9, 1, 'u', 17, 'u', 4, 'u', 'u', 'u', 11,
        'u', 13, 22, 20, 'u', 26, 'u', 'u', 18, 5, 'u', 'u', 23,
        'u', 27, 'u', 6, 'u', 24, 7, 'u', 8, 'u', 0, 'u'
    ]
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return tbl[((x * 0x06eb14f9) & 0xffffffff) >> 26]


# %%
def div3(x: int) -> int:
    return (x * 2863311531) >> 33


# %%
def sqrt(x: float) -> float:
    half = 0.5 * x
    i = struct.unpack("i", struct.pack("f", x))[0]
    i = (0x5f375a86 - (i >> 1)) & 0xffffffff
    x = struct.unpack("f", struct.pack("i", i))[0]
    x = x * (1.5 - half * x * x)
    return x
