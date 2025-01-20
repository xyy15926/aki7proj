#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: patterns.py
#   Author: xyy15926
#   Created: 2023-12-03 21:05:51
#   Updated: 2025-01-14 20:09:08
#   Description:
# ---------------------------------------------------------

# %%
REGEXS = {
    # 1. r"(?<![\.\d\-/:\[\{\(])" and r"(?![,\.\d\-/:\]\}\)])" may be used
    #   to exclude integer in float, datetime, interval and frozenset.
    "int"           : r"\d+",                               # Integer without sign
    "sint"          : r"[+-]?\d+",                          # Integer with sign
    "mint"          : r"[+-]?"                              # Comma seperated integer
                      r"(?:\d+|\d{1,3}(?:,\d{3})*)",
    "float"         : r"\d*\.\d+",                          # Float without sign
    "sfloat"        : r"[+-]?\d*\.\d+",                     # Float with sign
    "mfloat"        : r"[+-]?"                              # Comma sepertaed float
                      r"(?:\d*|\d{1,3}(?:,\d{3})*)"         # Interger part
                      r"\.\d+",                             # Decimal part
    # 1. It's hard to handle 0229 in leap year, so all 0229 will be recogonized
    #   to be valid.
    # 2. Only `-`, `/` is valid seperator between year, month and day.
    "date"          : r"[0-9]{4}[-/]"                       # yyyy
                      r"(?:"
                      r"(?:0[1-9]|1[0-2])[-/]"              # MM
                      r"(?:0[1-9]|1[0-9]|2[0-9])"           # dd: 1-29
                      r"|"
                      r"(?:0[13-9]|1[0-2])[-/]"             # MM: 1,3,...,12
                      r"30"                                 # dd: 29, 30
                      r"|"
                      r"(?:0[13578]|1[02])[-/]"             # MM: 1,3,5,7,8,10,12
                      r"31"                                 # dd: 31
                      r")",
    # 1. Only 18XX, 19XX, 20XX allowed.
    "idcard"        : r"[1-9]\d{5}"                         # Governmental region code
                      r"(?:18|19|20)\d{2}"                  # yyyy
                      r"(?:"
                      r"(?:0[1-9]|1[0-2])"                  # MM
                      r"(?:0[1-9]|1[0-9]|2[0-9])"           # dd: 1-29
                      r"|"
                      r"(?:0[13-9]|1[0-2])"                 # MM: 1,3,...,12
                      r"30"                                 # dd: 29, 30
                      r"|"
                      r"(?:0[13578]|1[02])"                 # MM: 1,3,5,7,8,10,12
                      r"31"                                 # dd: 31
                      r")"
                      r"\d{3}[0-9Xx]",                      # Check code
    "mobile"        : r"1(?:3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])"
                      r"\d{8}",
    # 1. Only `:`, `/` is valid seperator between hour, minute and second.
    "time"          : r"(?:[01]\d|2[0-3]):"                 # hh
                      r"(?:[0-5]\d):"                       # mm
                      r"(?:[0-5]\d)",                       # ss
    "na"            : r"[nN][aA][nNTt]?",
    # 1. No speced allowed after `[(` or before `)]`.
    "interval"      : r"[\(\[]"                             # Left bracket
                      r"[+-]?\d*(?:\.\d+)?"                 # Left edge
                      r", *"                                # Seperator
                      r"[+-]?\d*(?:\.\d+)?"                 # Right edge
                      r"[\)\]]",                            # Right bracket
    # 1. No speced allowed after `{` or before `}`.
    # 2. Only `1-9A-Za-z._` are allowed to construct element in set.
    "set"           : r"\{"
                      r"(?:[\w\.]+, *)*[\w\.]+"             # Comma seperated elements
                      r"\}",
}
