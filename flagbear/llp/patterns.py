#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: patterns.py
#   Author: xyy15926
#   Created: 2023-12-03 21:05:51
#   Updated: 2024-10-21 20:29:42
#   Description:
# ---------------------------------------------------------

# %%
from datetime import date, time
import numpy as np

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


# %%
LEX_ENDFLAG = "$END"

LEX_RESERVEDS = {
    "if"            : "IF",
    "else"          : "ELSE",
    "or"            : "OR",
    "and"           : "AND",
    "not"           : "NOT",
    "endif"         : "ENDIF",
    "in"            : "IN",
}

LEX_SKIPS = {"NL", "SEMI", "BLANK"}

# `dict` is ordered after Py36. So following token specifications could be
# configured with `dict` instead of list or tuple.
# ATTENTION: Unsigned instead of signed int and float pattern are used for
#   default Lexer token specifications.
# ATTENTION: `\w` in `STRING` doesn't contains chars such as `.`, `*`, etc.
LEX_TOKEN_SPECS = {
    "STRING"        : (r"\"\w*\""                       , lambda x: x[1:-1]),
    "ID"            : r"[A-Za-z_]+[A-Za-z_0-9]*",
    "FLOAT"         : (REGEXS["float"]                  , float),
    "INT"           : (REGEXS["int"]                    , int),
    "ADD"           : r"\+",
    "SUB"           : r"-",
    "MUL"           : r"\*",
    "DIV"           : r"/",
    "EQ"            : r"==",
    "NEQ"           : r"!=",
    "GE"            : r"\>=",
    "GT"            : r"\>",
    "LE"            : r"\<=",
    "LT"            : r"\<",
    "ASG"           : r"=",
    "BAND"          : r"&",
    "DAND"          : r"&&",
    "BOR"           : r"\|",
    "DOR"           : r"\|\|",
    "NOT"           : r"\!",
    "AT"            : r"@",
    "LPAR"          : r"\(",
    "RPAR"          : r"\)",
    "LSPAR"         : r"\[",
    "RSPAR"         : r"\]",
    "LBPAR"         : r"\{",
    "RBPAR"         : r"\}",
    "DOLLAR"        : r"\$",
    "DDOLLAR"       : r"\$\$",
    "COMMA"         : r",",
    "DOT"           : r"\.",
    "NL"            : r"\n+",
    "SEMI"          : r";",
    "BLANK"         : r"[ \t]+",
}

LEX_TOKEN_PRECS = {
    "NOT"       : (1    , 1     , lambda x: not bool(x)),
    "AT"        : (1    , 2     , lambda x, y: x in y),
    "BAND"      : (1    , 2     , lambda x, y: x and y),
    "BOR"       : (1    , 2     , lambda x, y: x or y),
    "EQ"        : (2    , 2     , lambda x, y: x == y),
    "NEQ"       : (2    , 2     , lambda x, y: x != y),
    "LT"        : (2    , 2     , lambda x, y: x < y),
    "LE"        : (2    , 2     , lambda x, y: x <= y),
    "GT"        : (2    , 2     , lambda x, y: x > y),
    "GE"        : (2    , 2     , lambda x, y: x >= y),
    "ADD"       : (3    , 2     , lambda x, y: x + y),
    "SUB"       : (3    , 2     , lambda x, y: x - y),
    "MUL"       : (4    , 2     , lambda x, y: x * y),
    "DIV"       : (4    , 2     , lambda x, y: x / y),
    "LPAR"      : (-999 , 0     , None),
    "RPAR"      : (999  , 0     , None),
}

# %%
SYN_STARTSYM = "S"

SYN_ARITH_PRODS = [
    ("S"        , ("expr", )                    , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("FLOAT", )                   , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("SUB", "FLOAT")              , lambda x: -x[1]           , 3     , "R"),
    ("expr"     , ("INT", )                     , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("SUB", "INT")                , lambda x: -x[1]           , 3     , "R"),
    ("expr"     , ("expr", "ADD", "expr")       , lambda x: x[0] + x[2]     , 1     , "L"),
    ("expr"     , ("expr", "SUB", "expr")       , lambda x: x[0] - x[2]     , 1     , "L"),
    ("expr"     , ("expr", "MUL", "expr")       , lambda x: x[0] * x[2]     , 2     , "L"),
    ("expr"     , ("expr", "DIV", "expr")       , lambda x: x[0] / x[2]     , 2     , "L"),
    ("expr"     , ("LPAR", "expr", "RPAR")      , lambda x: x[1]            , 0     , "R"),
]

# %%
# TODO
SYN_EXPR_PRODS = [
    ("S"        , ("expr", )                    , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("ID", )                      , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("FLOAT", )                   , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("INT", )                     , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("STRING", )                  , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("SUB", "expr")               , lambda x: -x[1]                   , 8     , "R"),
    ("expr"     , ("expr", "ADD", "expr")       , lambda x: x[0] + x[2]             , 3     , "L"),
    ("expr"     , ("expr", "SUB", "expr")       , lambda x: x[0] - x[2]             , 3     , "L"),
    ("expr"     , ("expr", "MUL", "expr")       , lambda x: x[0] * x[2]             , 4     , "L"),
    ("expr"     , ("expr", "DIV", "expr")       , lambda x: x[0] / x[2]             , 4     , "L"),
    ("expr"     , ("LPAR", "expr", "RPAR")      , lambda x: x[1]                    , 0     , "R"),
    ("eles"     , ()                            , lambda x: []                      , 0     , "L"),
    ("eles"     , ("expr", )                    , lambda x: [x[0]]                  , 0     , "L"),
    ("eles"     , ("expr", "COMMA", "expr")     , lambda x: [x[0], x[2]]            , 0     , "L"),
    ("eles"     , ("expr", "COMMA", "eles")     , lambda x: [x[0], *x[2]]           , 0     , "L"),
    ("expr"     , ("LBPAR", "eles", "RBPAR")    , lambda x: frozenset(x[1])         , 0     , "L"),
    ("expr"     , ("LSPAR", "eles", "RSPAR")    , lambda x: x[1]                    , 0     , "L"),
    ("expr"     , ("expr", "EQ", "expr")        , lambda x: x[0] == x[2]            , 2     , "L"),
    ("expr"     , ("expr", "NEQ", "expr")       , lambda x: x[0] != x[2]            , 2     , "L"),
    ("expr"     , ("expr", "LT", "expr")        , lambda x: x[0] < x[2]             , 2     , "L"),
    ("expr"     , ("expr", "GT", "expr")        , lambda x: x[0] > x[2]             , 2     , "L"),
    ("expr"     , ("expr", "LE", "expr")        , lambda x: x[0] <= x[2]            , 2     , "L"),
    ("expr"     , ("expr", "GE", "expr")        , lambda x: x[0] >= x[2]            , 2     , "L"),
    ("expr"     , ("expr", "OR", "expr")        , lambda x: x[0] or x[2]            , 1     , "L"),
    ("expr"     , ("expr", "BOR", "expr")       , lambda x: x[0].__or__(x[2])       , 1     , "L"),
    ("expr"     , ("expr", "AND", "expr")       , lambda x: x[0] and x[2]           , 1     , "L"),
    ("expr"     , ("expr", "BAND", "expr")      , lambda x: x[0].__and__(x[2])      , 1     , "L"),
    ("expr"     , ("NOT", "expr")               , lambda x: not x[1]                , 1     , "L"),
    ("expr"     , ("expr", "IN", "expr")        , lambda x: x[0] in x[2]            , 1     , "L"),
    ("expr"     , ("expr", "LSPAR", "expr", "RSPAR")    , lambda x: x[0][x[2]]      , 9     , "L"),
    ("expr"     , ("expr", "LPAR", "expr", "RPAR")      , lambda x: x[0](x[2])      , 9     , "L"),
    ("expr"     , ("expr", "LPAR", "eles", "RPAR")      , lambda x: x[0](*x[2])     , 9     , "L"),
]

CALLABLE_ENV = {
    "count"             : len,
    "sum"               : sum,
    "max"               : max,
    "min"               : min,
    "nnfilter"          : lambda x: [i for i in x if i is not None],
    "nncount"           : lambda x: len([i for i in x if i is not None]),
}
