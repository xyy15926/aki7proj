#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: tokens.py
#   Author: xyy15926
#   Created: 2025-01-14 20:07:07
#   Updated: 2025-01-14 20:07:07
#   Description:
# ---------------------------------------------------------

from flagbear.const.patterns import REGEXS

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
