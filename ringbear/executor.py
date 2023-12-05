#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: executor.py
#   Author: xyy15926
#   Created: 2023-03-29 18:01:19
#   Updated: 2023-10-14 16:03:17
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
import re
import string
from collections import deque
from collections.abc import Collection, Iterable
from functools import partial
from itertools import chain, groupby
from typing import Any

import numpy as np

from ringbear.dtyper import regex_caster

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# Operator mapper for looking up with format:
# {key: (priority, number of operands, execution), }
OPERATOR_MAPPER = {
    "==": (2, 2, lambda lhs, rhs: lhs == rhs,),
    "<": (2, 2, lambda lhs, rhs: lhs < rhs,),
    ">": (2, 2, lambda lhs, rhs: lhs > rhs,),
    "!=": (2, 2, lambda lhs, rhs: lhs != rhs,),
    "<=": (2, 2, lambda lhs, rhs: lhs <= rhs,),
    ">=": (2, 2, lambda lhs, rhs: lhs >= rhs,),
    # `__or__`, `__and__` and `_isin` is used here to be compatible with
    # ndarray.
    "|": (1, 2, lambda lhs, rhs: lhs.__or__(rhs)),
    "&": (1, 2, lambda lhs, rhs: lhs.__and__(rhs)),
    # It seems that `np.isin` can't handle set well.
    "@": (
        2,
        2,
        lambda lhs, rhs: lhs in rhs
        if np.isscalar(lhs)
        else np.isin(lhs, list(rhs)),
    ),
    "!@": (
        2,
        2,
        lambda lhs, rhs: lhs not in rhs
        if np.isscalar(lhs)
        else np.isin(lhs, list(rhs)),
    ),
    "+": (3, 2, lambda lhs, rhs: lhs + rhs,),
    "-": (3, 2, lambda lhs, rhs: lhs - rhs,),
    "*": (4, 2, lambda lhs, rhs: lhs * rhs,),
    "/": (4, 2, lambda lhs, rhs: lhs / rhs,),
    "!": (5, 1, lambda lhs: not bool(lhs),),
    "!!": (5, 1, lambda lhs: bool(lhs),),
    "$": (5, 1, lambda lhs: regex_caster(lhs),),
    "(": (999, 0, None),
    ")": (-998, 0, None),
}
OPERATOR_PATTERNS = (
    r"\=\=|\<|\>|\!\=|\<\=|\>\=|\||\&|\@|\!\@|" r"\+|\-|\*|\/|\!|\!\!|\$|\(|\)"
)


# %%
def _get_n_safely(val: Collection, n: int | str) -> Any:
    """
    Description:
    Get n-th element in `val`.
    """
    n = int(n)
    if n >= len(val):
        return None
    else:
        # Note that `val` could be any sequence supporting `[]`. And as we
        # want the n-the element, so `.iloc` is called to ensure this.
        if hasattr(val, "iloc"):
            return val.iloc[n]
        else:
            return val[n]


def _concat_if_seq(val: Any) -> str:
    """
    Description:
    Join `val` with `:` if val is a sequence and longer than 2, else original
    `val` will be returned directly.
    """
    # `Sequence` is not used here because `pd.Series` is not Sequence.
    if isinstance(val, Collection) and len(val) > 1:
        # val_str = ["" if repr(i) in ("None", "nan", "<NA>", "NaT")
        #            else repr(i) for i in val]
        return ":".join([repr(i) for i in val])
    elif len(val) == 1:
        return next(iter(val))
    else:
        return val


AGG_MAPPER = {
    "mean": lambda x: sum(x) / len(x),
    "max": max,
    "min": min,
    "list": list,
    "count": len,
    "[]": list,
    "exists": lambda x: np.any([bool(val) for val in x]),
    "first": partial(_get_n_safely, n=0),
    "last": partial(_get_n_safely, n=-1),
    "concat": _concat_if_seq,
}
MATCHED_AGGS = {
    r"\[(\d+)\]": _get_n_safely,
}

# %%
STR_CHECKER = {
    "prefix": lambda x, y: str.startswith(x, y),
    "suffix": lambda x, y: str.endswith(x, y),
    "contains": lambda x, y: y in x,
    "fullmatch": lambda x, y: re.fullmatch(y, x),
    "match": lambda x, y: re.match(y, x),
    "search": lambda x, y: re.search(y, x),
}


# %%
def exec_expstr(expstr: str | list, nest: dict | None = None,) -> Any:
    """
    Description:
    Execute expression represented by operands and operator defined in
    `OPERATOR_MAPPER`.

    Params:
    expstr: expression
    nest: the namespace for searching operands if provided.

    Return:
    """
    # set_trace()
    ops = (
        RPNize(split_expression(expstr))
        if isinstance(expstr, str)
        else deque(expstr)
    )
    # Store operands to be executed.
    opd_st = deque()
    # Pop operands and operators in `ops` to execute.
    while len(ops) > 0:
        op = ops.popleft()
        # If operator popped, pop operands of the number the `op` needs
        # from `opd_st`.
        if op in OPERATOR_MAPPER:
            cur_opts = list(
                reversed([opd_st.pop() for i in range(OPERATOR_MAPPER[op][1])])
            )
            opd_st.append(OPERATOR_MAPPER[op][2](*cur_opts))
        # If operand popped, try to get value from `nest` with operand as
        # the key or keep unchanged.
        else:
            if nest is not None:
                op = nest.get(op, op)
            opd_st.append(op)

    return opd_st[-1]


# %%
def RPNize(ops: list) -> deque:
    """
    Description:
    Transform infix notation into postfix notation, a.k.a. reversed polished
    notations, as RPN.

    Params:
    ops: infix notation

    Return:
    """
    ops = deque(ops)
    store_q, opd_st = deque(), deque("(")
    while ops:
        # set_trace()
        op = ops.popleft()
        if op not in OPERATOR_MAPPER:
            store_q.append(op)
        else:
            in_level = OPERATOR_MAPPER[opd_st[-1]][0]
            # Reverse the priority of `(` in the stack.
            if opd_st[-1] == "(":
                in_level *= -1
            out_level = OPERATOR_MAPPER[op][0]

            # Pop operand in `opd_st` if newly poped `op` is of lower prior
            # than stack-top operand in `opd_st`.
            while in_level > out_level:
                store_q.append(opd_st.pop())
                in_level = OPERATOR_MAPPER[opd_st[-1]][0]
                if opd_st[-1] in "()":
                    in_level *= -1

            if op == ")":
                assert opd_st[-1] == "("
                opd_st.pop()
            else:
                opd_st.append(op)

    while len(opd_st) > 1:
        store_q.append(opd_st.pop())

    return store_q


# %%
def split_expression(expstr: str) -> deque:
    """
    Description:
    Split `expstr` to operators and operands.
    Note that blanks between adjacent operators could be omitted if no more
    than one way to cut adjacent operators. Or expression may not work as
    expected.
    As looking back and some others features is not necessary here, traversal
    is applied here instead of regex.

    Params:
    expstr:

    Return:
    deque:
    """
    # Predefine characters for operator, bracket and identifier.
    operators = "+*<@!>=-/$&|"
    ids = string.ascii_letters + string.digits + "_"
    parentheses = "()"
    curly_brackets = "{}"

    ops_list, pos = deque(), 0
    while pos < len(expstr):
        while expstr[pos] == " ":
            pos += 1
        # Adjacent operators should be seperated with ` ` for precision.
        if expstr[pos] in operators:
            next_stop = pos
            # Keep moving if character is `operators`.
            while next_stop < len(expstr) and expstr[next_stop] in operators:
                next_stop += 1
            # `re.findall` will be called to split operators so that ` ` won't
            # necessary for adjacent operator in some cases. But this may not
            # work as expected as `re.findall` is greedy and elements in
            # `OPERATOR_PATTERNS` may lead to different cutting ways.
            if next_stop - pos > 1:
                ops_list.extend(
                    re.findall(OPERATOR_PATTERNS, expstr[pos:next_stop])
                )
            else:
                ops_list.append(expstr[pos:next_stop])
            pos = next_stop
        # Append parentheses immidiately as they won't be any part of other
        # operators or identifier.
        elif expstr[pos] in parentheses:
            ops_list.append(expstr[pos : pos + 1])
            pos += 1
        # Treat the range surround by curly brackets as whole.
        elif expstr[pos] in curly_brackets:
            next_stop = expstr.find("}", pos) + 1
            ops_list.append(expstr[pos:next_stop])
            pos = next_stop
        # Handle identifier characters.
        elif expstr[pos] in ids:
            next_stop = pos
            while next_stop < len(expstr) and expstr[next_stop] in ids:
                next_stop += 1
            # Check if `-` is negative flag.
            if (
                len(ops_list) > 0
                and ops_list[-1] == "—"
                and (len(ops_list) == 1 or ops_list[-2] in OPERATOR_MAPPER)
            ):
                ops_list[-1] = expstr[pos - 1 : next_stop]
            else:
                ops_list.append(expstr[pos:next_stop])
            pos = next_stop
        else:
            logger.warning(
                "Invalid character %s in expression %s.", expstr[pos], expstr
            )
            pos += 1
    return ops_list


# %%
def exec_aggstr(
    agg_str: str, values: list, skipna: bool | None = None,
) -> Any:
    """
    Description:
    Apply aggregation specified by `agg_str` to `values`.

    Params:
    agg_str:
    values:
    skipna: skip None in aggregation.
        `None`: skip `None` according to `agg_str`

    Return:
    """
    # If returning list directly, maybe keeping None is better to keep equal
    # length, for `zipflat` in `flat_dict` for example.
    if skipna is None:
        skipna = agg_str not in ["[]", "list"]
    if skipna:
        values = [i for i in values if i is not None]

    if agg_str in AGG_MAPPER:
        try:
            ret = AGG_MAPPER[agg_str](values)
            return ret
        except Exception as e:
            logger.debug(
                "Unsupported values, %s, for aggregation, %s.",
                values,
                agg_str,
                e,
            )
            return None
    else:
        for ptn, handler in MATCHED_AGGS.items():
            matched = re.match(ptn, agg_str)
            if matched:
                return handler(values, *matched.groups())
    return None


# %%
def aggregate_with_key(
    seqs: Iterable,
    key: int | tuple | list = 0,
    agg_how: list | tuple | str | callable = "list",
) -> list:
    """
    Description:
    Aggragate elements in sequences in `seqs`, with `key` specifying group key
    and `agg_how` specifying how to aggregate.

    Params:
    seqs: Iterable[[[SAME_LENGTH], ], ] or Iterable[[SAME_LENGTH], ]
    key: specifying key with positions
    agg_how: how to aggregate fields in [SAME_LENGTH],

    Return:
    aggregated
    """
    # Flat `seqs` with `chain` if necessary
    seq = next(iter(next(iter(seqs))))
    if isinstance(seq, Collection):
        iters, length = chain.from_iterable(seqs), len(seq)
    else:
        iters, length = seqs, len(next(iter(seqs)))

    # Align elements with the same key in sequence in `seqs` together.
    # Then aggregate elements by `how`.
    rets = []
    agg_how = [agg_how,] * length if isinstance(agg_how, str) else agg_how
    for k, v in groupby(
        iters,
        key=lambda x: frozenset([x[i] for i in key])
        if isinstance(key, (tuple, list))
        else x[key],
    ):
        ret = []
        for idx, how, *eles in zip(*[range(length), agg_how, *v]):
            if how == "drop":
                continue
            elif callable(how):
                ret.append(how(eles))
            else:
                ret.append(exec_aggstr(how, eles))
        rets.append(ret)

    return rets
