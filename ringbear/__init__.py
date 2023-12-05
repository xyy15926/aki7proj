#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: __init__.py
#   Author: xyy15926
#   Created: 2023-07-13 15:02:58
#   Updated: 2023-07-24 16:35:20
#   Description:
# ---------------------------------------------------------

from .unifier import (
    extract_field,
    flat_dict,
)

from .dtyper import (
    TYPE_REGEX,
    TYPE_ENUMS,
    STR_CASTER,
    VALID_CASTER,
    STR_TYPE_ORDERED,
    min_key,
    max_key,
    is_overlapped,
    intervals_from_list,
    concat_interval,
    tidy_up,
    infer_dtype,
    detect_str_dtype,
    detect_dtype_anyway,
    infer_major_dtype,
    regex_caster,
    keep_dtype_caster,
)

from .executor import (
    exec_expstr,
    RPNize,
    split_expression,
    exec_aggstr,
    aggregate_with_key,
)

from .npam import (
    get_outliers_with_sigma,
    remove_outlier_with_sigma,
    calculate_criterion
)

from .pdler import (
    autotype_ser,
    cut_numeric_with_outliers,
    filter_outliers,
    fill_numeric_ser,
    build_marker,
    drop_records,
    drop_fields,
)

from .scier import (
    OneToOneFunctionTransformer,
    build_parent_from_children,
    extract_paths_from_tree,
    calculate_criterion_from_tree,
    tree_cut_ordered,
    chi_pairwise,
    chi_pairwise_itl,
    chimerge_cut_ordered,
)

from .pdchain import (
    sketch_handler,
    sketch_categorical_with_label,
    sketch_ordered_with_label,
    sketch_series_basicly,
    sketch_categorical_alone,
    sketch_numeric_alone,
)

from .biclf import (
    lift_ordered,
    woe_ordered,
    iv_ordered,
    select_nodes_with_freqs,
)

