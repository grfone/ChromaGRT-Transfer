"""Express a dataset in another dataset's condition vocabulary.

ChromaGRT's condition vector is the table's condition columns, taken in order
from the first ``column.usp.code_*`` column to the end of the table. Its width
and its order are therefore properties of the dataset, not of the model, and
two datasets built from the same metadata schema do not necessarily share
them: the RP subset of RepoRT uses six USP column codes and the HILIC subset
uses four different ones.

A checkpoint trained on one of them can only be loaded into the other if both
present the same condition columns in the same order. This module performs
that projection: the target dataset is expressed in the reference dataset's
condition space, so a condition the reference does not know becomes a zero
column instead of a differently named one.
"""

from __future__ import annotations

import pandas as pd

CONDITION_BLOCK_PREFIX = "column.usp.code_"


def condition_columns(df: pd.DataFrame) -> list[str]:
    """Return the condition columns of a curated dataset, in order.

    The condition block starts at the first ``column.usp.code_*`` column and
    runs to the end of the table.
    """
    for index, column in enumerate(df.columns):
        if column.startswith(CONDITION_BLOCK_PREFIX):
            return list(df.columns[index:])
    raise ValueError(f"No condition columns found in table with columns {list(df.columns)[:10]}")


def project_condition_columns(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` expressed in ``reference``'s condition vocabulary.

    Condition columns the target does not have are added as zeros and
    condition columns the target has that the reference does not are dropped.
    The leading, non-condition columns are kept as they are.
    """
    reference_columns = condition_columns(reference)
    target_columns = condition_columns(df)
    target_column_set = set(target_columns)
    leading = [column for column in df.columns if column not in target_column_set]
    projected = df[leading].copy()
    for column in reference_columns:
        if column in target_column_set:
            projected[column] = df[column]
        else:
            projected[column] = 0
    return projected
