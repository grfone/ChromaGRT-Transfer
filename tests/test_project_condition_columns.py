"""Test for the condition-vocabulary projection.

The projection is what makes a checkpoint trained on one dataset load into a
run on another, so its contract is worth pinning down: same condition columns,
same order, same width; columns the target lacks are zero; columns the target
has that the reference does not are dropped.
"""

from __future__ import annotations

import pandas as pd

from src.benchmarks.shared_folds.project_condition_columns import (
    condition_columns,
    project_condition_columns,
)

LEADING = ["cc_id", "molecule_id", "rt", "smiles.std"]
RP_CONDITION = [
    "column.usp.code_L1",
    "column.usp.code_L10",
    "column.usp.code_L109",
    "column.usp.code_L11",
    "column.usp.code_L7",
    "column.usp.code_nan",
    "tanaka_kpb",
    "column.length",
    "eluent.A.h2o",
    "eluent.B.h2o",
    "t [min]_0",
    "B [%]_0",
    "flow rate [ml/min]_0",
]
HILIC_CONDITION = [
    "column.usp.code_L114",
    "column.usp.code_L122",
    "column.usp.code_L3",
    "column.usp.code_L68",
    "tanaka_kpb",
    "column.length",
    "eluent.A.h2o",
    "eluent.B.h2o",
    "t [min]_0",
    "B [%]_0",
    "flow rate [ml/min]_0",
]


def _reference() -> pd.DataFrame:
    return pd.DataFrame(columns=LEADING + RP_CONDITION)


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cc_id": "0001",
                "molecule_id": "0007_1",
                "rt": 12.5,
                "smiles.std": "CCO",
                "column.usp.code_L114": 1.0,
                "column.usp.code_L122": 0.0,
                "column.usp.code_L3": 0.0,
                "column.usp.code_L68": 0.0,
                "tanaka_kpb": 3.1,
                "column.length": 100.0,
                "eluent.A.h2o": 95.0,
                "eluent.B.h2o": 5.0,
                "t [min]_0": 0.0,
                "B [%]_0": 5.0,
                "flow rate [ml/min]_0": 1.0,
            }
        ]
    )


def test_condition_columns_start_at_the_first_usp_column():
    assert condition_columns(_reference()) == RP_CONDITION
    assert condition_columns(_target()) == HILIC_CONDITION


def test_projection_gives_the_reference_vocabulary():
    projected = project_condition_columns(_target(), _reference())
    assert list(projected.columns) == LEADING + RP_CONDITION
    assert list(projected["cc_id"]) == ["0001"]
    assert projected["eluent.A.h2o"].iloc[0] == 95.0
    assert projected["tanaka_kpb"].iloc[0] == 3.1


def test_missing_columns_are_zero_and_extra_columns_are_dropped():
    projected = project_condition_columns(_target(), _reference())
    assert projected["column.usp.code_L1"].iloc[0] == 0
    assert projected["column.usp.code_nan"].iloc[0] == 0
    assert "column.usp.code_L114" not in projected.columns


def test_projection_is_idempotent():
    reference = _reference()
    once = project_condition_columns(reference, reference)
    assert list(once.columns) == list(reference.columns)
