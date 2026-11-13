"""Simple validation tests (no Docker or LLM required)."""

import pandas as pd
from agent import validate_cleaned

def test_good_data():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "patient_name": ["A", "B"],
        "age": [30, 40],
        "lab_result": [120, 130],
        "status": ["done", "done"],
        "visit_date": ["2026-01-01", "2026-01-02"],
    })
    ok, reason = validate_cleaned(df, original_row_count=2)
    assert ok, reason

def test_duplicates_rejected():
    df = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "patient_name": ["A", "A"],
        "age": [30, 30],
        "lab_result": [120, 120],
        "status": ["done", "done"],
        "visit_date": ["2026-01-01", "2026-01-01"],
    })
    ok, reason = validate_cleaned(df, original_row_count=2)
    assert not ok

def test_blank_rows_rejected():
    df = pd.DataFrame({
        "patient_id": ["P1", None],
        "patient_name": ["A", None],
        "age": [30, None],
        "lab_result": [120, None],
        "status": ["done", None],
        "visit_date": ["2026-01-01", None],
    })
    ok, _ = validate_cleaned(df, original_row_count=2)
    assert not ok

if __name__ == "__main__":
    test_good_data()
    test_duplicates_rejected()
    test_blank_rows_rejected()
    print("All validation tests passed.")
