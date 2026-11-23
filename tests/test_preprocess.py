import pandas as pd
import pytest

from data_preprocessing.preprocess import preprocess
from features.build_features import build_features

COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def make_df(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_preprocess_removes_duplicates_and_missing_values():
    row = [0.0] * 30 + [0]
    df = make_df([row, row, [None] * 30 + [1]])
    out = preprocess(df)
    assert len(out) == 1


def test_preprocess_rejects_missing_columns():
    with pytest.raises(ValueError):
        preprocess(pd.DataFrame({"Time": [0], "Amount": [1.0]}))


def test_build_features_splits_target():
    df = make_df([[0.0] * 30 + [0], [1.0] * 30 + [1]])
    X, y = build_features(df)
    assert "Class" not in X.columns
    assert list(y) == [0, 1]
