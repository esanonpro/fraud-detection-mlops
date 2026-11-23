import json

import numpy as np
import pytest

from evaluation.gate import run_gate
from evaluation.metrics import compute_metrics


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.6, 0.9, 0.4])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m.recall == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.5)
    assert m.auc_roc == pytest.approx(0.75)


def write(path, recall):
    path.write_text(json.dumps({"recall": recall, "precision": 0.1, "auc_roc": 0.9}))
    return str(path)


def test_gate_accepts_equal_or_better(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_gate(write(tmp_path / "c.json", 0.90), write(tmp_path / "b.json", 0.87))
    assert e.value.code == 0


def test_gate_rejects_lower_recall(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_gate(write(tmp_path / "c.json", 0.80), write(tmp_path / "b.json", 0.87))
    assert e.value.code == 1


def test_gate_accepts_when_no_baseline(tmp_path):
    with pytest.raises(SystemExit) as e:
        run_gate(write(tmp_path / "c.json", 0.80), str(tmp_path / "absent.json"))
    assert e.value.code == 0
