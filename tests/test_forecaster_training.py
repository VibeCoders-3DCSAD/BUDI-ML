from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from training.scripts.modeling.train_forecaster import validate_wfv_target_coverage


def test_wfv_validation_rejects_missing_target_for_test_period():
    data = pd.DataFrame(
        {
            "year_month": ["2023-12"],
            "target_expenses": [np.nan],
        }
    )
    folds = [{"test_periods": ["2024-01"]}]

    with pytest.raises(ValueError, match="Regenerate them"):
        validate_wfv_target_coverage(data, folds)
