"""XGBoost training and split-conformal prediction intervals.

Two training entry points:

``train_xgb_model(...)``
    Straightforward XGBoost regressor with early stopping on a held-out
    validation fold.

``compute_conformal_intervals(...)``
    Split-conformal procedure: given a trained model, a calibration set
    held out from both train and test, and a miscoverage level alpha,
    compute symmetric prediction intervals that cover the true clearing
    price with probability (1 - alpha) on exchangeable new data.

Why split-conformal over bootstrap or quantile regression: it's
distribution-free, computationally cheap, and the coverage guarantee holds
under exchangeability alone. For a production pricing system, this is the
standard choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class TrainedModel:
    """Container for a trained XGBoost model plus bookkeeping.

    Trained on the *price ratio* target: ``clearing_price / baseline_value_usd``.

    Why this target. The generator (and any real secondhand-phone marketplace)
    applies pricing effects multiplicatively: a launch-month depression of
    6.5% is a consistent 0.065 shift on the ratio target regardless of whether
    the device costs $100 or $1000. Training on absolute clearing_price buries
    this signal: baseline_value_usd dominates feature importance because it's
    directly correlated with the answer, and XGBoost never gets to the
    multiplicative shift. Training on the ratio surfaces the shift as the
    single biggest predictor XGBoost has access to.

    This is the standard formulation in actuarial pricing (loss ratio),
    retail forecasting (sell-through rate), and any domain where the effect
    of interest is multiplicative. ``predict()`` multiplies back through
    baseline_value_usd to return dollar predictions.
    """

    booster: xgb.Booster
    feature_names: list[str]
    best_iteration: int
    target_type: str = "ratio"  # 'ratio', 'log', or 'absolute'

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Model output in its native target space."""

        dmatrix = xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)
        return np.asarray(
            self.booster.predict(dmatrix, iteration_range=(0, self.best_iteration + 1))
        )

    def predict(
        self, X: pd.DataFrame, baseline_value: np.ndarray | pd.Series | None = None
    ) -> np.ndarray:
        """Dollar-space predictions.

        For ratio-target models, requires ``baseline_value`` to multiply
        through. The caller must pass ``df['baseline_value_usd']`` from the
        row-aligned feature frame.
        """

        raw = self.predict_raw(X)
        if self.target_type == "ratio":
            if baseline_value is None:
                raise ValueError("ratio-target models require baseline_value at predict time")
            return raw * np.asarray(baseline_value, dtype=float)
        if self.target_type == "log":
            return np.exp(raw)
        return raw


@dataclass
class ConformalResult:
    """Conformal interval output.

    ``target_type`` tracks whether the half-width is in ratio space, log
    space, or dollars. For ratio-target models, intervals are multiplicative
    and require baseline_value to materialize in dollar space.
    """

    predictions: np.ndarray  # dollar-space predictions
    half_width: float  # in native target space
    alpha: float
    target_type: str = "absolute"
    baseline_value: np.ndarray | None = None  # for ratio models

    @property
    def lower(self) -> np.ndarray:
        if self.target_type == "ratio":
            assert self.baseline_value is not None
            ratio_pred = self.predictions / self.baseline_value
            return np.maximum((ratio_pred - self.half_width) * self.baseline_value, 0.0)
        if self.target_type == "log":
            return self.predictions * np.exp(-self.half_width)
        return np.maximum(self.predictions - self.half_width, 0.0)

    @property
    def upper(self) -> np.ndarray:
        if self.target_type == "ratio":
            assert self.baseline_value is not None
            ratio_pred = self.predictions / self.baseline_value
            return (ratio_pred + self.half_width) * self.baseline_value
        if self.target_type == "log":
            return self.predictions * np.exp(self.half_width)
        return self.predictions + self.half_width


def _default_params() -> dict[str, object]:
    """Default XGBoost parameters.

    Tuned for ratio-target pricing models where the target has small numerical
    variance (mostly in 0.7 to 1.3 range). With MSE loss and that target
    scale, per-round improvements are tiny in absolute terms and XGBoost's
    default early-stopping behavior can fire prematurely (some versions stop
    at iteration 0). We address that with fixed-round training and a lower
    learning rate that compensates for the additional rounds.
    """

    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 20260420,
        "nthread": 1,  # deterministic across runs; ratio target is fast anyway
    }


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_boost_round: int = 500,
    early_stopping_rounds: int | None = None,
    params: dict[str, object] | None = None,
    target_type: str = "ratio",
    baseline_train: pd.Series | None = None,
    baseline_val: pd.Series | None = None,
) -> TrainedModel:
    """Train an XGBoost regressor on the specified target type.

    Uses fixed-round training by default (``early_stopping_rounds=None``)
    because ratio-target MSE improvements per round are small enough that
    early stopping tolerance is version-dependent. 500 rounds at
    learning_rate=0.03 is a conservative budget that converges cleanly on
    this feature set and guarantees identical behavior across XGBoost
    versions.

    ``target_type``:
        - 'ratio'    (default): target is clearing_price / baseline_value_usd.
                     Requires baseline_train and baseline_val to be passed.
        - 'log'      : target is log(clearing_price).
        - 'absolute' : target is clearing_price directly. Naive baseline.
    """

    params = params or _default_params()
    feature_names = list(X_train.columns)

    if target_type == "ratio":
        if baseline_train is None or baseline_val is None:
            raise ValueError("ratio target requires baseline_train and baseline_val")
        y_train_t = y_train.to_numpy() / baseline_train.to_numpy()
        y_val_t = y_val.to_numpy() / baseline_val.to_numpy()
    elif target_type == "log":
        y_train_t = np.log(y_train.to_numpy())
        y_val_t = np.log(y_val.to_numpy())
    elif target_type == "absolute":
        y_train_t = y_train.to_numpy()
        y_val_t = y_val.to_numpy()
    else:
        raise ValueError(f"unknown target_type: {target_type}")

    dtrain = xgb.DMatrix(X_train, label=y_train_t, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val_t, feature_names=feature_names)

    train_kwargs: dict[str, object] = {
        "num_boost_round": num_boost_round,
        "evals": [(dtrain, "train"), (dval, "val")],
        "verbose_eval": False,
    }
    if early_stopping_rounds is not None:
        train_kwargs["early_stopping_rounds"] = early_stopping_rounds

    booster = xgb.train(params, dtrain, **train_kwargs)

    # When early stopping is off, use the final iteration.
    best_iteration = booster.best_iteration if early_stopping_rounds else num_boost_round - 1
    return TrainedModel(
        booster=booster,
        feature_names=feature_names,
        best_iteration=best_iteration,
        target_type=target_type,
    )


def compute_conformal_intervals(
    model: TrainedModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_pred: pd.DataFrame,
    alpha: float = 0.2,
    baseline_cal: pd.Series | None = None,
    baseline_pred: pd.Series | None = None,
) -> ConformalResult:
    """Split-conformal prediction intervals at (1 - alpha) coverage.

    For ratio-target models, residuals are computed on the ratio target and
    intervals are multiplicative relative to baseline_value in dollar space.
    This produces intervals like "$229 +/- $34" that scale naturally with
    device price, rather than a fixed dollar half-width that would be too
    wide for cheap devices and too narrow for premium ones.
    """

    cal_raw_pred = model.predict_raw(X_cal)
    if model.target_type == "ratio":
        if baseline_cal is None or baseline_pred is None:
            raise ValueError("ratio target conformal requires baseline values")
        cal_actual_ratio = y_cal.to_numpy() / baseline_cal.to_numpy()
        residuals = np.abs(cal_actual_ratio - cal_raw_pred)
    elif model.target_type == "log":
        cal_actual = np.log(y_cal.to_numpy())
        residuals = np.abs(cal_actual - cal_raw_pred)
    else:
        residuals = np.abs(y_cal.to_numpy() - cal_raw_pred)

    n = len(residuals)
    q_index = min(int(np.ceil((1 - alpha) * (n + 1))) - 1, n - 1)
    sorted_residuals = np.sort(residuals)
    half_width = float(sorted_residuals[q_index])

    if model.target_type == "ratio":
        predictions = model.predict(X_pred, baseline_value=baseline_pred)
        return ConformalResult(
            predictions=predictions,
            half_width=half_width,
            alpha=alpha,
            target_type="ratio",
            baseline_value=np.asarray(baseline_pred, dtype=float),
        )
    predictions = model.predict(X_pred)
    return ConformalResult(
        predictions=predictions,
        half_width=half_width,
        alpha=alpha,
        target_type=model.target_type,
    )


def feature_importance(model: TrainedModel, importance_type: str = "gain") -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame.

    ``importance_type``: 'gain', 'weight', 'cover', or 'total_gain'. 'gain'
    is the average improvement in loss when a feature is used for splitting
    and is the standard metric for tree-model feature attribution.
    """

    scores = model.booster.get_score(importance_type=importance_type)
    # xgboost returns only features that were actually used; fill missing.
    records = [
        {"feature": name, "importance": float(scores.get(name, 0.0))}
        for name in model.feature_names
    ]
    df = pd.DataFrame(records).sort_values("importance", ascending=False)
    return df.reset_index(drop=True)


__all__ = [
    "ConformalResult",
    "TrainedModel",
    "compute_conformal_intervals",
    "feature_importance",
    "train_xgb_model",
]
