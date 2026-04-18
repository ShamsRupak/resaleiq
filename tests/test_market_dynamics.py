"""Unit tests for market dynamics: especially the planted segment logic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from resaleiq.config import IPHONE_LAUNCH_MONTHS, NOISE
from resaleiq.data_generation.market_dynamics import (
    FairValueContext,
    clearing_price_from_fair_value,
    days_since_latest_iphone_launch,
    grade_variance,
    iphone_launch_effect,
    is_iphone_launch_month,
    is_newly_released,
    seasonality_factor,
)


class TestIPhoneLaunchDetection:
    def test_known_launch_months_true(self) -> None:
        for year, month in IPHONE_LAUNCH_MONTHS:
            assert is_iphone_launch_month(date(year, month, 15))

    def test_non_launch_months_false(self) -> None:
        assert not is_iphone_launch_month(date(2025, 3, 15))
        assert not is_iphone_launch_month(date(2025, 7, 4))

    def test_days_since_launch_monotonic(self) -> None:
        d1 = date(2025, 1, 1)
        d2 = date(2025, 6, 1)
        assert days_since_latest_iphone_launch(d2) > days_since_latest_iphone_launch(d1)


class TestSeasonality:
    def test_post_holiday_negative(self) -> None:
        assert seasonality_factor(date(2025, 1, 15), "Apple Mid") < 0

    def test_december_positive(self) -> None:
        assert seasonality_factor(date(2025, 12, 15), "Apple Flagship") > 0

    def test_summer_negative(self) -> None:
        assert seasonality_factor(date(2025, 7, 10), "Android Mid") < 0


class TestGradeVariance:
    def test_grade_d_highest(self) -> None:
        assert grade_variance("D") == max(
            grade_variance(g) for g in NOISE.grade_variance_multiplier
        )

    def test_a_plus_lowest(self) -> None:
        assert grade_variance("A+") == min(
            grade_variance(g) for g in NOISE.grade_variance_multiplier
        )


class TestIPhoneLaunchEffect:
    def test_zero_outside_planted_segment(self) -> None:
        rng = np.random.default_rng(0)
        # Non-Android-Mid category: always zero.
        assert iphone_launch_effect(date(2025, 9, 15), "Apple Flagship", rng) == 0.0
        # Android Mid but non-launch month: zero.
        assert iphone_launch_effect(date(2025, 3, 15), "Android Mid", rng) == 0.0

    def test_nonzero_in_planted_segment(self) -> None:
        rng = np.random.default_rng(42)
        shocks = [iphone_launch_effect(date(2025, 9, 15), "Android Mid", rng) for _ in range(2000)]
        mean_shock = float(np.mean(shocks))
        # Mean should be close to the configured depression (-20%).
        assert mean_shock < -0.17
        assert mean_shock > -0.23


class TestNewlyReleased:
    def test_true_within_window(self) -> None:
        assert is_newly_released(date(2025, 9, 1), date(2025, 9, 20))

    def test_false_outside_window(self) -> None:
        assert not is_newly_released(date(2024, 1, 1), date(2025, 1, 1))

    def test_false_when_future(self) -> None:
        # Device hasn't been released yet.
        assert not is_newly_released(date(2026, 1, 1), date(2025, 6, 1))


class TestClearingPriceFormula:
    @pytest.fixture
    def base_context(self) -> FairValueContext:
        return FairValueContext(
            baseline_value_usd=500.0,
            manufacturer="Apple",
            device_category="Apple Mid",
            release_date=date(2024, 1, 1),
            grade="A",
            transaction_date=date(2025, 4, 15),
        )

    def test_clearing_price_near_baseline(self, base_context: FairValueContext) -> None:
        rng = np.random.default_rng(0)
        prices = np.array([clearing_price_from_fair_value(base_context, rng) for _ in range(500)])
        # The mean clearing price should land within a few percent of baseline.
        assert abs(prices.mean() - 500.0) / 500.0 < 0.05

    def test_clearing_price_positive(self, base_context: FairValueContext) -> None:
        rng = np.random.default_rng(123)
        for _ in range(200):
            price = clearing_price_from_fair_value(base_context, rng)
            assert price > 0

    def test_grade_d_higher_variance_than_a_plus(self) -> None:
        rng = np.random.default_rng(1)
        common_kwargs = dict(
            baseline_value_usd=400.0,
            manufacturer="Apple",
            device_category="Apple Mid",
            release_date=date(2024, 1, 1),
            transaction_date=date(2025, 4, 15),
        )
        a_plus = FairValueContext(grade="A+", **common_kwargs)
        grade_d = FairValueContext(grade="D", **common_kwargs)
        a_prices = np.array([clearing_price_from_fair_value(a_plus, rng) for _ in range(400)])
        d_prices = np.array([clearing_price_from_fair_value(grade_d, rng) for _ in range(400)])
        # Normalized variance should be markedly larger for Grade D.
        a_cv = a_prices.std() / a_prices.mean()
        d_cv = d_prices.std() / d_prices.mean()
        assert d_cv > a_cv * 2
