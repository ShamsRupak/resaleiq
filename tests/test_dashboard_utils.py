"""Tests for dashboard shared utilities.

Focuses on the pure functions: formatters and brand constants. The
Postgres/parquet loaders are not tested here because they depend on live
filesystem state; they're exercised end-to-end by running the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so `import dashboard.utils` works from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFormatters:
    """Pure-function formatters. No streamlit dependency needed."""

    def test_fmt_dollars_thousands(self):
        from dashboard.utils import fmt_dollars

        assert fmt_dollars(12345) == "$12K"
        assert fmt_dollars(12345, 1) == "$12.3K"

    def test_fmt_dollars_millions(self):
        from dashboard.utils import fmt_dollars

        assert fmt_dollars(2500000, 1) == "$2.5M"

    def test_fmt_dollars_small(self):
        from dashboard.utils import fmt_dollars

        assert fmt_dollars(42) == "$42"
        assert fmt_dollars(42.567, 2) == "$42.57"

    def test_fmt_dollars_handles_nan(self):
        from dashboard.utils import fmt_dollars

        assert fmt_dollars(float("nan")) == "-"

    def test_fmt_pct(self):
        from dashboard.utils import fmt_pct

        assert fmt_pct(0.1234) == "12.34%"
        assert fmt_pct(0.1234, 1) == "12.3%"
        assert fmt_pct(0.05, 0) == "5%"

    def test_fmt_pct_handles_nan(self):
        from dashboard.utils import fmt_pct

        assert fmt_pct(float("nan")) == "-"

    def test_fmt_int(self):
        from dashboard.utils import fmt_int

        assert fmt_int(1234567) == "1,234,567"

    def test_fmt_int_handles_nan(self):
        from dashboard.utils import fmt_int

        assert fmt_int(float("nan")) == "-"


class TestBrandConstants:
    def test_palette_hex_format(self):
        from dashboard.utils import (
            BRAND_BLUE,
            BRAND_GRAY,
            BRAND_GREEN,
            BRAND_LIGHT,
            BRAND_NAVY,
            BRAND_RED,
        )

        for c in [BRAND_NAVY, BRAND_BLUE, BRAND_LIGHT, BRAND_RED, BRAND_GREEN, BRAND_GRAY]:
            assert c.startswith("#")
            assert len(c) == 7

    def test_model_order_matches_labels(self):
        from dashboard.utils import MODEL_LABELS, MODEL_ORDER

        # Every model in MODEL_ORDER must have a human-readable label
        for m in MODEL_ORDER:
            assert m in MODEL_LABELS
            assert len(MODEL_LABELS[m]) > 0

    def test_segment_colors_complete(self):
        from dashboard.utils import SEGMENT_COLORS

        for category in ["Android Mid", "Apple Flagship", "Apple Mid"]:
            assert category in SEGMENT_COLORS
