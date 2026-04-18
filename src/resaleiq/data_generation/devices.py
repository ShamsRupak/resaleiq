"""Hand-curated device catalog. Every model, MSRP, and release date comes from
the public record. The catalog is deliberately denormalized: ``device_category``
is stored on the row for convenient segmentation in downstream joins.

The goal is roughly 80 distinct models spanning four model years, weighted toward
Apple (which accounts for the majority of wholesale pre-owned volume) with
representative Samsung, Google, and OnePlus inventory.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from resaleiq.config import DEVICE_CATEGORIES

# Guard that makes refactors explicit if DEVICE_CATEGORIES changes shape.
assert set(DEVICE_CATEGORIES) == {
    "Apple Flagship",
    "Apple Mid",
    "Android Flagship",
    "Android Mid",
    "Android Budget",
}


# (manufacturer, model_family, release_date, msrp_new, device_category)
# Release dates and MSRPs are approximate US launch values. The catalog is
# intentionally bounded to models released before mid-2025 so the 18-month
# window (Oct 2024 to Mar 2026) has meaningful used-market activity.
_CATALOG: list[tuple[str, str, date, float, str]] = [
    # Apple: iPhone 11 family
    ("Apple", "iPhone 11", date(2019, 9, 20), 699.0, "Apple Mid"),
    ("Apple", "iPhone 11 Pro", date(2019, 9, 20), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 11 Pro Max", date(2019, 9, 20), 1099.0, "Apple Flagship"),
    # Apple: iPhone 12 family
    ("Apple", "iPhone 12 mini", date(2020, 11, 13), 699.0, "Apple Mid"),
    ("Apple", "iPhone 12", date(2020, 10, 23), 799.0, "Apple Mid"),
    ("Apple", "iPhone 12 Pro", date(2020, 10, 23), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 12 Pro Max", date(2020, 11, 13), 1099.0, "Apple Flagship"),
    # Apple: iPhone 13 family
    ("Apple", "iPhone 13 mini", date(2021, 9, 24), 699.0, "Apple Mid"),
    ("Apple", "iPhone 13", date(2021, 9, 24), 799.0, "Apple Mid"),
    ("Apple", "iPhone 13 Pro", date(2021, 9, 24), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 13 Pro Max", date(2021, 9, 24), 1099.0, "Apple Flagship"),
    # Apple: iPhone 14 family
    ("Apple", "iPhone 14", date(2022, 9, 16), 799.0, "Apple Mid"),
    ("Apple", "iPhone 14 Plus", date(2022, 10, 7), 899.0, "Apple Mid"),
    ("Apple", "iPhone 14 Pro", date(2022, 9, 16), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 14 Pro Max", date(2022, 9, 16), 1099.0, "Apple Flagship"),
    # Apple: iPhone 15 family
    ("Apple", "iPhone 15", date(2023, 9, 22), 799.0, "Apple Mid"),
    ("Apple", "iPhone 15 Plus", date(2023, 9, 22), 899.0, "Apple Mid"),
    ("Apple", "iPhone 15 Pro", date(2023, 9, 22), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 15 Pro Max", date(2023, 9, 22), 1199.0, "Apple Flagship"),
    # Apple: iPhone 16 family (September 2024 launch: first iPhone launch in window)
    ("Apple", "iPhone 16", date(2024, 9, 20), 799.0, "Apple Mid"),
    ("Apple", "iPhone 16 Plus", date(2024, 9, 20), 899.0, "Apple Mid"),
    ("Apple", "iPhone 16 Pro", date(2024, 9, 20), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 16 Pro Max", date(2024, 9, 20), 1199.0, "Apple Flagship"),
    # Apple: iPhone 17 family (September 2025 launch: second iPhone launch in window)
    ("Apple", "iPhone 17", date(2025, 9, 19), 799.0, "Apple Mid"),
    ("Apple", "iPhone 17 Plus", date(2025, 9, 19), 899.0, "Apple Mid"),
    ("Apple", "iPhone 17 Pro", date(2025, 9, 19), 999.0, "Apple Flagship"),
    ("Apple", "iPhone 17 Pro Max", date(2025, 9, 19), 1199.0, "Apple Flagship"),
    # Apple: iPhone SE line (mid-tier budget)
    ("Apple", "iPhone SE (2022)", date(2022, 3, 18), 429.0, "Apple Mid"),
    # Samsung: Galaxy S21 family
    ("Samsung", "Galaxy S21", date(2021, 1, 29), 799.0, "Android Mid"),
    ("Samsung", "Galaxy S21+", date(2021, 1, 29), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy S21 Ultra", date(2021, 1, 29), 1199.0, "Android Flagship"),
    # Samsung: Galaxy S22 family
    ("Samsung", "Galaxy S22", date(2022, 2, 25), 799.0, "Android Mid"),
    ("Samsung", "Galaxy S22+", date(2022, 2, 25), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy S22 Ultra", date(2022, 2, 25), 1199.0, "Android Flagship"),
    # Samsung: Galaxy S23 family
    ("Samsung", "Galaxy S23", date(2023, 2, 17), 799.0, "Android Mid"),
    ("Samsung", "Galaxy S23+", date(2023, 2, 17), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy S23 Ultra", date(2023, 2, 17), 1199.0, "Android Flagship"),
    # Samsung: Galaxy S24 family
    ("Samsung", "Galaxy S24", date(2024, 1, 31), 799.0, "Android Mid"),
    ("Samsung", "Galaxy S24+", date(2024, 1, 31), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy S24 Ultra", date(2024, 1, 31), 1299.0, "Android Flagship"),
    # Samsung: Galaxy S25 family (Jan 2025)
    ("Samsung", "Galaxy S25", date(2025, 1, 22), 799.0, "Android Mid"),
    ("Samsung", "Galaxy S25+", date(2025, 1, 22), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy S25 Ultra", date(2025, 1, 22), 1299.0, "Android Flagship"),
    # Samsung: Galaxy A series (mid and budget)
    ("Samsung", "Galaxy A14", date(2023, 4, 6), 199.0, "Android Budget"),
    ("Samsung", "Galaxy A24", date(2023, 5, 1), 249.0, "Android Budget"),
    ("Samsung", "Galaxy A34", date(2023, 3, 16), 379.0, "Android Mid"),
    ("Samsung", "Galaxy A54", date(2023, 3, 24), 449.0, "Android Mid"),
    ("Samsung", "Galaxy A15", date(2024, 1, 5), 199.0, "Android Budget"),
    ("Samsung", "Galaxy A25", date(2024, 1, 5), 299.0, "Android Budget"),
    ("Samsung", "Galaxy A35", date(2024, 3, 11), 399.0, "Android Mid"),
    ("Samsung", "Galaxy A55", date(2024, 3, 11), 479.0, "Android Mid"),
    # Samsung: Galaxy Z fold/flip line
    ("Samsung", "Galaxy Z Flip 4", date(2022, 8, 26), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy Z Fold 4", date(2022, 8, 26), 1799.0, "Android Flagship"),
    ("Samsung", "Galaxy Z Flip 5", date(2023, 8, 11), 999.0, "Android Flagship"),
    ("Samsung", "Galaxy Z Fold 5", date(2023, 8, 11), 1799.0, "Android Flagship"),
    ("Samsung", "Galaxy Z Flip 6", date(2024, 7, 24), 1099.0, "Android Flagship"),
    ("Samsung", "Galaxy Z Fold 6", date(2024, 7, 24), 1899.0, "Android Flagship"),
    # Google: Pixel 6 through 9
    ("Google", "Pixel 6", date(2021, 10, 28), 599.0, "Android Mid"),
    ("Google", "Pixel 6 Pro", date(2021, 10, 28), 899.0, "Android Flagship"),
    ("Google", "Pixel 6a", date(2022, 7, 28), 449.0, "Android Mid"),
    ("Google", "Pixel 7", date(2022, 10, 13), 599.0, "Android Mid"),
    ("Google", "Pixel 7 Pro", date(2022, 10, 13), 899.0, "Android Flagship"),
    ("Google", "Pixel 7a", date(2023, 5, 10), 499.0, "Android Mid"),
    ("Google", "Pixel 8", date(2023, 10, 12), 699.0, "Android Mid"),
    ("Google", "Pixel 8 Pro", date(2023, 10, 12), 999.0, "Android Flagship"),
    ("Google", "Pixel 8a", date(2024, 5, 14), 499.0, "Android Mid"),
    ("Google", "Pixel 9", date(2024, 8, 22), 799.0, "Android Mid"),
    ("Google", "Pixel 9 Pro", date(2024, 8, 22), 999.0, "Android Flagship"),
    ("Google", "Pixel 9 Pro XL", date(2024, 8, 22), 1099.0, "Android Flagship"),
    # OnePlus
    ("OnePlus", "OnePlus 10 Pro", date(2022, 3, 31), 899.0, "Android Flagship"),
    ("OnePlus", "OnePlus 11", date(2023, 2, 7), 699.0, "Android Flagship"),
    ("OnePlus", "OnePlus 12", date(2024, 2, 6), 799.0, "Android Flagship"),
    ("OnePlus", "OnePlus Nord N30", date(2023, 6, 27), 299.0, "Android Budget"),
    # Motorola
    ("Motorola", "Moto G Power (2024)", date(2024, 3, 1), 299.0, "Android Budget"),
    ("Motorola", "Moto G Stylus 5G (2024)", date(2024, 3, 1), 399.0, "Android Mid"),
    ("Motorola", "Razr 40 Ultra", date(2023, 7, 14), 999.0, "Android Flagship"),
    ("Motorola", "Razr 50 Ultra", date(2024, 7, 10), 999.0, "Android Flagship"),
]


def build_devices() -> pd.DataFrame:
    """Return the device catalog as a DataFrame with a stable device_id.

    The device_id is 1-indexed and assigned in catalog order so repeated
    generation runs produce identical IDs.
    """

    rows = [
        {
            "device_id": i + 1,
            "manufacturer": manufacturer,
            "model_family": model_family,
            "release_date": release_date,
            "msrp_new": msrp_new,
            "device_category": device_category,
        }
        for i, (manufacturer, model_family, release_date, msrp_new, device_category) in enumerate(
            _CATALOG
        )
    ]
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df
