"""Turn a claimed size into outcome units, and small shared statistics helpers."""
from __future__ import annotations

from typing import Optional

from .schema import Claim, Estimate


def claimed_effect(claim: Optional[Claim], est: Optional[Estimate]) -> Optional[float]:
    """Signed claimed effect in outcome units (per unit of treatment for a continuous
    treatment), or None if the claim gives no size or it cannot be converted.

    percent  -> size/100 * baseline (control) mean        ("by 15%")
    pp       -> size/100 (outcome is a 0/1 rate)           ("by 5 percentage points")
    sd       -> size * outcome SD                          ("by 0.2 standard deviations")
    times    -> (size - 1) * baseline mean                 ("doubles" = 2 times)
    absolute -> size                                       ("by 3 points")
    The sign follows claim.direction (decrease -> negative). no_effect -> 0.
    """
    if claim is None:
        return None
    if claim.direction == "no_effect":
        return 0.0
    if claim.size is None or claim.size_unit is None:
        return None
    size = abs(float(claim.size))
    unit = claim.size_unit
    base = est.baseline_mean if est else None
    sd = est.outcome_sd if est else None
    if unit == "percent":
        if base is None or base == 0:
            return None
        val = size / 100.0 * abs(base)
    elif unit == "pp":
        val = size / 100.0
    elif unit == "sd":
        if not sd:
            return None
        val = size * sd
    elif unit == "times":
        if base is None:
            return None
        val = abs(size - 1.0) * abs(base)   # "halves" (0.5x) and "doubles" (2x) both have magnitude |x-1|
    else:
        val = size
    return -val if claim.direction == "decrease" else val
