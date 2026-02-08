"""
Minimal ExecutionManager for validating EPSILON and bootstrap_bypass fixes
"""

import asyncio
import contextlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

# Constants
EPSILON = Decimal("1e-6")

class ExecutionManager:
    """
    Minimal ExecutionManager class containing the EPSILON tolerance fix
    and bootstrap_bypass logic for validation.
    """

    def __init__(self, config: Any, shared_state: Any, exchange_client: Any, alert_callback=None):
        self.config = config
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_quote_affordability(
        self,
        qa: Decimal,
        spendable_dec: Decimal,
        acc_val: Decimal,
        min_required: float,
        policy_context: Optional[Dict[str, Any]] = None,
        eps: Decimal = Decimal("0.01"),
        headroom: Decimal = Decimal("1.0"),
        taker_fee: Decimal = Decimal("0.001"),
        bypass_min_notional: bool = False,
    ) -> Tuple[bool, Decimal, str]:
        """
        ✅ FIX #1: EPSILON tolerance for capital checks
        ✅ FIX #2: bootstrap_bypass for min_notional validation

        Validate if a quote amount is affordable given current capital constraints.
        """

        effective_qa = qa + acc_val

        # 3) ACCUMULATE_MODE/BOOTSTRAP_BYPASS CHECK: Skip min_notional validation for special modes
        # This allows P0 dust promotion and bootstrap execution to work without being blocked by min_notional guards
        accumulate_mode = False
        bootstrap_bypass = False
        no_downscale_planned_quote = False
        if policy_context:
            accumulate_mode = bool(policy_context.get("_accumulate_mode", False))
            bootstrap_bypass = bool(policy_context.get("bootstrap_bypass", False))
            no_downscale_planned_quote = bool(
                policy_context.get("_no_downscale_planned_quote", False)
                or policy_context.get("no_downscale_planned_quote", False)
            )

        # Skip min_notional check only for bootstrap_bypass mode
        # CRITICAL: bootstrap_bypass allows first trade on flat portfolio to execute
        bypass_min_notional = bootstrap_bypass

        if not bypass_min_notional:
            # 3) If planned quote + accumulation is below the floor, decide between MIN_NOTIONAL vs NAV shortfall
            if effective_qa < Decimal(str(min_required)) - eps:
                # If the user is effectively trying to spend all they have (including accumulation)
                # and that amount is still below the venue/config floor, classify as NAV shortfall.
                if spendable_dec > 0 and (qa <= spendable_dec + eps) and (spendable_dec + acc_val < Decimal(str(min_required)) - eps):
                    gap = (Decimal(str(min_required)) - (spendable_dec + acc_val)).max(Decimal("0"))
                    return (False, gap, "INSUFFICIENT_QUOTE")
                # Otherwise, the caller asked below the floor while having enough NAV.
                gap = (Decimal(str(min_required)) - effective_qa).max(Decimal("0"))
                return (False, gap, "QUOTE_LT_MIN_NOTIONAL")

        gross_needed = qa * (Decimal("1") + taker_fee) * headroom
        # CRITICAL FIX: Use EPSILON tolerance to avoid float/timing false negatives
        # Problem: spendable_dec < gross_needed can fail due to tiny float differences
        # Solution: Allow small tolerance (EPSILON = 1e-6) for capital availability
        if spendable_dec < gross_needed - EPSILON:
            # Point 3: Dynamic Resizing (Downscaling)
            # If we have less than planned, but enough for minNotional, we downscale.
            max_qa = spendable_dec / ((Decimal("1") + taker_fee) * headroom)
            if no_downscale_planned_quote:
                gap = (qa - max_qa).max(Decimal("0"))
                return (False, gap, "INSUFFICIENT_QUOTE")
            if max_qa >= Decimal(str(min_required)) or bypass_min_notional:
                self.logger.info(
                    f"[EM] Dynamic Resizing: Downscaling {qa} -> {max_qa:.2f} to fit spendable {spendable_dec:.2f}")
                return (True, max_qa, "OK_DOWNSCALED")
            else:
                # Point 2: Accumulation Pivot
                gap = (gross_needed - spendable_dec).max(Decimal("0"))
                return (False, gap, "INSUFFICIENT_QUOTE")

        return (True, qa, "OK")