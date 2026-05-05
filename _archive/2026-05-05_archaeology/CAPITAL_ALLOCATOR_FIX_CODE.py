"""
CAPITAL ALLOCATOR FIX IMPLEMENTATION
=====================================
Date: May 4, 2026
Purpose: Implement dynamic reserve calculation and 60/20/20 capital split
Target File: src/l6_governance/capital_allocator.py

This file contains the three replacement functions needed to fix the capital
allocation architecture. See CAPITAL_ALLOCATOR_FIX_ANALYSIS.md for context.

ROOT CAUSE:
- Line 766: default_bootstrap_reserve = 2.0 (FIXED, not dynamic)
- Impact: Prevents 60/20/20 split from functioning (becomes 85/0/15)
- Solution: Replace with NAV-percentage-based tiered reserve system
"""

from decimal import Decimal

# ============================================================================
# PART 1: DYNAMIC RESERVE CALCULATION (replaces fixed $2.00)
# ============================================================================


def calculate_dynamic_reserve(nav: float, cfg: dict, logger=None) -> float:
    """
    Calculate NAV-based reserve with tiered percentages.

    This replaces the hardcoded $2.00 reserve in line 766 with a dynamic
    percentage-based system that scales with account size.

    Reserve Tiers:
    - NAV < $50 (MICRO):     20% reserve (safety for small accounts)
    - NAV $50-$200 (SMALL):  15% reserve (moderate accounts)
    - NAV >= $200 (NORMAL):  10% reserve (larger accounts)

    Args:
        nav: Net Asset Value in USDT
        cfg: Config dict containing:
            - RESERVE_PCT_MICRO (default 0.20)
            - RESERVE_PCT_SMALL (default 0.15)
            - RESERVE_PCT_NORMAL (default 0.10)
            - RESERVE_MIN_USDT (default 1.00)
            - RESERVE_MAX_USDT (default NAV * 0.40)
        logger: Optional logger instance

    Returns:
        Dynamic reserve amount in USDT

    Example:
        >>> calculate_dynamic_reserve(84.55, cfg)
        16.91  # 20% of $84.55 for micro tier

    Previous:
        >>> default_bootstrap_reserve  # Hardcoded
        2.0    # Same for all account sizes - WRONG
    """
    # Get config values with sensible defaults
    reserve_pct_micro = cfg.get("RESERVE_PCT_MICRO", 0.20)
    reserve_pct_small = cfg.get("RESERVE_PCT_SMALL", 0.15)
    reserve_pct_normal = cfg.get("RESERVE_PCT_NORMAL", 0.10)
    reserve_min = cfg.get("RESERVE_MIN_USDT", 1.00)
    reserve_max = cfg.get("RESERVE_MAX_USDT", nav * 0.40)

    # Select percentage tier based on NAV
    if nav < 50:
        tier = "MICRO"
        reserve_pct = reserve_pct_micro
    elif nav < 200:
        tier = "SMALL"
        reserve_pct = reserve_pct_small
    else:
        tier = "NORMAL"
        reserve_pct = reserve_pct_normal

    # Calculate dynamic reserve
    dynamic_reserve = nav * reserve_pct

    # Apply bounds
    bounded_reserve = max(reserve_min, min(reserve_max, dynamic_reserve))

    if logger:
        logger.debug(
            f"Dynamic reserve calc: NAV=${nav:.2f}, tier={tier}, "
            f"pct={reserve_pct*100:.0f}%, dynamic=${dynamic_reserve:.2f}, "
            f"bounded=${bounded_reserve:.2f}"
        )

    return bounded_reserve


# ============================================================================
# PART 2: 60/20/20 CAPITAL ALLOCATION SPLIT
# ============================================================================


def allocate_capital_60_20_20(free_usdt: float, cfg: dict, logger=None) -> dict[str, float]:
    """
    Split allocatable capital into 60/20/20 tiers.

    After reserve is deducted from NAV, the remaining allocatable capital
    is split across three distinct budgets:
    - 60% → Trading Core (BTC/ETH main strategies)
    - 20% → Trading Alts (growth/emerging coins)
    - 20% → Dust Healing (recovery/liquidation)

    This ensures dust healing operations get guaranteed budget floor,
    preventing starvation when trading signals compete.

    Args:
        free_usdt: Allocatable capital (after reserve) in USDT
        cfg: Config dict containing:
            - ALLOC_PCT_CORE (default 0.60)
            - ALLOC_PCT_ALTS (default 0.20)
            - ALLOC_PCT_DUST (default 0.20)
        logger: Optional logger instance

    Returns:
        Dict with keys:
        {
            'trading_core': float,      # 60% - BTC/ETH strategies
            'trading_alts': float,      # 20% - Alt coin strategies
            'dust_healing': float,      # 20% - Dust liquidation
            'effective_trading': float  # 80% - Total trading budget
        }

    Example:
        >>> allocate_capital_60_20_20(67.64, cfg)
        {
            'trading_core': 40.58,      # 60% of $67.64
            'trading_alts': 13.53,      # 20% of $67.64
            'dust_healing': 13.53,      # 20% of $67.64
            'effective_trading': 54.11  # 60% + 20%
        }

    Previous:
        >>> # No explicit allocation tiers
        # System used: 100% trading / 0% dust healing
        # Result: Dust healing starved, can't liquidate
    """
    # Get allocation percentages from config
    pct_core = cfg.get("ALLOC_PCT_CORE", 0.60)
    pct_alts = cfg.get("ALLOC_PCT_ALTS", 0.20)
    pct_dust = cfg.get("ALLOC_PCT_DUST", 0.20)

    # Normalize percentages to ensure they sum to 1.0
    total_pct = pct_core + pct_alts + pct_dust
    if abs(total_pct - 1.0) > 0.01:  # Tolerance for rounding
        pct_core = pct_core / total_pct
        pct_alts = pct_alts / total_pct
        pct_dust = pct_dust / total_pct

    # Calculate allocations using Decimal for precision
    free_decimal = Decimal(str(free_usdt))
    core = float(free_decimal * Decimal(str(pct_core)))
    alts = float(free_decimal * Decimal(str(pct_alts)))
    dust = float(free_decimal * Decimal(str(pct_dust)))

    allocation = {
        "trading_core": core,
        "trading_alts": alts,
        "dust_healing": dust,
        "effective_trading": core + alts,
    }

    if logger:
        logger.debug(
            f"60/20/20 split: Free=${free_usdt:.2f} → "
            f"Core=${core:.2f}(60%) + Alts=${alts:.2f}(20%) + "
            f"Dust=${dust:.2f}(20%)"
        )

    return allocation


# ============================================================================
# PART 3: MASTER ORCHESTRATOR FUNCTION
# ============================================================================


async def allocate_with_nav_dynamics(
    self,  # CapitalAllocator instance
    nav: float,
    free_usdt: float,
    mode: str = "NORMAL",
) -> dict[str, any]:
    """
    Master allocation orchestrator combining dynamic reserve + 60/20/20 split.

    This is the main entry point for capital allocation decisions. It:
    1. Calculates dynamic reserve based on NAV
    2. Deducts reserve from free_usdt to get allocatable
    3. Splits allocatable into 60/20/20 tiers
    4. Determines trading mode (MICRO, NORMAL, GROWTH)
    5. Returns complete allocation structure

    This replaces the old allocation logic that used:
    - Fixed $2.00 reserve (line 766)
    - No explicit dust healing allocation
    - No trading tier separation

    Args:
        self: CapitalAllocator instance with logger and config
        nav: Current Net Asset Value in USDT
        free_usdt: Free capital available in USDT
        mode: Trading mode ("NORMAL", "RECOVERY", "GROWTH")

    Returns:
        Complete allocation dict:
        {
            'reserve': float,              # Locked reserve
            'allocatable': float,          # After reserve
            'trading_core': float,         # 60% of allocatable
            'trading_alts': float,         # 20% of allocatable
            'dust_healing': float,         # 20% of allocatable
            'effective_trading': float,    # core + alts
            'mode': str,                   # MICRO/NORMAL/GROWTH
            'capital_floor_met': bool,     # >= min_agent_budget
            'allocation_mode': str,        # LOCKED/RESTRICTED/FULL
        }

    Example:
        >>> allocation = await allocate_with_nav_dynamics(
        ...     self, nav=84.55, free_usdt=8.73
        ... )
        >>> allocation['dust_healing']
        1.75  # Can now liquidate dust with budget floor!

    Previous:
        >>> # Old logic would return:
        >>> {'trading': 8.73, 'dust_healing': 0}  # Dust starved!
    """
    # Get config
    cfg = self._cfg  # or however configs are accessed
    min_agent_budget = cfg.get("MIN_AGENT_BUDGET", 10.0)

    # Step 1: Calculate dynamic reserve
    reserve = calculate_dynamic_reserve(nav, cfg, self.logger)

    # Step 2: Calculate allocatable (free capital after reserve)
    allocatable = max(0, free_usdt - reserve)

    # Step 3: Determine capital floor status
    capital_floor_met = free_usdt >= min_agent_budget

    # Step 4: Determine allocation mode
    if not capital_floor_met:
        allocation_mode = "LOCKED"  # Can't trade, must liquidate dust
    elif free_usdt < (min_agent_budget * 1.5):
        allocation_mode = "RESTRICTED"  # Limited trading
    else:
        allocation_mode = "FULL"  # Normal operation

    # Step 5: Split allocatable if we have room
    if allocatable > 0:
        split = allocate_capital_60_20_20(allocatable, cfg, self.logger)
    else:
        split = {
            "trading_core": 0,
            "trading_alts": 0,
            "dust_healing": 0,
            "effective_trading": 0,
        }

    # Step 6: Determine mode based on NAV
    if nav < 50:
        trading_mode = "MICRO"
    elif nav < 200:
        trading_mode = "NORMAL"
    else:
        trading_mode = "GROWTH"

    # Step 7: Assemble complete allocation
    allocation = {
        "reserve": reserve,
        "allocatable": allocatable,
        "trading_core": split["trading_core"],
        "trading_alts": split["trading_alts"],
        "dust_healing": split["dust_healing"],
        "effective_trading": split["effective_trading"],
        "mode": trading_mode,
        "capital_floor_met": capital_floor_met,
        "allocation_mode": allocation_mode,
    }

    if self.logger:
        self.logger.info(
            f"NAV Allocation: NAV=${nav:.2f}, Free=${free_usdt:.2f}, "
            f"Reserve=${reserve:.2f}, Allocatable=${allocatable:.2f}, "
            f"Mode={trading_mode}/{allocation_mode}, "
            f"Split=${split['trading_core']:.2f}/"
            f"${split['trading_alts']:.2f}/${split['dust_healing']:.2f}"
        )

    return allocation


# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================
"""
DEPLOYMENT CHECKLIST:

[ ] 1. Add to capital_allocator.py imports:
    from decimal import Decimal

[ ] 2. Replace lines 766-790 with calculate_dynamic_reserve() function

[ ] 3. Add allocate_capital_60_20_20() as class method

[ ] 4. Add allocate_with_nav_dynamics() as async class method

[ ] 5. Update calling code in MetaController:
    OLD: allocation = self.capital_allocator.get_allocation(free_usdt)
    NEW: allocation = await self.capital_allocator.allocate_with_nav_dynamics(
             nav=current_nav, free_usdt=free_capital
         )

[ ] 6. Update dust healing component:
    OLD: dust_budget = leftover_capital
    NEW: dust_budget = allocation['dust_healing']

[ ] 7. Add config to .env:
    RESERVE_PCT_MICRO=0.20
    RESERVE_PCT_SMALL=0.15
    RESERVE_PCT_NORMAL=0.10
    RESERVE_MIN_USDT=1.00
    ALLOC_PCT_CORE=0.60
    ALLOC_PCT_ALTS=0.20
    ALLOC_PCT_DUST=0.20

[ ] 8. Test with current $84.55 NAV:
    Expected reserve: $16.91 (20%)
    Expected allocatable: $67.64
    Expected core: $40.58
    Expected alts: $13.53
    Expected dust: $13.53

[ ] 9. Monitor allocation ratios in production logs

[ ] 10. Verify dust healing executes with budget floor
"""
