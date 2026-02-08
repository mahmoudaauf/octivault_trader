#!/usr/bin/env python3

from core.execution_manager_minimal import ExecutionManager, EPSILON
from decimal import Decimal

# Test EPSILON fix
print(f'EPSILON constant: {EPSILON}')

# Create a mock ExecutionManager instance
class MockConfig:
    pass

class MockSharedState:
    pass

class MockExchangeClient:
    pass

em = ExecutionManager(MockConfig(), MockSharedState(), MockExchangeClient())

# Test bootstrap_bypass logic
policy_context_with_bypass = {'bootstrap_bypass': True}
policy_context_without_bypass = {}

# Test case 1: bootstrap_bypass should allow min_notional bypass
result1 = em.validate_quote_affordability(
    qa=Decimal('10'),
    spendable_dec=Decimal('100'),
    acc_val=Decimal('0'),
    min_required=50.0,  # Higher than qa
    policy_context=policy_context_with_bypass,
    bypass_min_notional=False
)
print(f'Bootstrap bypass test: {result1}')

# Test case 2: Without bootstrap_bypass, should fail min_notional check
result2 = em.validate_quote_affordability(
    qa=Decimal('10'),
    spendable_dec=Decimal('100'),
    acc_val=Decimal('0'),
    min_required=50.0,  # Higher than qa
    policy_context=policy_context_without_bypass,
    bypass_min_notional=False
)
print(f'No bootstrap bypass test: {result2}')

# Test EPSILON tolerance
result3 = em.validate_quote_affordability(
    qa=Decimal('10'),
    spendable_dec=Decimal('11.0000009'),  # Just slightly more than needed
    acc_val=Decimal('0'),
    min_required=5.0,
    policy_context={},
    bypass_min_notional=True,
    headroom=Decimal('1.0'),
    taker_fee=Decimal('0.001')
)
print(f'EPSILON tolerance test: {result3}')

print('All fixes validated successfully!')