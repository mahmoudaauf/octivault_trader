"""
Native carry: funding-rate carry strategy support for the native runtime.

Subpackage per the funding-carry engineering-study plan (see
/Users/mauf/.claude/plans/can-you-study-and-partitioned-pinwheel.md) -- kept
separate from the rest of core_engine/native/ because this strategy is
structurally different (delta-neutral, two-leg, spot+perpetual-futures) from
everything else in the runtime (single-symbol, long-spot-only), and is
deliberately built as a parallel, additive system rather than an extension
of the existing Position/Decision/gate model.
"""
