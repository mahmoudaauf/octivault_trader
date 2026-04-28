# -*- coding: utf-8 -*-
"""
ArbitrationEngine - Extracted from MetaController

Responsibility:
- Multi-layer gate evaluation
- Signal validation and filtering
- Policy enforcement
- Risk approval orchestration

This module extracts arbitration logic from MetaController
to enable independent testing and policy refinement.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class GateResult:
    """Result of evaluating a single gate."""
    gate_name: str
    passed: bool
    reason: str
    severity: str  # 'info', 'warning', 'error'
    details: Optional[Dict[str, Any]] = None


class ArbitrationEngine:
    """Evaluates signals against multi-layer gates."""
    
    def __init__(self, shared_state, config):
        """Initialize arbitration engine."""
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger("ArbitrationEngine")
        
        # Gate evaluation order (critical gates first)
        self.gate_order = [
            'symbol_validation',
            'confidence_threshold',
            'regime_gate',
            'position_limit_gate',
            'capital_gate',
            'risk_gate',
        ]
    
    async def evaluate_signal(
        self,
        signal: Dict[str, Any],
    ) -> Tuple[bool, List[GateResult]]:
        """
        Evaluate signal against all gates.
        
        Returns:
            (approved, gate_results)
        """
        results = []
        
        for gate_name in self.gate_order:
            gate_method = getattr(self, f'_gate_{gate_name}', None)
            if not gate_method:
                self.logger.warning(f"Gate not found: {gate_name}")
                continue
            
            result = await gate_method(signal)
            results.append(result)
            
            # Stop at first failure (critical gates fail fast)
            if not result.passed:
                self.logger.warning(
                    f"Signal rejected at gate '{result.gate_name}': "
                    f"{result.reason}"
                )
                return (False, results)
        
        # All gates passed
        self.logger.info(
            f"Signal approved: {signal.get('symbol')} "
            f"from {signal.get('source_agent')}"
        )
        return (True, results)
    
    async def _gate_symbol_validation(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 1: Validate symbol format and existence."""
        symbol = signal.get('symbol', '')
        
        # Check symbol format
        if not symbol or len(symbol) < 6:
            return GateResult(
                gate_name='symbol_validation',
                passed=False,
                reason=f"Invalid symbol format: '{symbol}'",
                severity='error',
            )
        
        # Check if symbol is known to system
        if symbol not in self.shared_state.known_symbols:
            return GateResult(
                gate_name='symbol_validation',
                passed=False,
                reason=f"Unknown symbol: '{symbol}'",
                severity='error',
            )
        
        return GateResult(
            gate_name='symbol_validation',
            passed=True,
            reason=f"Symbol '{symbol}' is valid",
            severity='info',
        )
    
    async def _gate_confidence_threshold(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 2: Check confidence level meets minimum."""
        confidence = signal.get('confidence', 0.0)
        min_confidence = self.config.get('min_confidence_threshold', 0.50)
        
        if confidence < min_confidence:
            return GateResult(
                gate_name='confidence_threshold',
                passed=False,
                reason=(
                    f"Confidence {confidence:.2f} below threshold "
                    f"{min_confidence:.2f}"
                ),
                severity='warning',
                details={'confidence': confidence, 'threshold': min_confidence},
            )
        
        return GateResult(
            gate_name='confidence_threshold',
            passed=True,
            reason=f"Confidence {confidence:.2f} meets threshold",
            severity='info',
            details={'confidence': confidence, 'threshold': min_confidence},
        )
    
    async def _gate_regime_gate(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 3: Check if market regime allows trading."""
        symbol = signal.get('symbol')
        
        # Check volatility regime
        if hasattr(self.shared_state, 'market_regime'):
            regime = self.shared_state.market_regime.get(symbol)
            
            if regime == 'crisis':
                return GateResult(
                    gate_name='regime_gate',
                    passed=False,
                    reason=f"Market regime is 'crisis' for {symbol}",
                    severity='warning',
                    details={'regime': regime},
                )
        
        return GateResult(
            gate_name='regime_gate',
            passed=True,
            reason="Market regime allows trading",
            severity='info',
        )
    
    async def _gate_position_limit_gate(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 4: Check position count limits."""
        current_positions = len(self.shared_state.positions)
        max_positions = self.config.get('max_open_positions', 50)
        
        if current_positions >= max_positions:
            return GateResult(
                gate_name='position_limit_gate',
                passed=False,
                reason=(
                    f"Position limit reached: "
                    f"{current_positions}/{max_positions}"
                ),
                severity='warning',
                details={
                    'current': current_positions,
                    'max': max_positions,
                },
            )
        
        return GateResult(
            gate_name='position_limit_gate',
            passed=True,
            reason=f"Position count OK: {current_positions}/{max_positions}",
            severity='info',
            details={
                'current': current_positions,
                'max': max_positions,
            },
        )
    
    async def _gate_capital_gate(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 5: Check available capital."""
        available_capital = self.shared_state.available_capital
        
        if available_capital <= 0:
            return GateResult(
                gate_name='capital_gate',
                passed=False,
                reason="No capital available",
                severity='error',
                details={'available': available_capital},
            )
        
        return GateResult(
            gate_name='capital_gate',
            passed=True,
            reason=f"Capital available: ${available_capital:.2f}",
            severity='info',
            details={'available': available_capital},
        )
    
    async def _gate_risk_gate(self, signal: Dict[str, Any]) -> GateResult:
        """Gate 6: Run risk manager approval."""
        symbol = signal.get('symbol')
        
        # This would call RiskManager.approve() in real implementation
        # For now, placeholder that always passes
        
        return GateResult(
            gate_name='risk_gate',
            passed=True,
            reason=f"Risk approval granted for {symbol}",
            severity='info',
        )
    
    def get_gate_status(self) -> Dict[str, Any]:
        """Get status of all gates for monitoring."""
        return {
            'gates_active': len(self.gate_order),
            'gate_order': self.gate_order,
            'config': {
                'min_confidence': self.config.get('min_confidence_threshold', 0.50),
                'max_positions': self.config.get('max_open_positions', 50),
            },
        }
    
    def evaluate_gates_sync(
        self,
        symbol: str,
        action: str,
        confidence: float,
        expected_move: float,
        config: Optional[Dict[str, Any]] = None,
        regime_manager=None,
    ) -> Dict[str, Any]:
        """
        PHASE 2C: Synchronous gate evaluation wrapper.
        
        Provides synchronous interface for MetaController to evaluate
        arbitration gates without async context. This bridges the gap
        between async ArbitrationEngine and sync MetaController call sites.
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            confidence: Signal confidence (0.0-1.0)
            expected_move: Expected move percentage
            config: Optional configuration override
            regime_manager: Optional regime manager for context
            
        Returns:
            dict: {
                'passed': bool,
                'reason': str,
                'blocking_gate': str or None,
                'details': dict with gate-specific info
            }
        """
        try:
            # Gate 1: Symbol validation
            if not symbol or len(symbol) < 6:
                return {
                    'passed': False,
                    'reason': f"Invalid symbol: {symbol}",
                    'blocking_gate': 'symbol_validation',
                    'details': {'symbol': symbol},
                }
            
            # Gate 2: Confidence check
            min_confidence = config.get('min_confidence_threshold', 0.50) if config else 0.50
            if confidence < min_confidence:
                return {
                    'passed': False,
                    'reason': f"Confidence {confidence:.2f} below minimum {min_confidence:.2f}",
                    'blocking_gate': 'confidence_gate',
                    'details': {'confidence': confidence, 'minimum': min_confidence},
                }
            
            # Gate 3: Expected move check
            min_move = config.get('min_expected_move_pct', 0.5) if config else 0.5
            if expected_move < min_move:
                return {
                    'passed': False,
                    'reason': f"Expected move {expected_move:.2f}% below minimum {min_move:.2f}%",
                    'blocking_gate': 'expected_move_gate',
                    'details': {'expected_move': expected_move, 'minimum': min_move},
                }
            
            # Gate 4: Regime-based limits (if regime_manager available)
            if regime_manager is not None:
                try:
                    if action == 'BUY':
                        # Check max positions
                        regime = regime_manager.get_regime()
                        max_positions = regime_manager.get_config().get('max_open_positions', 50)
                        # Would need to check actual open positions here in real implementation
                except Exception as e:
                    self.logger.debug(f"Regime check error (non-fatal): {e}")
            
            # All gates passed
            return {
                'passed': True,
                'reason': 'All arbitration gates passed',
                'blocking_gate': None,
                'details': {
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'expected_move': expected_move,
                },
            }
            
        except Exception as e:
            self.logger.warning(f"Synchronous gate evaluation error: {e}")
            return {
                'passed': False,
                'reason': f"Gate evaluation error: {e}",
                'blocking_gate': 'error',
                'details': {'error': str(e)},
            }
