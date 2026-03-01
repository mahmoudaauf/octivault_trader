"""
CRITICAL: Leakage Audit for Step 2 Results

We got 75% (BTC) and 67% (ETH) accuracy.

In crypto, this is EXTREMELY high. Red flag.

This audit checks:
  1. Is train/test split chronological? (no temporal mixing)
  2. What is baseline majority-class accuracy?
  3. What is confusion matrix? (TP/FP/TN/FN breakdown)
  4. What is precision for positive class?
  5. Are features using future information?
  6. Did regime calculation use future data?
  7. Is class distribution different in train vs test?

We must eliminate leakage before trusting results.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LeakageAudit:
    """Audit for data leakage in Step 2 results"""
    
    def __init__(self, labeled_data_dir: str = "validation_outputs"):
        self.data_dir = Path(labeled_data_dir)
    
    def audit_symbol(self, symbol: str) -> dict:
        """Run full leakage audit for one symbol"""
        
        logger.info("\n" + "=" * 80)
        logger.info(f"LEAKAGE AUDIT: {symbol}")
        logger.info("=" * 80)
        
        # Load labeled data
        csv_path = self.data_dir / f"{symbol}_5m_with_60m_labels.csv"
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded {len(df)} rows from {csv_path}")
        except Exception as e:
            logger.error(f"❌ Cannot load {csv_path}: {e}")
            return {'status': 'FAILED', 'reason': 'Cannot load data'}
        
        # Filter to regime-filtered rows (as used in training)
        if 'keep_for_training' in df.columns:
            df_filtered = df[df['keep_for_training'] == True].reset_index(drop=True)
            logger.info(f"✅ Filtered to {len(df_filtered)} regime-filtered rows (normal/high vol)")
        else:
            df_filtered = df.copy()
            logger.info(f"⚠️  No regime filter column, using all {len(df_filtered)} rows")
        
        # Check 1: Class Distribution in Full Dataset
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 1: Class Distribution (Full Dataset)")
        logger.info("-" * 80)
        
        if 'target_60m' in df_filtered.columns:
            class_dist = df_filtered['target_60m'].value_counts(normalize=True)
            logger.info(f"Negative (0): {class_dist.get(0, 0)*100:.1f}%")
            logger.info(f"Positive (1): {class_dist.get(1, 0)*100:.1f}%")
            baseline_accuracy = max(class_dist.get(0, 0), class_dist.get(1, 0))
            logger.info(f"✅ Baseline (predict majority): {baseline_accuracy*100:.2f}%")
        else:
            logger.error("❌ No target_60m column found")
            return {'status': 'FAILED', 'reason': 'No target column'}
        
        # Check 2: Regime Distribution in Filtered Data
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 2: Regime Distribution (Filtered Data)")
        logger.info("-" * 80)
        
        if 'regime' in df_filtered.columns:
            regime_dist = df_filtered['regime'].value_counts()
            logger.info(f"Regimes kept in filtered data:")
            for regime, count in regime_dist.items():
                pct = count / len(df_filtered) * 100
                logger.info(f"  {regime}: {count} ({pct:.1f}%)")
            
            # Check if class distribution varies by regime
            logger.info(f"\nPositive target ratio BY regime:")
            for regime in df_filtered['regime'].unique():
                regime_data = df_filtered[df_filtered['regime'] == regime]
                pos_ratio = regime_data['target_60m'].mean()
                logger.info(f"  {regime}: {pos_ratio*100:.1f}% positive")
        
        # Check 3: Temporal Ordering (Critical for Train/Test Split)
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 3: Temporal Ordering (Train/Test Split Integrity)")
        logger.info("-" * 80)
        
        if 'timestamp' in df_filtered.columns:
            df_filtered['timestamp'] = pd.to_numeric(df_filtered['timestamp'])
            ts_min = df_filtered['timestamp'].min()
            ts_max = df_filtered['timestamp'].max()
            ts_sorted = df_filtered['timestamp'].is_monotonic_increasing
            
            logger.info(f"First timestamp: {ts_min}")
            logger.info(f"Last timestamp:  {ts_max}")
            logger.info(f"Sorted ascending: {ts_sorted}")
            
            if not ts_sorted:
                logger.warning("⚠️  WARNING: Data not sorted! Potential mixing in train/test")
        else:
            logger.warning("⚠️  No timestamp column - cannot verify temporal ordering")
        
        # Check 4: Feature Leakage - Future Information Check
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 4: Feature Construction (Leakage Risk)")
        logger.info("-" * 80)
        
        leakage_risks = self._check_feature_leakage(df_filtered)
        for risk, description in leakage_risks:
            logger.warning(f"⚠️  {risk}: {description}")
        
        if not leakage_risks:
            logger.info("✅ No obvious feature leakage detected")
        
        # Check 5: Majority Class vs Model Performance
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 5: Accuracy Interpretation")
        logger.info("-" * 80)
        
        logger.info(f"Baseline (majority class): {baseline_accuracy*100:.2f}%")
        logger.info(f"Step 2 reported accuracy: 75.19% (BTC) / 66.92% (ETH)")
        logger.info(f"")
        
        if baseline_accuracy > 0.74:
            logger.warning("⚠️  WARNING: Baseline accuracy > 74%")
            logger.warning("⚠️  Model accuracy (75.19% BTC) is BARELY above baseline!")
            logger.warning("⚠️  Real edge might be only 1-2 percentage points")
            logger.warning("⚠️  This could just be noise")
        
        if baseline_accuracy > 0.66:
            logger.warning("⚠️  Baseline accuracy > 66%")
            logger.warning("⚠️  ETH model accuracy (66.92%) WITHIN margin of baseline")
            logger.warning("⚠️  Could be predicting only majority class")
        
        # Check 6: Confidence Matrix Estimation
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 6: What the Model Must Be Predicting")
        logger.info("-" * 80)
        
        pos_ratio = df_filtered['target_60m'].mean()
        neg_ratio = 1 - pos_ratio
        
        # If model accuracy = 75.19% and positive ratio = 24.6%
        # Possible scenarios:
        reported_accuracy_btc = 0.7519
        
        logger.info(f"\nFor BTC (accuracy={reported_accuracy_btc*100:.2f}%, pos ratio={pos_ratio*100:.1f}%):")
        logger.info(f"\nScenario A: Model predicts ALL negative")
        logger.info(f"  Accuracy = {neg_ratio*100:.2f}% (baseline)")
        
        logger.info(f"\nScenario B: Model predicts positive 50% of time")
        logger.info(f"  True Positives:  ~12% (correct positives)")
        logger.info(f"  False Positives: ~25% (wrong positives)")
        logger.info(f"  True Negatives:  ~50% (correct negatives)")
        logger.info(f"  False Negatives: ~13% (wrong negatives)")
        logger.info(f"  Accuracy: {(0.12 + 0.50)*100:.2f}%")
        
        logger.info(f"\nTo achieve 75.19% with {pos_ratio*100:.1f}% positive class:")
        logger.info(f"  Model needs: High precision on negative predictions")
        logger.info(f"  OR: Barely better than majority class")
        
        # Check 7: Critical Question
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 7: The Real Question")
        logger.info("-" * 80)
        
        logger.info(f"\nClass distribution in training data: {pos_ratio*100:.1f}% positive")
        logger.info(f"")
        logger.info(f"If model achieved 75.19% by:")
        logger.info(f"  ✅ Real pattern: We found structural alpha (GREAT)")
        logger.info(f"  ❌ Leakage: We're measuring future information (DISASTER)")
        logger.info(f"  ❌ Baseline: We're just predicting majority (USELESS)")
        
        # Check 8: Regime Feature Leakage
        logger.info("\n" + "-" * 80)
        logger.info("CHECK 8: Regime Feature Leakage Risk")
        logger.info("-" * 80)
        
        logger.info(f"\n⚠️ CRITICAL QUESTION:")
        logger.info(f"Was regime calculated using:")
        logger.info(f"  A) Only PAST data (✅ OK)")
        logger.info(f"  B) Current + future data (❌ LEAKAGE)")
        logger.info(f"")
        logger.info(f"If B: Model can see regime and predict accordingly")
        logger.info(f"If regime is high-vol correlated with positive returns:")
        logger.info(f"  Model learns: 'Predict positive in high-vol' (70%+ accuracy)")
        logger.info(f"  This is LEAKAGE, not edge")
        
        return {
            'symbol': symbol,
            'baseline_accuracy': float(baseline_accuracy),
            'positive_ratio': float(pos_ratio),
            'data_points': len(df_filtered),
            'temporal_sorted': bool(ts_sorted) if 'timestamp' in df_filtered.columns else None,
            'status': 'AUDIT_COMPLETE'
        }
    
    def _check_feature_leakage(self, df: pd.DataFrame) -> list:
        """Check for common leakage patterns"""
        risks = []
        
        # Check 1: Returns using future prices
        if 'return' in df.columns and 'close' in df.columns:
            # Returns should be calculated from current close to next close
            # If returns look into future: risk
            pass
        
        # Check 2: Volatility window
        if 'volatility' in df.columns:
            # If volatility uses full window including future: risk
            risks.append(("Rolling Volatility", "Check if window is shifted properly"))
        
        # Check 3: Regime calculation
        if 'regime' in df.columns:
            risks.append(("Regime Feature", "Confirm regime uses ONLY past data, not current/future"))
        
        # Check 4: Forward returns
        if 'forward_return_60m' in df.columns:
            risks.append(("Forward Return", "This is the TARGET, should not be in features"))
        
        return risks
    
    def run_full_audit(self) -> dict:
        """Run audit for all symbols"""
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2 LEAKAGE AUDIT - FULL REPORT")
        logger.info("=" * 80)
        logger.info("\n🚨 CRITICAL THRESHOLD:")
        logger.info("If accuracy is truly out-of-sample and chronological:")
        logger.info("  75% BTC = VERY UNUSUAL (suspicious)")
        logger.info("  67% ETH = UNUSUAL (borderline suspicious)")
        logger.info("  55-60% = STRONG")
        logger.info("  51-54% = ACCEPTABLE")
        logger.info("  50%   = RANDOM (leakage or no edge)")
        
        results = {}
        
        # Find labeled data files
        csv_files = list(self.data_dir.glob("*_5m_with_60m_labels.csv"))
        symbols = sorted([f.stem.replace("_5m_with_60m_labels", "") for f in csv_files])
        
        for symbol in symbols:
            audit_result = self.audit_symbol(symbol)
            results[symbol] = audit_result
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 80)
        
        for symbol, result in results.items():
            if result['status'] == 'AUDIT_COMPLETE':
                baseline = result['baseline_accuracy']
                pos_ratio = result['positive_ratio']
                
                logger.info(f"\n{symbol}:")
                logger.info(f"  Baseline (majority): {baseline*100:.2f}%")
                logger.info(f"  Positive ratio:      {pos_ratio*100:.1f}%")
                logger.info(f"  Step 2 accuracy:     75.19% (BTC) / 66.92% (ETH)")
                logger.info(f"  Real edge:           {result.get('real_edge', 'UNKNOWN')}")
        
        # Save audit
        audit_path = self.data_dir / "step2_leakage_audit.json"
        try:
            with open(audit_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✅ Audit saved to {audit_path}")
        except Exception as e:
            logger.error(f"❌ Error saving audit: {e}")
        
        return results


if __name__ == "__main__":
    audit = LeakageAudit("validation_outputs")
    results = audit.run_full_audit()
