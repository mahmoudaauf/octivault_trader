# Phase 9.5: Validation Framework
## Backtest Strategy Before Architecture Changes

**Purpose:** Validate 30m cumulative return prediction is viable  
**Duration:** Before any system code changes  
**Success Criteria:** Edge > 0.20%, Win rate > 45%, Persistence > 53%  

---

## 🔬 The Validation Process

### DO NOT implement system changes until these pass:

1. ✅ 30m labels are constructible
2. ✅ Forward move distribution is positive
3. ✅ Model can predict at > 53% accuracy
4. ✅ Break-even probability < actual win rate
5. ✅ Edge persists across regimes

---

## 🛠️ Step 1: Construct 30m Cumulative Return Labels

### Purpose: Create ground truth targets for 30m horizon

```python
import pandas as pd
import numpy as np

def build_30m_cumulative_return_labels(
    df,
    prediction_horizon_candles=6,
    edge_threshold_pct=0.0020,
    min_data_points=100
):
    """
    Build binary targets: "Does 30m cumulative return exceed edge threshold?"
    
    Args:
        df: DataFrame with 'close' column (sorted by time)
        prediction_horizon_candles: Number of 5m candles (default 6 = 30m)
        edge_threshold_pct: Minimum target return (default 0.20%)
        min_data_points: Minimum points to have for valid forward label
        
    Returns:
        DataFrame with new columns:
            - forward_return_30m: Actual cumulative return
            - target_30m: Binary (1 if return > threshold, 0 otherwise)
            - forward_return_valid: Whether label is valid
    """
    
    # Initialize
    df_copy = df.copy()
    df_copy['forward_return_30m'] = np.nan
    df_copy['target_30m'] = np.nan
    df_copy['forward_return_valid'] = False
    
    # For each historical point
    for i in range(len(df_copy) - prediction_horizon_candles):
        entry_price = df_copy['close'].iloc[i]
        exit_price = df_copy['close'].iloc[i + prediction_horizon_candles]
        
        # Calculate cumulative return
        cumulative_return = (exit_price - entry_price) / entry_price
        
        # Store results
        df_copy.loc[df_copy.index[i], 'forward_return_30m'] = cumulative_return
        df_copy.loc[df_copy.index[i], 'target_30m'] = 1 if cumulative_return > edge_threshold_pct else 0
        df_copy.loc[df_copy.index[i], 'forward_return_valid'] = True
    
    # Remove points without valid forward labels
    df_copy = df_copy[df_copy['forward_return_valid'] == True].copy()
    
    return df_copy


# Usage Example:
# df_with_targets = build_30m_cumulative_return_labels(df_ohlcv)
# Then save for analysis:
# df_with_targets.to_csv('ohlcv_with_30m_targets.csv')
```

### Output: Inspect Target Distribution

```python
def analyze_target_distribution(df):
    """
    Inspect: Is 30m return distribution realistic?
    """
    
    forward_returns = df['forward_return_30m'].dropna()
    
    print("=" * 70)
    print("30M CUMULATIVE RETURN DISTRIBUTION")
    print("=" * 70)
    
    print(f"\nTotal observations: {len(forward_returns)}")
    print(f"Valid labels: {len(df[df['target_30m'].notna()])}")
    
    print("\nReturn Statistics:")
    print(f"  Mean:              {forward_returns.mean():.4%}")
    print(f"  Median:            {forward_returns.median():.4%}")
    print(f"  Std Dev:           {forward_returns.std():.4%}")
    print(f"  Min:               {forward_returns.min():.4%}")
    print(f"  Max:               {forward_returns.max():.4%}")
    
    print("\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(forward_returns, p)
        print(f"  {p:2d}th: {val:.4%}")
    
    # Target distribution
    target_dist = df['target_30m'].value_counts(normalize=True)
    print("\nTarget Distribution:")
    print(f"  Positive (>0.20%):  {target_dist.get(1, 0):.1%}")
    print(f"  Negative (≤0.20%):  {target_dist.get(0, 0):.1%}")
    
    print("\n" + "=" * 70)
    
    # Validation
    positive_ratio = target_dist.get(1, 0)
    if 0.25 < positive_ratio < 0.45:
        print("✅ Target distribution looks realistic (30-40% positive)")
    elif positive_ratio < 0.25:
        print("⚠️  Too few positive targets (<25%) - check data quality")
    elif positive_ratio > 0.45:
        print("⚠️  Too many positive targets (>45%) - threshold may be too low")
    else:
        print("⚠️  Unexpected distribution")
    
    return forward_returns


# Usage:
# forward_returns = analyze_target_distribution(df_with_targets)
```

---

## 📊 Step 2: Measure Forward Move Distribution

### Question: Is forward move > 0.37%? (√6 × 0.15%)

```python
def validate_expected_move_assumptions(df):
    """
    Check: Does realized 30m volatility match assumptions?
    
    Theory:
        σ(30m) = σ(5m) × √6
        If σ(5m) ≈ 0.15%, then σ(30m) ≈ 0.37%
    
    Reality:
        Measure median 30m move
        Check if it's >= 0.37%
    """
    
    forward_returns = df['forward_return_30m'].dropna()
    absolute_moves = forward_returns.abs()
    
    print("=" * 70)
    print("EXPECTED MOVE VALIDATION (√TIME SCALING)")
    print("=" * 70)
    
    print("\nAbsolute Move Distribution:")
    print(f"  Mean:          {absolute_moves.mean():.4%}")
    print(f"  Median:        {absolute_moves.median():.4%}")
    print(f"  Std Dev:       {absolute_moves.std():.4%}")
    
    print("\nPercentiles (Absolute):")
    print(f"  25th:  {np.percentile(absolute_moves, 25):.4%}")
    print(f"  50th:  {np.percentile(absolute_moves, 50):.4%}")
    print(f"  75th:  {np.percentile(absolute_moves, 75):.4%}")
    print(f"  90th:  {np.percentile(absolute_moves, 90):.4%}")
    print(f"  95th:  {np.percentile(absolute_moves, 95):.4%}")
    
    median_move = absolute_moves.median()
    theoretical_move = 0.0037  # √6 × 0.15%
    
    print(f"\nComparison:")
    print(f"  Theoretical (√6 × 0.15%):  {theoretical_move:.4%}")
    print(f"  Actual median:             {median_move:.4%}")
    print(f"  Ratio (actual / theory):   {median_move / theoretical_move:.2f}x")
    
    print("\n" + "=" * 70)
    
    # Validation
    if median_move >= 0.0030:
        print("✅ Forward move distribution supports 30m horizon")
    else:
        print("⚠️  Forward moves smaller than expected - verify data")
    
    return absolute_moves


# Usage:
# absolute_moves = validate_expected_move_assumptions(df_with_targets)
```

---

## 🤖 Step 3: Train Test Model on 30m Targets

### Purpose: Does model learn directional persistence?

```python
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection as skm

def train_test_model_30m_horizon(df, feature_columns, lookback_periods=50):
    """
    Quick test model: Can LSTM predict 30m cumulative return > 0.20%?
    
    This is NOT the production model.
    This is validation that the problem is learnable.
    
    Args:
        df: DataFrame with features and 'target_30m'
        feature_columns: List of feature column names
        lookback_periods: Number of historical periods (default 50 for 250m)
        
    Returns:
        Model, history, test metrics
    """
    
    # Prepare data
    X = df[feature_columns].values
    y = df['target_30m'].dropna().values
    
    # Ensure lengths match
    valid_idx = df[df['target_30m'].notna()].index
    X_valid = df.loc[valid_idx, feature_columns].values
    y_valid = df.loc[valid_idx, 'target_30m'].values
    
    # Create sequences
    sequences = []
    labels = []
    for i in range(len(X_valid) - lookback_periods):
        sequences.append(X_valid[i:i+lookback_periods])
        labels.append(y_valid[i+lookback_periods])
    
    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    
    # Train-test split
    X_train, X_test, y_train, y_test = skm.train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Target distribution (train): {np.mean(y_train):.1%} positive")
    print(f"Target distribution (test): {np.mean(y_test):.1%} positive")
    
    # Build model
    model = keras.Sequential([
        keras.layers.LSTM(64, activation='relu', input_shape=(lookback_periods, len(feature_columns))),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ],
        verbose=0
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    print("\n" + "=" * 70)
    print("TEST MODEL PERFORMANCE")
    print("=" * 70)
    print(f"\nAccuracy:  {test_accuracy:.1%}")
    print(f"AUC:       {test_auc:.3f}")
    print(f"Precision: {precision_score(y_test, y_pred_binary):.1%}")
    print(f"Recall:    {recall_score(y_test, y_pred_binary):.1%}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_binary):.3f}")
    
    print("\n" + "=" * 70)
    
    # Validation
    if test_accuracy >= 0.53:
        print("✅ Model accuracy > 53% - pattern is learnable")
    else:
        print("⚠️  Model accuracy ≤ 53% - pattern may not be learnable")
        print("   Consider: Different features, longer/shorter horizon")
    
    return model, history, {
        'accuracy': test_accuracy,
        'auc': test_auc,
        'predictions': y_pred_prob,
        'y_test': y_test
    }


# Usage:
# model, history, metrics = train_test_model_30m_horizon(
#     df_with_targets,
#     feature_columns=['open', 'high', 'low', 'close', 'volume', ...],
#     lookback_periods=50
# )
```

---

## 💰 Step 4: Calculate Break-Even Probability

### Question: Can we win > break_even_prob?

```python
def calculate_break_even_probability(
    expected_move_pct=0.0037,
    cost_pct=0.0015,
    volatility_regime="normal"
):
    """
    Calculate: What win rate do we need?
    
    break_even_prob = cost / expected_move
    
    If actual win_rate > break_even_prob:
        → Strategy has positive EV
    
    If actual win_rate < break_even_prob:
        → Strategy has negative EV
    """
    
    break_even = cost_pct / expected_move_pct
    
    print("=" * 70)
    print("BREAK-EVEN PROBABILITY CALCULATION")
    print("=" * 70)
    
    print(f"\nRegime: {volatility_regime.upper()}")
    print(f"Expected Move:  {expected_move_pct:.4%}")
    print(f"Cost:           {cost_pct:.4%}")
    print(f"Edge:           {(expected_move_pct - cost_pct):.4%}")
    
    print(f"\nBreak-Even Win Rate:")
    print(f"  = Cost / Expected Move")
    print(f"  = {cost_pct:.4%} / {expected_move_pct:.4%}")
    print(f"  = {break_even:.2%}")
    
    print(f"\nInterpretation:")
    print(f"  Need {break_even:.1%} of trades to be winners")
    print(f"  (Remaining {1-break_even:.1%} can be losers)")
    
    print("\n" + "=" * 70)
    
    return break_even


def compare_predicted_vs_actual_win_rate(
    y_test,
    y_pred_prob,
    break_even_prob,
    confidence_threshold=0.55
):
    """
    Compare: Predicted win rate vs actual realized
    
    Question: Among high-confidence predictions,
              what % were actually correct?
    """
    
    # Filter to high-confidence predictions
    high_conf_mask = y_pred_prob > confidence_threshold
    y_test_filtered = y_test[high_conf_mask]
    y_pred_filtered = (y_pred_prob[high_conf_mask] > 0.5).astype(int)
    
    # Calculate win rate
    actual_win_rate = np.mean(y_test_filtered == y_pred_filtered)
    
    print("=" * 70)
    print("PREDICTED VS ACTUAL WIN RATE")
    print("=" * 70)
    
    print(f"\nFilter: Predictions with confidence > {confidence_threshold:.1%}")
    print(f"Sample size: {np.sum(high_conf_mask)} / {len(y_test)}")
    
    print(f"\nBreak-Even Win Rate: {break_even_prob:.2%}")
    print(f"Actual Win Rate:    {actual_win_rate:.2%}")
    
    print(f"\nDifference: {actual_win_rate - break_even_prob:+.2%}")
    
    print("\n" + "=" * 70)
    
    # Validation
    if actual_win_rate > break_even_prob + 0.05:
        print("✅ Win rate > break-even (with 5% margin) - Strategy viable")
    elif actual_win_rate > break_even_prob:
        print("⚠️  Win rate > break-even (marginal) - Monitor carefully")
    else:
        print("❌ Win rate < break-even - Strategy not viable")
    
    return actual_win_rate


# Usage:
# be_prob = calculate_break_even_probability(
#     expected_move_pct=0.0037,
#     cost_pct=0.0015,
#     volatility_regime="normal"
# )
#
# actual_wr = compare_predicted_vs_actual_win_rate(
#     y_test, y_pred_prob, be_prob, confidence_threshold=0.55
# )
```

---

## 🌦️ Step 5: Test Regime Sensitivity

### Question: Does pattern break in different market conditions?

```python
def analyze_regime_sensitivity(df, y_predictions):
    """
    Split data by volatility regime.
    Test if prediction accuracy changes between regimes.
    
    If accuracy changes > 15%, pattern is regime-dependent.
    """
    
    # Calculate rolling 20-period volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    
    # Define regimes
    vol_low = df['volatility_20'].quantile(0.33)
    vol_high = df['volatility_20'].quantile(0.67)
    
    df['regime'] = 'normal'
    df.loc[df['volatility_20'] < vol_low, 'regime'] = 'low'
    df.loc[df['volatility_20'] > vol_high, 'regime'] = 'high'
    
    print("=" * 70)
    print("REGIME SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    accuracy_by_regime = {}
    
    for regime in ['low', 'normal', 'high']:
        regime_mask = df['regime'] == regime
        regime_size = regime_mask.sum()
        
        if regime_size > 0:
            # Get predictions for this regime
            regime_targets = df.loc[regime_mask, 'target_30m'].dropna().values
            regime_predictions = y_predictions[:len(regime_targets)]  # Align
            
            accuracy = np.mean(regime_predictions == regime_targets)
            accuracy_by_regime[regime] = accuracy
            
            print(f"\n{regime.upper()} Volatility Regime:")
            print(f"  Data points:  {regime_size}")
            print(f"  Accuracy:     {accuracy:.1%}")
            print(f"  Volatility range: {df.loc[regime_mask, 'volatility_20'].min():.4%} - {df.loc[regime_mask, 'volatility_20'].max():.4%}")
    
    # Check sensitivity
    if len(accuracy_by_regime) > 0:
        accuracy_range = max(accuracy_by_regime.values()) - min(accuracy_by_regime.values())
        
        print(f"\n" + "=" * 70)
        print(f"Accuracy range across regimes: {accuracy_range:.1%}")
        
        if accuracy_range > 0.15:
            print("⚠️  Pattern is regime-dependent (>15% variation)")
            print("   Need regime-aware model or regime detection")
        else:
            print("✅ Pattern is robust across regimes (<15% variation)")
    
    return accuracy_by_regime


# Usage:
# regime_accuracy = analyze_regime_sensitivity(df_with_targets, y_pred_binary)
```

---

## ✅ Validation Checklist

Before implementing Phase 9.5, confirm:

```
□ Step 1: Label Construction
  ✓ 30m cumulative return labels built successfully
  ✓ Data spans minimum 500+ observations
  ✓ No NaN errors in forward labels
  
□ Step 2: Distribution Analysis
  ✓ Positive target ratio 25-45% (realistic edge threshold)
  ✓ Median forward move ≥ 0.30% (supports √time scaling)
  ✓ 75th percentile ≥ 0.50% (good tail behavior)
  
□ Step 3: Model Trainability
  ✓ Test model accuracy ≥ 53% (vs 51% random)
  ✓ AUC ≥ 0.55 (discriminative power)
  ✓ No overfitting signs (val_loss tracking train_loss)
  
□ Step 4: Break-Even Feasibility
  ✓ Calculated break-even ≤ 45%
  ✓ Actual win rate > break-even + 5%
  ✓ Margin of safety confirmed
  
□ Step 5: Regime Robustness
  ✓ Accuracy variation < 15% across regimes
  ✓ No regime completely breaks pattern
  ✓ Low vol regime maintains ≥ 52% accuracy
  
□ Cost Model Validated
  ✓ Actual slippage for 30m < 0.15%
  ✓ Edge > 0.20% confirmed
  ✓ Capital lock-up feasible
```

---

## 🚀 If All Checks Pass

Then proceed to Phase 9.5b (Cost Modeling):

1. Build production MLForecaster with 30m targets
2. Update break-even logic in ExecutionManager
3. Update expected_move calculation
4. Implement regime-aware cost model
5. Adjust position sizing for 30m duration

---

## ⛔ If Any Check Fails

Do NOT proceed. Instead:

- Adjust horizon (try 3-candle or 9-candle instead)
- Adjust edge threshold (try 0.15% or 0.25%)
- Check data quality (gaps, outliers)
- Verify feature engineering
- Re-evaluate Phase 8 baseline

---

## 📊 Running the Full Validation

```python
# Step 1: Build labels
df_targets = build_30m_cumulative_return_labels(df_ohlcv)
df_targets.to_csv('validation_with_targets.csv')

# Step 2: Analyze distribution
forward_returns = analyze_target_distribution(df_targets)
absolute_moves = validate_expected_move_assumptions(df_targets)

# Step 3: Train model
model, history, metrics = train_test_model_30m_horizon(
    df_targets,
    feature_columns=['open', 'high', 'low', 'close', 'volume'],
    lookback_periods=50
)

# Step 4: Calculate break-even
be_prob = calculate_break_even_probability(
    expected_move_pct=0.0037,
    cost_pct=0.0015
)

actual_wr = compare_predicted_vs_actual_win_rate(
    metrics['y_test'],
    metrics['predictions'],
    be_prob
)

# Step 5: Test regimes
regime_acc = analyze_regime_sensitivity(df_targets, metrics['predictions'])

# Summary
print("\n" + "=" * 70)
if forward_returns.median() > 0.003 and \
   metrics['accuracy'] > 0.53 and \
   actual_wr > be_prob + 0.05 and \
   max(regime_acc.values()) - min(regime_acc.values()) < 0.15:
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("   Ready to proceed to Phase 9.5b (Cost Modeling)")
else:
    print("⚠️  VALIDATION INCOMPLETE")
    print("   Do not proceed to implementation until all checks pass")
```

---

**Remember:** This validation is your insurance policy against implementing an unsound architecture.

Better to spend 1 week validating than 3 weeks fixing a broken system.

