"""
5-fold Stratified Cross-Validation on FULL dataset (500 samples)
===============================================================

Runs CV on the complete improved model (base + interaction + polynomial features
with best hyperparameters and optimal threshold) to get robust estimates.
Reports mean/std of Precision, Recall, F1, ROC-AUC.
Also confirms model is saved correctly and loadable by pipeline.py.
"""

import json
import math
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from detection import FEATURE_NAMES, ERROR_CATEGORY, extract_features, UPI_HIGH_AMOUNT_LOG_THRESHOLD, HIGH_VALUE_AMOUNT_LOG_THRESHOLD

# ── Load full dataset ────────────────────────────────────────────────────────────

def load_full_dataset():
    """Load all 500 events from the full dataset."""
    X, y, raw = [], [], []
    with open('synthetic_payment_failures.jsonl', 'r') as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X.append(x)
            y.append(ev['recovered'])
            raw.append(ev)
    return np.array(X), np.array(y), raw

# ── Feature engineering (matching best model from test_interactions.py) ──────────

def add_polynomial_features(X, raw):
    """Add polynomial features: amount_log^2, amount_log^3, prior_retry_count^2, hour_of_day^2"""
    n = X.shape[0]
    poly = np.zeros((n, 4))
    for i, ev in enumerate(raw):
        amt_log = X[i, 0]
        prc = X[i, 1]
        hod = X[i, 8]
        poly[i, 0] = amt_log ** 2
        poly[i, 1] = amt_log ** 3
        poly[i, 2] = prc ** 2
        poly[i, 3] = hod ** 2
    return np.hstack([X, poly])

def add_interaction_features_v2(X, raw):
    """Add indicator-based interaction features matching data generator logic"""
    n = X.shape[0]
    inter = np.zeros((n, 4))
    for i, ev in enumerate(raw):
        amt = ev['amount']
        prc = ev['prior_retry_count']
        pm = ev['payment_method']
        hod = ev['hour_of_day']
        ec = ev['error_code']
        cat = ERROR_CATEGORY.get(ec, 'unknown')

        # 1: UPI * I(amount > 5000)
        if pm == 'upi' and amt > 5000:
            inter[i, 0] = 1.0
        # 2: I(hour in 0-5) * temporary - I(hour in 8-20) * temporary
        if cat == 'temporary':
            if 0 <= hod <= 5:
                inter[i, 1] = -1.0
            elif 8 <= hod <= 20:
                inter[i, 1] = 1.0
        # 3: UPI * I(prior_retry >= 3)
        if pm == 'upi' and prc >= 3:
            inter[i, 2] = 1.0
        # 4: I(amount > 10000) * temporary
        if amt > 10000 and cat == 'temporary':
            inter[i, 3] = 1.0
    return np.hstack([X, inter])

def build_best_model_features(X, raw):
    """Build the complete best feature set: base + interactions + polynomials"""
    X_with_inter = add_interaction_features_v2(X, raw)
    X_full = add_polynomial_features(X_with_inter, raw)
    return X_full

# ── 5-Fold Stratified CV ─────────────────────────────────────────────────────────

def run_stratified_cv():
    print("=" * 70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION ON FULL DATASET (500 samples)")
    print("=" * 70)

    # Load data
    X_base, y, raw = load_full_dataset()
    print(f"\nDataset loaded: {len(y)} samples, {sum(y)} positive ({sum(y)/len(y)*100:.1f}%)")

    # Build best feature set
    X = build_best_model_features(X_base, raw)
    print(f"Feature matrix shape: {X.shape}")

    # Best hyperparameters from test_interactions.py
    C = 2.0  # best C found
    class_weight = 'balanced'
    solver = 'lbfgs'
    max_iter = 1000
    random_state = 42

    # Optimal threshold found
    optimal_threshold = 0.58  # from test_interactions.py

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_precision = []
    cv_recall = []
    cv_f1 = []
    cv_roc_auc = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
        model.fit(X_train_scaled, y_train)

        # Predict probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= optimal_threshold).astype(int)

        # Metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_f1.append(f1)
        cv_roc_auc.append(roc_auc)

        print(f"\n  Fold {fold+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

    # Compute mean/std
    cv_precision = np.array(cv_precision)
    cv_recall = np.array(cv_recall)
    cv_f1 = np.array(cv_f1)
    cv_roc_auc = np.array(cv_roc_auc)

    print("\n" + "=" * 70)
    print("CV RESULTS (mean ± std over 5 folds)")
    print("=" * 70)
    print(f"Precision:  {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"Recall:     {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"F1 Score:   {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"ROC-AUC:    {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}")
    print("=" * 70)

    return {
        'cv_precision_mean': float(cv_precision.mean()),
        'cv_precision_std': float(cv_precision.std()),
        'cv_recall_mean': float(cv_recall.mean()),
        'cv_recall_std': float(cv_recall.std()),
        'cv_f1_mean': float(cv_f1.mean()),
        'cv_f1_std': float(cv_f1.std()),
        'cv_roc_auc_mean': float(cv_roc_auc.mean()),
        'cv_roc_auc_std': float(cv_roc_auc.std()),
    }

# ── Confirm model is saved and loadable ──────────────────────────────────────────

def confirm_model_saved():
    """Verify that detector_model.pkl and detector_scaler.pkl exist and work with pipeline.py"""
    print("\n" + "=" * 70)
    print("MODEL SAVING / LOADING VERIFICATION")
    print("=" * 70)

    # Check files exist
    import os
    model_exists = os.path.exists('detector_model.pkl')
    scaler_exists = os.path.exists('detector_scaler.pkl')
    print(f"detector_model.pkl exists: {model_exists}")
    print(f"detector_scaler.pkl exists: {scaler_exists}")

    # Try loading with pipeline.py's logic
    try:
        import detection
        with open('detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('detector_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Model and scaler load successfully")

        # Test prediction on a sample event
        with open('synthetic_payment_failures.jsonl', 'r') as f:
            ev = json.loads(f.readline())
        x, _ = detection.extract_features(ev)
        x_scaled = scaler.transform(x.reshape(1, -1))
        p = float(model.predict_proba(x_scaled)[0, 1])
        print(f"✓ Test prediction works: p_recoverable = {p:.4f}")

        model_saved = True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_saved = False

    return model_saved

# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run CV
    cv_results = run_stratified_cv()

    # Confirm model saved
    model_saved = confirm_model_saved()

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"CV Precision:  {cv_results['cv_precision_mean']:.4f} ± {cv_results['cv_precision_std']:.4f}")
    print(f"CV Recall:     {cv_results['cv_recall_mean']:.4f} ± {cv_results['cv_recall_std']:.4f}")
    print(f"CV F1:         {cv_results['cv_f1_mean']:.4f} ± {cv_results['cv_f1_std']:.4f}")
    print(f"CV ROC-AUC:    {cv_results['cv_roc_auc_mean']:.4f} ± {cv_results['cv_roc_auc_std']:.4f}")
    print(f"Model saved & loadable: {model_saved}")
    print("=" * 70)

    # Save results to JSON for StructuredOutput
    results = {
        'cv_precision_mean': cv_results['cv_precision_mean'],
        'cv_precision_std': cv_results['cv_precision_std'],
        'cv_recall_mean': cv_results['cv_recall_mean'],
        'cv_recall_std': cv_results['cv_recall_std'],
        'cv_f1_mean': cv_results['cv_f1_mean'],
        'cv_f1_std': cv_results['cv_f1_std'],
        'cv_roc_auc_mean': cv_results['cv_roc_auc_mean'],
        'cv_roc_auc_std': cv_results['cv_roc_auc_std'],
        'model_saved': model_saved,
    }

    import json
    with open('cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to cv_results.json")