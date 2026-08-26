"""
detection.py — Recoverability Detection Module
==============================================

Train: Logistic Regression (L2, C=2.0, class_weight='balanced')
Features: standardized before training (StandardScaler)
  Base (14): amount_log, prior_retry_count, error_code_category_*, payment_method_*, hour_of_day
  Interactions (4): upi_high_amount, overnight_temporary, daytime_temporary, upi_high_retry, high_value_temporary
  Polynomial (4): amount_log^2, amount_log^3, prior_retry_count^2, hour_of_day^2
Explanation: feature_value_standardized * coefficient per feature

Input event schema (from Phase 1):
  amount, prior_retry_count, error_code, payment_method, hour_of_day

Output per event:
  {
    "p_recoverable": 0.0,
    "fallback_flag": false,
    "explanation": {
      "amount_log": 0.37,
      "prior_retry_count": -0.41,
      "error_code_category_temporary": 0.62,
      ...
    }
  }
"""

import json
import math
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Feature Engineering ────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "amount_log",
    "prior_retry_count",
    "error_code_category_temporary",
    "error_code_category_insufficient_funds",
    "error_code_category_card_declined",
    "error_code_category_auth_failed",
    "payment_method_upi",
    "payment_method_card",
    "hour_of_day",
    # Threshold-based interaction features matching data generator logic:
    "upi_high_amount",          # UPI + amount > 5000
    "overnight_temporary",      # hour 0-5 + temporary category
    "daytime_temporary",        # hour 8-20 + temporary category
    "upi_high_retry",           # UPI + prior_retry_count >= 3
    "high_value_temporary",     # amount > 10000 + temporary category
    # Polynomial features:
    "amount_log_sq",            # amount_log^2
    "amount_log_cu",            # amount_log^3
    "prior_retry_sq",           # prior_retry_count^2
    "hour_of_day_sq",           # hour_of_day^2
]

# Threshold constants for interaction features (must match data generator)
UPI_HIGH_AMOUNT_THRESHOLD = 5000.0
UPI_HIGH_AMOUNT_LOG_THRESHOLD = math.log(UPI_HIGH_AMOUNT_THRESHOLD + 1)  # log(5001)
HIGH_VALUE_AMOUNT_THRESHOLD = 10000.0
HIGH_VALUE_AMOUNT_LOG_THRESHOLD = math.log(HIGH_VALUE_AMOUNT_THRESHOLD + 1)  # log(10001)

# Map error codes to categories (from Phase 1)
ERROR_CATEGORY = {
    "payment_gateway_error":        "temporary",
    "payment_processing_failed":    "temporary",
    "payment_failed":               "temporary",
    "payment_capture_failed":       "temporary",
    "payment_declined":             "card_declined",
    "payment_authorization_failed": "card_declined",
    "payment_insufficient_funds":   "insufficient_funds",
    "payment_authentication_failed":"auth_failed",
    "payment_invalid_amount":       "invalid_request",
    "payment_invalid_currency":     "invalid_request",
    "payment_expired_card":         "expired_card",
}


def add_interaction_features_v2(features: dict, event: dict) -> dict:
    """Add indicator-based interaction features matching data generator logic."""
    amt = event.get("amount", 0)
    prc = event.get("prior_retry_count", 0)
    pm = event.get("payment_method", "")
    hod = event.get("hour_of_day", 12)
    ec = event.get("error_code", "")
    cat = ERROR_CATEGORY.get(ec, "unknown")

    # 1: UPI * I(amount > 5000)
    features["upi_high_amount"] = 1.0 if (pm == "upi" and amt > 5000) else 0.0

    # 2a: Overnight (0-5) + temporary category
    features["overnight_temporary"] = 1.0 if (cat == "temporary" and 0 <= hod <= 5) else 0.0

    # 2b: Daytime (8-20) + temporary category
    features["daytime_temporary"] = 1.0 if (cat == "temporary" and 8 <= hod <= 20) else 0.0

    # 3: UPI * I(prior_retry >= 3)
    features["upi_high_retry"] = 1.0 if (pm == "upi" and prc >= 3) else 0.0

    # 4: I(amount > 10000) * temporary
    features["high_value_temporary"] = 1.0 if (amt > 10000 and cat == "temporary") else 0.0

    return features


def add_polynomial_features(features: dict) -> dict:
    """Add polynomial features."""
    amt_log = features["amount_log"]
    prc = features["prior_retry_count"]
    hod = features["hour_of_day"]

    features["amount_log_sq"] = amt_log ** 2
    features["amount_log_cu"] = amt_log ** 3
    features["prior_retry_sq"] = prc ** 2
    features["hour_of_day_sq"] = hod ** 2

    return features


def extract_features(event: dict) -> tuple[np.ndarray, bool]:
    """
    Build feature vector from event. Returns (features_array, fallback_flag).
    fallback_flag is True if any field was missing/malformed and neutral fallback was used.
    """
    fallback_flag = False
    features = {}

    # amount_log: log(amount + 1)
    try:
        amt = float(event.get("amount", 0))
        if amt <= 0:
            raise ValueError
        features["amount_log"] = math.log(amt + 1)
    except (TypeError, ValueError):
        features["amount_log"] = 7.3  # median-ish fallback
        fallback_flag = True

    # prior_retry_count
    try:
        prc = int(event.get("prior_retry_count", 0))
        if prc < 0 or prc > 5:
            raise ValueError
        features["prior_retry_count"] = float(prc)
    except (TypeError, ValueError):
        features["prior_retry_count"] = 1.0  # median-ish fallback
        fallback_flag = True

    # error_code_category binary features
    error_code = event.get("error_code", "")
    category = ERROR_CATEGORY.get(error_code, "unknown")
    features["error_code_category_temporary"]        = 1.0 if category == "temporary"        else 0.0
    features["error_code_category_insufficient_funds"] = 1.0 if category == "insufficient_funds" else 0.0
    features["error_code_category_card_declined"]    = 1.0 if category == "card_declined"    else 0.0
    features["error_code_category_auth_failed"]      = 1.0 if category == "auth_failed"      else 0.0
    # Note: invalid_request and expired_card are the reference (all zeros)

    # payment_method binary features
    pm = event.get("payment_method", "")
    features["payment_method_upi"] = 1.0 if pm == "upi" else 0.0
    features["payment_method_card"] = 1.0 if pm == "card" else 0.0
    # netbanking and wallet are reference (all zeros)

    # hour_of_day
    try:
        hod = int(event.get("hour_of_day", 12))
        if hod < 0 or hod > 23:
            raise ValueError
        features["hour_of_day"] = float(hod)
    except (TypeError, ValueError):
        features["hour_of_day"] = 12.0
        fallback_flag = True

    # Threshold-based interaction features (matching data generator logic exactly)
    features = add_interaction_features_v2(features, event)

    # Polynomial features
    features = add_polynomial_features(features)

    # Return as ordered array matching FEATURE_NAMES
    x = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    return x, fallback_flag


# ── Model Training / Loading ────────────────────────────────────────────────────

MODEL_PATH = "models/detector_model.pkl"
SCALER_PATH = "models/detector_scaler.pkl"

# Data paths
DATA_DIR = "data"
TRAIN_PATH = "data/synthetic_payment_failures_train.jsonl"
TEST_PATH = "data/synthetic_payment_failures_test.jsonl"

# Optimal threshold found from cross-validation
OPTIMAL_THRESHOLD = 0.47


def train_and_save():
    """Train on train split, evaluate on test split, save model and scaler."""
    # Load train data
    X_train = []
    y_train = []
    with open(TRAIN_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X_train.append(x)
            y_train.append(ev["recovered"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load test data
    X_test = []
    y_test = []
    with open(TEST_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X_test.append(x)
            y_test.append(ev["recovered"])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with improved hyperparameters
    model = LogisticRegression(
        C=2.0,
        penalty="l2",
        class_weight={0: 1, 1: 3},  # Optimized from grid search
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on holdout using optimal threshold
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=" * 65)
    print("PHASE 2 GATE — Detection Module Validation")
    print("=" * 65)
    print(f"\nHoldout Metrics (100 test events) with threshold={OPTIMAL_THRESHOLD}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # Also show default threshold (0.5) for comparison
    y_pred_default = (y_proba >= 0.5).astype(int)
    precision_d = precision_score(y_test, y_pred_default, zero_division=0)
    recall_d = recall_score(y_test, y_pred_default, zero_division=0)
    f1_d = f1_score(y_test, y_pred_default, zero_division=0)
    print(f"\n  (Default threshold=0.5: Precision={precision_d:.4f}, Recall={recall_d:.4f}, F1={f1_d:.4f})")

    # Warn if suspiciously perfect
    if precision > 0.95 and recall > 0.95:
        print("\n  ⚠ WARNING: Metrics suspiciously high (>95%) — check label noise in Phase 1!")
    else:
        print("\n  OK Metrics look reasonable (not trivially perfect)")

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Scaler saved to {SCALER_PATH}")

    # Print coefficients for reference
    print("\nCoefficients (for explanation computation):")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        print(f"  {name:35s}: {coef:+.4f}")
    print(f"  intercept:                        {model.intercept_[0]:+.4f}")

    return model, scaler, (precision, recall, f1)


def load_model_and_scaler():
    """Load trained model and scaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_recoverability(event: dict) -> dict:
    """
    Run detection on a single event.
    Returns dict with p_recoverable, fallback_flag, explanation.
    """
    model, scaler = load_model_and_scaler()

    x_raw, fallback_flag = extract_features(event)
    x_scaled = scaler.transform(x_raw.reshape(1, -1))

    p_recoverable = float(model.predict_proba(x_scaled)[0, 1])

    # Explanation: standardized feature value * coefficient
    explanation = {}
    for i, name in enumerate(FEATURE_NAMES):
        explanation[name] = round(float(x_scaled[0, i] * model.coef_[0, i]), 4)

    return {
        "p_recoverable": round(p_recoverable, 4),
        "fallback_flag": fallback_flag,
        "explanation": explanation,
    }


# ── Demo / CLI ────────────────────────────────────────────────────────────────

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_save()
    else:
        # Quick demo on a few test events
        model, scaler = load_model_and_scaler()
        print("Demo predictions on test events:")
        with open(TEST_PATH, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                ev = json.loads(line)
                result = predict_recoverability(ev)
                print(f"\n  Event: {ev['event_id']}")
                print(f"    amount={ev['amount']}, method={ev['payment_method']}, "
                      f"error={ev['error_code']}, retries={ev['prior_retry_count']}")
                print(f"    p_recoverable={result['p_recoverable']}, "
                      f"fallback={result['fallback_flag']}")
                print(f"    explanation: {result['explanation']}")


if __name__ == "__main__":
    main()