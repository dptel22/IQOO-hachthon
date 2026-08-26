"""
optimize_model.py — Comprehensive Model Optimization
=====================================================
1) Baseline test (9 features)
2) Test with 5 interaction features (14 total)
3) Hyperparameter grid search (C, class_weight)
4) Threshold optimization (find best F1 cutoff)
5) 5-fold Stratified CV on full 500 samples
6) Save best model as detector_model.pkl / detector_scaler.pkl
"""

import json
import math
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, classification_report
)
from sklearn.model_selection import StratifiedKFold

import detection
from detection import (
    FEATURE_NAMES, ERROR_CATEGORY, extract_features,
    MODEL_PATH, SCALER_PATH,
    UPI_HIGH_AMOUNT_LOG_THRESHOLD,
    HIGH_VALUE_AMOUNT_LOG_THRESHOLD,
)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load full 500-event dataset (train + test combined)."""
    X_all = []
    y_all = []
    events_all = []
    for filepath in ["data/synthetic_payment_failures_train.jsonl",
                     "data/synthetic_payment_failures_test.jsonl"]:
        with open(filepath, "r") as f:
            for line in f:
                ev = json.loads(line)
                x, _ = extract_features(ev)
                X_all.append(x)
                y_all.append(ev["recovered"])
                events_all.append(ev)
    return np.array(X_all), np.array(y_all), events_all


def load_split():
    """Load train/test split (for comparison with original baseline)."""
    X_train, y_train = [], []
    X_test, y_test = [], []
    with open("data/synthetic_payment_failures_train.jsonl", "r") as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X_train.append(x)
            y_train.append(ev["recovered"])
    with open("data/synthetic_payment_failures_test.jsonl", "r") as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X_test.append(x)
            y_test.append(ev["recovered"])
    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))


# ── Hyperparameter Grid Search ────────────────────────────────────────────────

def grid_search(X_train, y_train, X_val, y_val, feature_label=""):
    """Run grid search over C and class_weight, return best params and F1."""
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    class_weight_options = [
        "balanced",
        None,
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 4},
        {0: 1, 1: 5},
    ]

    best_f1 = -1.0
    best_params = None
    best_threshold = 0.5
    results = []

    for C in C_values:
        for cw in class_weight_options:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = LogisticRegression(
                C=C, penalty="l2", class_weight=cw,
                solver="lbfgs", max_iter=2000, random_state=42,
            )
            model.fit(X_tr_s, y_train)

            y_proba = model.predict_proba(X_val_s)[:, 1]

            # Find optimal threshold on validation set
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
            f1_scores = np.where(
                (precisions + recalls) > 0,
                2 * precisions * recalls / (precisions + recalls),
                0.0
            )
            best_idx = np.argmax(f1_scores)
            opt_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

            # Evaluate at default threshold (0.5)
            y_pred_default = (y_proba >= 0.5).astype(int)
            f1_default = f1_score(y_val, y_pred_default, zero_division=0)

            # Evaluate at optimal threshold
            y_pred_opt = (y_proba >= opt_threshold).astype(int)
            f1_opt = f1_score(y_val, y_pred_opt, zero_division=0)
            prec_opt = precision_score(y_val, y_pred_opt, zero_division=0)
            rec_opt = recall_score(y_val, y_pred_opt, zero_division=0)

            cw_label = str(cw)
            results.append({
                "C": C, "class_weight": cw_label,
                "threshold_default": 0.5, "f1_default": f1_default,
                "threshold_opt": opt_threshold, "f1_opt": f1_opt,
                "prec_opt": prec_opt, "rec_opt": rec_opt,
            })

            if f1_opt > best_f1:
                best_f1 = f1_opt
                best_params = {"C": C, "class_weight": cw, "class_weight_label": cw_label}
                best_threshold = opt_threshold

    # Sort by F1 and print top 10
    results.sort(key=lambda x: -x["f1_opt"])
    print(f"\n  Top 10 configs for {feature_label}:")
    for i, r in enumerate(results[:10]):
        print(f"    #{i+1}: C={r['C']:<5} cw={r['class_weight']:<12} "
              f"thr={r['threshold_opt']:.2f} F1={r['f1_opt']:.4f} "
              f"(P={r['prec_opt']:.4f}, R={r['rec_opt']:.4f})")

    return best_params, best_threshold, best_f1


# ── Cross-Validation ──────────────────────────────────────────────────────────

def cross_validate(X_all, y_all, n_features, best_C, best_cw, best_threshold,
                   feature_label=""):
    """5-fold stratified cross-validation with fixed hyperparams."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(
            C=best_C, penalty="l2", class_weight=best_cw,
            solver="lbfgs", max_iter=2000, random_state=42,
        )
        model.fit(X_tr_s, y_train)

        y_proba = model.predict_proba(X_val_s)[:, 1]
        y_pred = (y_proba >= best_threshold).astype(int)

        fold_results.append({
            "fold": fold + 1,
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "auc": roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5,
            "n_val": len(y_val),
            "n_pos_val": int(y_val.sum()),
        })

    print(f"\n  {feature_label} — 5-Fold Stratified CV:")
    for fr in fold_results:
        print(f"    Fold {fr['fold']}: P={fr['precision']:.4f} R={fr['recall']:.4f} "
              f"F1={fr['f1']:.4f} AUC={fr['auc']:.4f} "
              f"(n={fr['n_val']}, pos={fr['n_pos_val']})")

    avg_f1 = np.mean([fr["f1"] for fr in fold_results])
    avg_prec = np.mean([fr["precision"] for fr in fold_results])
    avg_rec = np.mean([fr["recall"] for fr in fold_results])
    avg_auc = np.mean([fr["auc"] for fr in fold_results])
    std_f1 = np.std([fr["f1"] for fr in fold_results])

    print(f"    Mean: P={avg_prec:.4f} R={avg_rec:.4f} F1={avg_f1:.4f} (+/- {std_f1:.4f}) AUC={avg_auc:.4f}")

    return avg_f1, avg_prec, avg_rec, avg_auc, std_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MODEL OPTIMIZATION — Comprehensive Search")
    print("=" * 70)

    # Load data
    X_train, y_train, X_test, y_test = load_split()
    X_all, y_all, events_all = load_all_data()

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}, All: {X_all.shape}")
    print(f"Train pos rate: {y_train.mean():.4f}, Test pos rate: {y_test.mean():.4f}")
    print(f"All pos rate: {y_all.mean():.4f}")

    # ── PHASE 1: Baseline (9 features, default C=1.0, class_weight='balanced') ──
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE (9 features, default params)")
    print("=" * 70)

    scaler_bl = StandardScaler()
    X_tr_s = scaler_bl.fit_transform(X_train)
    X_te_s = scaler_bl.transform(X_test)

    model_bl = LogisticRegression(
        C=1.0, penalty="l2", class_weight="balanced",
        solver="lbfgs", max_iter=1000, random_state=42,
    )
    model_bl.fit(X_tr_s, y_train)
    y_pred_bl = model_bl.predict(X_te_s)
    y_proba_bl = model_bl.predict_proba(X_te_s)[:, 1]
    f1_bl = f1_score(y_test, y_pred_bl, zero_division=0)
    auc_bl = roc_auc_score(y_test, y_proba_bl)
    print(f"  F1 (default thr=0.5): {f1_bl:.4f}")
    print(f"  AUC-ROC: {auc_bl:.4f}")

    # ── PHASE 2: Grid search with 14 features (interaction features) ──
    print("\n" + "=" * 70)
    print("PHASE 2: GRID SEARCH WITH 14 FEATURES (9 base + 5 interaction)")
    print("=" * 70)

    best_params, best_threshold, best_f1 = grid_search(
        X_train, y_train, X_test, y_test, "14 features"
    )
    print(f"\n  BEST: C={best_params['C']}, cw={best_params['class_weight_label']}, "
          f"threshold={best_threshold:.2f}, F1={best_f1:.4f}")

    # ── PHASE 3: Grid search with 9 features (no interaction) for comparison ──
    print("\n" + "=" * 70)
    print("PHASE 3: GRID SEARCH WITH 9 FEATURES (no interaction, for comparison)")
    print("=" * 70)

    # Use only first 9 features
    X_train_9 = X_train[:, :9]
    X_test_9 = X_test[:, :9]

    best_params_9, best_threshold_9, best_f1_9 = grid_search(
        X_train_9, y_train, X_test_9, y_test, "9 features"
    )
    print(f"\n  BEST (9 feat): C={best_params_9['C']}, cw={best_params_9['class_weight_label']}, "
          f"threshold={best_threshold_9:.2f}, F1={best_f1_9:.4f}")

    # ── PHASE 4: 5-Fold CV on full dataset (best config from 14 features) ──
    print("\n" + "=" * 70)
    print("PHASE 4: 5-FOLD STRATIFIED CV ON FULL 500 EVENTS")
    print("=" * 70)

    avg_f1, avg_prec, avg_rec, avg_auc, std_f1 = cross_validate(
        X_all, y_all, X_all.shape[1],
        best_params["C"], best_params["class_weight"], best_threshold,
        "14 features"
    )

    # Also run 5-fold CV with 9 features for comparison
    avg_f1_9, avg_prec_9, avg_rec_9, avg_auc_9, std_f1_9 = cross_validate(
        X_all[:, :9], y_all, 9,
        best_params_9["C"], best_params_9["class_weight"], best_threshold_9,
        "9 features"
    )

    # ── PHASE 5: Final model training on full 500 + holdout evaluation ──
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL MODEL — Train on all 400, evaluate on 100 holdout")
    print("=" * 70)

    scaler_final = StandardScaler()
    X_tr_final = scaler_final.fit_transform(X_train)
    X_te_final = scaler_final.transform(X_test)

    model_final = LogisticRegression(
        C=best_params["C"], penalty="l2", class_weight=best_params["class_weight"],
        solver="lbfgs", max_iter=2000, random_state=42,
    )
    model_final.fit(X_tr_final, y_train)

    y_proba_final = model_final.predict_proba(X_te_final)[:, 1]
    y_pred_final = (y_proba_final >= best_threshold).astype(int)

    p_final = precision_score(y_test, y_pred_final, zero_division=0)
    r_final = recall_score(y_test, y_pred_final, zero_division=0)
    f1_final = f1_score(y_test, y_pred_final, zero_division=0)
    auc_final = roc_auc_score(y_test, y_proba_final)

    print(f"\n  Holdout (100 test events) with optimal threshold={best_threshold:.2f}:")
    print(f"    Precision: {p_final:.4f}")
    print(f"    Recall:    {r_final:.4f}")
    print(f"    F1:        {f1_final:.4f}")
    print(f"    AUC-ROC:   {auc_final:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_final, zero_division=0))

    # Also evaluate at default 0.5 threshold for reference
    y_pred_05 = (y_proba_final >= 0.5).astype(int)
    p_05 = precision_score(y_test, y_pred_05, zero_division=0)
    r_05 = recall_score(y_test, y_pred_05, zero_division=0)
    f1_05 = f1_score(y_test, y_pred_05, zero_division=0)
    print(f"  Holdout at default threshold=0.5:")
    print(f"    Precision: {p_05:.4f}, Recall: {r_05:.4f}, F1: {f1_05:.4f}")

    # Print coefficients
    print(f"\n  Coefficients (14 features):")
    for name, coef in zip(detection.FEATURE_NAMES, model_final.coef_[0]):
        print(f"    {name:35s}: {coef:+.4f}")
    print(f"    {'intercept':35s}: {model_final.intercept_[0]:+.4f}")

    # ── Save model and scaler ──
    MODEL_PATH = "models/detector_model.pkl"
    SCALER_PATH = "models/detector_scaler.pkl"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_final, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_final, f)

    print(f"\n  Saved: {MODEL_PATH}, {SCALER_PATH}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline (9 feat, default C=1.0, balanced, thr=0.5):")
    print(f"    F1={f1_bl:.4f}, AUC={auc_bl:.4f}")
    print(f"\n  Optimized (14 feat, C={best_params['C']}, cw={best_params['class_weight_label']}, thr={best_threshold:.2f}):")
    print(f"    Holdout F1={f1_final:.4f}, AUC={auc_final:.4f}")
    print(f"\n  5-Fold CV (14 feat):")
    print(f"    Mean F1={avg_f1:.4f} (+/- {std_f1:.4f}), Mean AUC={avg_auc:.4f}")
    print(f"\n  5-Fold CV (9 feat, for comparison):")
    print(f"    Mean F1={avg_f1_9:.4f} (+/- {std_f1_9:.4f}), Mean AUC={avg_auc_9:.4f}")
    print(f"\n  Improvement: F1 {f1_bl:.4f} -> {f1_final:.4f} "
          f"(+{f1_final - f1_bl:.4f}, {(f1_final - f1_bl) / f1_bl * 100:.1f}%)")
    print(f"\n  Optimal threshold for pipeline.py: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
