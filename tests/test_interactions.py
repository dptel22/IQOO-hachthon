import json
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from detection import FEATURE_NAMES, ERROR_CATEGORY, extract_features

def load_data(filepath):
    X, y, raw = [], [], []
    with open(filepath) as f:
        for line in f:
            ev = json.loads(line)
            x, _ = extract_features(ev)
            X.append(x); y.append(ev['recovered']); raw.append(ev)
    return np.array(X), np.array(y), raw

X_train, y_train, tr = load_data('synthetic_payment_failures_train.jsonl')
X_test, y_test, te = load_data('synthetic_payment_failures_test.jsonl')

# Baseline
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_train)
X_te_s = sc.transform(X_test)
m = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
m.fit(X_tr_s, y_train)
print(f'BASELINE: F1={f1_score(y_test, m.predict(X_te_s)):.4f}, AUC={roc_auc_score(y_test, m.predict_proba(X_te_s)[:,1]):.4f}')

# Raw interactions (WRONG)
def add_raw_interactions(X, raw):
    n = X.shape[0]
    inter = np.zeros((n, 4))
    for i, ev in enumerate(raw):
        amt = ev['amount']; prc = ev['prior_retry_count']
        pm = ev['payment_method']; hod = ev['hour_of_day']
        ec = ev['error_code']; cat = ERROR_CATEGORY.get(ec, 'unknown')
        if pm == 'upi' and amt > 5000: inter[i, 0] = 1.0
        if cat == 'temporary':
            if 0 <= hod <= 5: inter[i, 1] = -1.0
            elif 8 <= hod <= 20: inter[i, 1] = 1.0
        if pm == 'upi' and prc >= 3: inter[i, 2] = -1.0
        if amt > 10000 and cat == 'temporary': inter[i, 3] = -1.0
    return np.hstack([X, inter])

X_tr_ri = add_raw_interactions(X_train, tr)
X_te_ri = add_raw_interactions(X_test, te)
sc2 = StandardScaler()
X_tr_ri_s = sc2.fit_transform(X_tr_ri)
X_te_ri_s = sc2.transform(X_te_ri)
m2 = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
m2.fit(X_tr_ri_s, y_train)
print(f'RAW INTERACTIONS: F1={f1_score(y_test, m2.predict(X_te_ri_s)):.4f}, AUC={roc_auc_score(y_test, m2.predict_proba(X_te_ri_s)[:,1]):.4f}')

# Feature-space interactions (RIGHT approach)
def add_fs_interactions(X, raw):
    n = X.shape[0]
    inter = np.zeros((n, 4))
    for i, ev in enumerate(raw):
        amt = ev['amount']; prc = ev['prior_retry_count']
        pm = ev['payment_method']; hod = ev['hour_of_day']
        ec = ev['error_code']; cat = ERROR_CATEGORY.get(ec, 'unknown')
        # Feature indices: 0=amount_log, 1=prior_retry, 2=temp, 6=upi, 8=hour
        if pm == 'upi' and amt > 5000: inter[i, 0] = X[i, 0] * X[i, 6]
        if cat == 'temporary': inter[i, 1] = X[i, 8] * X[i, 2]
        if pm == 'upi' and prc >= 3: inter[i, 2] = X[i, 1] * X[i, 6]
        if amt > 10000 and cat == 'temporary': inter[i, 3] = X[i, 0] * X[i, 2]
    return np.hstack([X, inter])

X_tr_fi = add_fs_interactions(X_train, tr)
X_te_fi = add_fs_interactions(X_test, te)
sc3 = StandardScaler()
X_tr_fi_s = sc3.fit_transform(X_tr_fi)
X_te_fi_s = sc3.transform(X_te_fi)
m3 = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
m3.fit(X_tr_fi_s, y_train)
print(f'FEATURE-SPACE INTERACTIONS: F1={f1_score(y_test, m3.predict(X_te_fi_s)):.4f}, AUC={roc_auc_score(y_test, m3.predict_proba(X_te_fi_s)[:,1]):.4f}')

# Better feature-space: use indicator * feature (matching generator's logic)
def add_fs_interactions_v2(X, raw):
    n = X.shape[0]
    inter = np.zeros((n, 4))
    for i, ev in enumerate(raw):
        amt = ev['amount']; prc = ev['prior_retry_count']
        pm = ev['payment_method']; hod = ev['hour_of_day']
        ec = ev['error_code']; cat = ERROR_CATEGORY.get(ec, 'unknown')
        # 1: UPI * I(amount > 5000)
        if pm == 'upi' and amt > 5000: inter[i, 0] = 1.0
        # 2: I(hour in 0-5) * temporary - I(hour in 8-20) * temporary
        if cat == 'temporary':
            if 0 <= hod <= 5: inter[i, 1] = -1.0
            elif 8 <= hod <= 20: inter[i, 1] = 1.0
        # 3: UPI * I(prior_retry >= 3)
        if pm == 'upi' and prc >= 3: inter[i, 2] = 1.0
        # 4: I(amount > 10000) * temporary
        if amt > 10000 and cat == 'temporary': inter[i, 3] = 1.0
    return np.hstack([X, inter])

X_tr_fi2 = add_fs_interactions_v2(X_train, tr)
X_te_fi2 = add_fs_interactions_v2(X_test, te)
sc4 = StandardScaler()
X_tr_fi2_s = sc4.fit_transform(X_tr_fi2)
X_te_fi2_s = sc4.transform(X_te_fi2)
m4 = LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
m4.fit(X_tr_fi2_s, y_train)
print(f'INDICATOR INTERACTIONS: F1={f1_score(y_test, m4.predict(X_te_fi2_s)):.4f}, AUC={roc_auc_score(y_test, m4.predict_proba(X_te_fi2_s)[:,1]):.4f}')

# Print feature correlations
print("\n=== Feature statistics ===")
for i, name in enumerate(FEATURE_NAMES):
    print(f"{name}: mean={X_train[:,i].mean():.4f}, std={X_train[:,i].std():.4f}")

print("\n=== Raw interaction statistics ===")
X_train_ri = add_raw_interactions(X_train, tr)
for j in range(4):
    col = X_train_ri[:, 9+j]
    print(f"  Raw Int {j+1}: mean={col.mean():.4f}, std={col.std():.4f}, non-zero={(col!=0).sum()}")

print("\n=== Feature-space interaction statistics ===")
X_train_fi = add_fs_interactions(X_train, tr)
for j in range(4):
    col = X_train_fi[:, 9+j]
    print(f"  FS Int {j+1}: mean={col.mean():.4f}, std={col.std():.4f}, non-zero={(col!=0).sum()}")