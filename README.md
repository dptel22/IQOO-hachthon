# AI Revenue Recovery Pipeline

**Submission by:** [Your Name], 3rd-year AIML Student

## Architecture Overview

This project implements a **payment degradation → root cause → recovery** pipeline that automatically decides whether to retry, escalate, or take no action on failed payments, with full audit trail compliance.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Synthetic  │───▶│  Detection   │───▶│  Orchestration  │───▶│  Measurement     │
│   Dataset   │    │  (Logistic   │    │  Pipeline       │    │  Report          │
│  Generator  │    │  Regression) │    │  (process_event)│    │  (measure.py)    │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────────┘
       │                   │                   │                      │
       ▼                   ▼                   ▼                      ▼
  500 events           p_recoverable      retry / escalate /      Precision/Recall/F1
  JSONL format         + explanation      no_action per event     + Pipeline Recovery
  (400/100 split)      (coefficients)     + audit log (JSONL)     Rate + Cost Analysis
```

### Key Design Decisions (Locked)

| Area | Decision |
|------|----------|
| **Model** | Logistic Regression only (L2, C=2.0, class_weight={0:1, 1:3}) — no XGBoost, no SHAP |
| **Pipeline** | Single `process_event()` function with private helpers — no separate agent modules |
| **Escalation Rules** | Exactly two: (1) `amount > ₹10,000` immediate escalation, (2) `live_retry_count >= 3` max retries exhausted |
| **Failure Handlers** | (1) Missing/malformed features → neutral fallback + flag, (2) Infra timeout → bounded retry (separate counter) |
| **Measurement** | Detection metrics (model quality) and Pipeline metrics (business outcome) kept **strictly separate** |

---

## Quick Start

```bash
# 1. Install dependencies
pip install scikit-learn numpy

# 2. Generate synthetic dataset (Phase 1)
python generate_dataset.py

# 3. Train detection model (Phase 2)
python detection.py train

# 4. Run full batch pipeline (Phases 3-4)
python pipeline.py batch

# 5. Generate measurement report (Phase 5)
python measure.py
```

Outputs created:
- `synthetic_payment_failures.jsonl` — 500 events
- `synthetic_payment_failures_train.jsonl` / `test.jsonl` — 400/100 split
- `detector_model.pkl` / `detector_scaler.pkl` — trained artifacts
- `audit_log.jsonl` — append-only audit trail (one line per decision point)
- `measurement_report.md` / `measurement_report.json` — final report

---

## Phase Details

### Phase 1: Dataset Generator (`generate_dataset.py`)

**Output:** `synthetic_payment_failures.jsonl` (500 lines)

**Schema per event:**
```json
{
  "event_id": "evt_<uuid>",
  "timestamp": "ISO-8601",
  "amount": 1234.56,
  "currency": "INR",
  "customer_id": "cust_<8chars>",
  "payment_method": "card|upi|netbanking|wallet",
  "error_code": "payment_gateway_error|payment_declined|...",
  "error_reason": "Human-readable error description",
  "hour_of_day": 0-23,
  "prior_retry_count": 0-5,
  "recovered": 0|1,
  "metadata": { "category": "...", "latent_score": 0.0, "p_recover": 0.0 }
}
```

**Label Generation (Critical — Not a Lookup Table):**
The `recovered` label is **NOT** a hard boolean rule. It's sampled from a Bernoulli distribution where:
```
latent_score = base_weight[category] + interaction_terms
p_recover = sigmoid(latent_score)
recovered ~ Bernoulli(p_recover)
```
Then **18% of labels are flipped** as noise.

**Interaction Terms (4 total):**
1. `payment_method × amount`: UPI > ₹5000 recovers better
2. `hour_of_day × error_category`: Overnight (0-5am) gateway errors worse; daytime (8am-8pm) better
3. `payment_method × prior_retry_count`: UPI retries ≥3 recover worse than card
4. `amount × error_category`: High-value (>₹10K) temporary failures recover worse

**Why this matters:** A hard-rule label would make the model a trivial lookup table (~95%+ accuracy). The interaction effects + noise ensure the logistic regression has genuine signal to learn.

**Validation Gate (printed on run):**
- Class balance: ~30% recovered=1 (not extreme)
- ~10% events > ₹10,000 (exercises escalation rule)
- Error-code distribution matches targets (40% temporary, 30% declines, etc.)
- No single feature perfectly separates classes

---

### Phase 2: Detection Module (`detection.py`)

**Model:** Logistic Regression, standardized features (`StandardScaler`)

**Features (9):**
| Feature | Type |
|---------|------|
| amount_log | float (log(amount+1)) |
| prior_retry_count | int |
| error_code_category_temporary | binary |
| error_code_category_insufficient_funds | binary |
| error_code_category_card_declined | binary |
| error_code_category_auth_failed | binary |
| payment_method_upi | binary |
| payment_method_card | binary |
| hour_of_day | int |

**Standardization is required** — without it, coefficient contributions in the audit trail would be dominated by feature scale.

**Output per event:**
```json
{
  "p_recoverable": 0.6531,
  "fallback_flag": false,
  "explanation": {
    "amount_log": 0.0681,
    "prior_retry_count": 0.2178,
    "error_code_category_temporary": 0.419,
    ...
  }
}
```

**Holdout Metrics (from `python detection.py train`):**
- Precision: 0.3158
- Recall: 0.6207
- F1: 0.4186

These are **detection metrics only** — they measure classifier quality against the dataset's `recovered` label. They are NOT the pipeline recovery rate.

---

### Phase 3-4: Orchestration Pipeline (`pipeline.py`)

**Entry Point:** `process_event(event: dict) -> dict`

**Internal Helpers (all private, called from within `process_event`):**
1. `_validate_event` — required fields, writes `validation_error` audit record on failure
2. `_check_idempotency` — SHA256(event_id + attempt + action_type), returns cached outcome if seen
3. `_extract_features` — builds feature vector, applies fallback for missing fields (Failure Handler #1)
4. `_predict_recoverability` — calls trained model; if unavailable → `no_action` EXCEPT `amount > 10000` still escalates
5. `_decide_action` — **Locked decision logic:**
   ```
   if amount > 10000:           escalate, "amount_above_threshold"
   elif live_retry_count >= 3:  escalate, "max_retries_exhausted"
   elif p_recoverable >= 0.5:   retry,    "risk_score_above_threshold"
   else:                        no_action,"risk_score_below_threshold"
   ```
6. `_execute_intervention` — simulates retry outcome with success prob per category; Failure Handler #2: infra timeout → bounded retry (max 2, 500ms→1500ms) with **separate `infra_retry_count`**
7. `_escalate_to_manual_review` — writes escalation audit record; terminal state = `escalated_unresolved`
8. `_write_audit_log` — appends JSONL line per decision point

**Critical Distinctions:**
- `prior_retry_count` (dataset feature) ≠ `live_retry_count` (pipeline counter)
- `live_retry_count` seeded from `prior_retry_count` at pipeline start, then incremented on actual retries
- `infra_retry_count` (for timeouts) is completely separate — never added to business retries

**Mid-sequence re-check:** After each retry, if success → stop immediately (recovered). If failure and `live_retry_count < 3` → continue. If failure and `live_retry_count == 3` → escalate.

**Recovery Credit Rule:** Only count as pipeline-recovered if the pipeline's own intervention directly preceded success. Escalated events that might later be resolved manually are tracked separately (`escalated_unresolved`), not counted as pipeline recovery.

---

### Phase 5: Measurement Report (`measure.py`)

**Three Separate Sections (Never Conflated):**

#### A. Detection Metrics (Model Quality)
From Phase 2 holdout — independent of pipeline execution.
- Precision: 0.3158, Recall: 0.6207, F1: 0.4186

#### B. Pipeline / Business Metrics (from audit log)
| Metric | Count | Amount |
|--------|-------|--------|
| Recovered | 123 | INR 272,300.95 |
| Correctly Stopped (rules fired) | 299 | INR 1,956,623.61 |
| Escalated Unresolved | 299 | INR 1,956,623.61 |
| No Action | 78 | INR 117,943.17 |
| **Recovery Rate** (recovered ÷ flagged for retry) | | **0.3868** |

#### C. Cost Analysis (Dummy Assumptions — Stated Explicitly)
- False Positive (retry attempted, didn't recover): ₹10 each
- False Negative (no retry/escalated but was recoverable): lost amount
- Total Estimated Cost: INR 441,958.86

> **No ROI computation** — explicitly excluded.

**Sanity Check:** `recovered + escalated_unresolved + no_action + validation_error = 500` ✓

---

### Phase 6: Audit Trail Schema (`audit_log.jsonl`)

One JSON line per decision point. An event generates multiple lines.

```json
{
  "event_id": "evt_...",
  "timestamp": "ISO-8601",
  "module": "validation | detection | execution | decision | escalation",
  "decision": "retry | no_action | escalate | validation_error",
  "reason": "amount_above_threshold | max_retries_exhausted | risk_score_above_threshold | risk_score_below_threshold",
  "input": { "sanitized feature/event snapshot" },
  "output": { "outcome of this step" },
  "explanation": { "amount_log": 0.37, ... },  // only on detection records
  "idempotency_key": "sha256...",
  "fallback_flag": false,
  "audit_timestamp": "ISO-8601"
}
```

No `shap_values`. Explanation = standardized_feature × coefficient (logistic regression only).

---

## Deliverables Checklist

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Phase 1 — dataset generator with documented coefficients |
| `detection.py` | Phase 2 — train/evaluate Logistic Regression |
| `pipeline.py` | Phase 3-4 — `process_event()` + internal helpers |
| `measure.py` | Phase 5 — measurement report (3 separate sections) |
| `synthetic_payment_failures.jsonl` | Full dataset (500 events) |
| `audit_log.jsonl` | Full audit trail from batch run |
| `measurement_report.md/json` | Final report |
| `README.md` | This file |
| `architecture_diagram.svg` | Single-page pipeline diagram |

---

## What Was NOT Built (Explicit Non-Goals)

- ❌ XGBoost, SHAP, or any ensemble/tree models
- ❌ Separate "Diagnosis/Intervention/Execution agent" modules
- ❌ Live payment gateway API integration
- ❌ ROI metric
- ❌ UI/dashboard (terminal output only)
- ❌ Time-of-day retry restrictions or customer-preference flags as escalation triggers
- ❌ ML-based diagnosis step — root cause = static lookup from `error_code`

---

## Reproducing Results

All random seeds are fixed:
- Dataset: `RANDOM_SEED = 42`
- Model: `random_state=42`
- Execution simulation: deterministic hashing of `event_id`

Running the pipeline end-to-end should produce identical results to those in `measurement_report.md`.