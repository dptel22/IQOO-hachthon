# Backend/Frontend Data Contract

**Version:** 1.0.0  
**Last Updated:** 2026-08-25  
**Backend Owner:** Dhruv Patel  
**Frontend Consumer:** External team (separate repo)

---

## Overview

This contract defines the static output files produced by the backend and the one live endpoint for failure injection. The frontend reads these files directly — no live API calls required for normal operation.

**Architecture:** Backend produces static JSON/JSONL files → Frontend reads them at build time or runtime via static hosting.

---

## 1. `events_stream.json`

**Purpose:** Powers the Event Stream + Event Detail screens.  
**Format:** JSON array of 500 event objects.  
**Sorted by:** `timestamp` (ascending).

### Schema

```json
{
  "event_id": "string (uuid format, e.g. evt_3c10316d-3f63-4065-be85-c8c4a9470663)",
  "timestamp": "string (ISO 8601, UTC, e.g. 2026-07-26T04:54:46.004871+00:00)",
  "amount": "number (float, INR, e.g. 1164.11)",
  "payment_method": "string (enum: card | upi | netbanking | wallet)",
  "error_code": "string (enum: payment_gateway_error | payment_processing_failed | payment_failed | payment_capture_failed | payment_declined | payment_authorization_failed | payment_insufficient_funds | payment_authentication_failed | payment_invalid_amount | payment_invalid_currency | payment_expired_card)",
  "p_recoverable": "number (float, 0.0-1.0, model probability of recovery, rounded to 4 decimals)",
  "decision": "string | null (enum: retry | no_action | escalate | null)",
  "reason": "string | null (enum: risk_score_above_threshold | risk_score_below_threshold | amount_above_threshold | max_retries_exhausted | success_on_retry | retry_failed_continue | null)",
  "explanation": "object (feature_name -> contribution, rounded to 4 decimals, see feature list below)",
  "fallback_flag": "boolean (true if any input field was missing/malformed and neutral fallback used)",
  "outcome": "string (enum: recovered | correctly_stopped | escalated_unresolved | no_action | validation_error)"
}
```

### Field Details

| Field | Type | Required | Enum Values | Description |
|-------|------|----------|-------------|-------------|
| event_id | string | Yes | — | Unique event identifier |
| timestamp | string | Yes | — | Event occurrence time (ISO 8601) |
| amount | number | Yes | — | Payment amount in INR |
| payment_method | string | Yes | card, upi, netbanking, wallet | Payment method used |
| error_code | string | Yes | See enum list above | Specific error code |
| p_recoverable | number | Yes | 0.0–1.0 | Model's predicted recovery probability |
| decision | string/null | Yes | retry, no_action, escalate, null | Pipeline decision |
| reason | string/null | Yes | See enum list above | Reason for decision |
| explanation | object | Yes | — | Feature → contribution mapping (standardized feature × coefficient) |
| fallback_flag | boolean | Yes | — | True if neutral fallbacks were used for missing/malformed inputs |
| outcome | string | Yes | recovered, correctly_stopped, escalated_unresolved, no_action, validation_error | Final business outcome |

### Explanation Feature Keys (always present, may be 0)

```
amount_log, prior_retry_count, error_code_category_temporary,
error_code_category_insufficient_funds, error_code_category_card_declined,
error_code_category_auth_failed, payment_method_upi, payment_method_card,
hour_of_day, upi_high_amount, overnight_temporary, daytime_temporary,
upi_high_retry, high_value_temporary, amount_log_sq, amount_log_cu,
prior_retry_sq, hour_of_day_sq
```

### Outcome Definitions

| Outcome | Meaning |
|---------|---------|
| recovered | Retry succeeded, payment recovered |
| correctly_stopped | Escalated due to amount > ₹10,000 OR max retries (3) exhausted — rules working as designed |
| escalated_unresolved | Escalated for other reasons (e.g., infra timeout), no human resolution in batch window |
| no_action | Score below threshold (p < 0.5), pipeline correctly did not retry |
| validation_error | Input event failed validation (missing/invalid required fields) |

---

## 2. `audit_log.jsonl`

**Purpose:** Powers the drill-down/audit view per event.  
**Format:** JSON Lines (one JSON object per line).  
**Lines:** ~2,669 (multiple records per event).

### Schema (per line)

```json
{
  "event_id": "string",
  "module": "string (enum: validation | idempotency | detection | decision | execution | escalation)",
  "decision": "string (e.g. score_computed, retry, no_action, escalate, validation_error, executed, idempotent_replay, execution_error)",
  "reason": "string (context-specific, e.g. model_inference, risk_score_above_threshold, amount_above_threshold, max_retries_exhausted, infra_timeout_exhausted)",
  "input": "object (module-specific input data)",
  "output": "object (module-specific output data, may contain terminal_state)",
  "explanation": "object (feature → contribution, only populated for detection/decision modules)",
  "idempotency_key": "string (SHA256 hash, empty string if not applicable)",
  "fallback_flag": "boolean",
  "audit_timestamp": "string (ISO 8601, UTC, when this audit record was written)"
}
```

### Module Types and Their `decision` Values

| Module | Decision Values |
|--------|----------------|
| validation | validation_error |
| idempotency | idempotent_replay, retry, escalate |
| detection | score_computed |
| decision | retry, no_action, escalate |
| execution | executed, execution_error |
| escalation | escalate |

### Terminal States (in `output.terminal_state`)

- `recovered` — retry succeeded
- `no_action` — score below threshold
- `escalated_unresolved` — escalated, no human resolution
- `pending_next_retry` — retry failed, will retry again (intermediate)
- `validation_error` — input validation failed

---

## 3. `measurement_report.json`

**Purpose:** Powers the Measurement screen.  
**Format:** Single JSON object with three top-level sections.

### Schema

```json
{
  "generated_at": "string (ISO 8601)",
  "detection_metrics": {
    "precision": "number (0.0-1.0)",
    "recall": "number (0.0-1.0)",
    "f1": "number (0.0-1.0)",
    "test_size": "integer",
    "positive_rate": "number (0.0-1.0)"
  },
  "pipeline_metrics": {
    "recovered": "integer",
    "recovered_amount": "number (INR)",
    "correctly_stopped": "integer",
    "correctly_stopped_amount": "number (INR)",
    "escalated_unresolved": "integer",
    "escalated_unresolved_amount": "number (INR)",
    "no_action": "integer",
    "no_action_amount": "number (INR)",
    "validation_error": "integer",
    "flagged_for_retry": "integer",
    "recovery_rate": "number (0.0-1.0, recovered / flagged_for_retry)",
    "false_positives": "integer",
    "false_negatives": "integer",
    "fp_cost": "number (INR)",
    "fn_cost": "number (INR)",
    "total_cost": "number (INR)"
  },
  "sanity_check": {
    "audit_events": "integer",
    "dataset_events": "integer",
    "partition_sum": "integer",
    "fully_accounted": "boolean",
    "issues": "array of strings"
  },
  "cost_assumptions": {
    "fp_cost_per_event": 10.0,
    "fn_cost_multiplier": 1.0
  }
}
```

### Section Definitions

| Section | Description |
|---------|-------------|
| detection_metrics | Model quality from Phase 2 holdout (100 events). Evaluates classifier in isolation against dataset `recovered` label. |
| pipeline_metrics | Business outcomes from full batch run (500 events) via audit log. NOT the same as detection metrics. |
| sanity_check | Verifies 100% event accounting, no silent drops. |

---

## 4. `model_metadata.json`

**Purpose:** Provides all model details so frontend doesn't need to guess.  
**Format:** Single JSON object.

### Schema

```json
{
  "model_name": "string (LogisticRegression)",
  "version": "string (semver)",
  "description": "string",
  "features": "array of strings (18 feature names in order)",
  "feature_descriptions": "object (feature_name -> description)",
  "coefficients": "object (feature_name -> coefficient value)",
  "intercept": "number",
  "threshold": "number (decision threshold, 0.47)",
  "class_weight": "object (0: 1, 1: 3)",
  "metrics": {
    "holdout": { "precision", "recall", "f1", "test_size", "positive_rate" },
    "cross_validation": { "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std" }
  },
  "scaler": {
    "type": "string (StandardScaler)",
    "mean": "array of 18 numbers",
    "scale": "array of 18 numbers"
  }
}
```

---

## 5. Live Endpoint: `POST /simulate-event`

**Purpose:** Failure-injection panel — frontend sends malformed/edge-case events, gets back same shape as `events_stream.json` items.  
**Protocol:** HTTP/JSON, no auth, single endpoint.

### Request

```http
POST /simulate-event
Content-Type: application/json

{
  "event_id": "string (optional, auto-generated if missing)",
  "timestamp": "string (optional, ISO 8601, defaults to now)",
  "amount": "number (required, can be invalid/negative to test validation)",
  "currency": "string (optional, defaults to INR)",
  "customer_id": "string (optional)",
  "payment_method": "string (optional, can be invalid to test validation)",
  "error_code": "string (optional, can be invalid)",
  "error_reason": "string (optional)",
  "hour_of_day": "integer (optional, 0-23, can be invalid)",
  "prior_retry_count": "integer (optional, 0-5, can be invalid)",
  "metadata": "object (optional, ignored by pipeline)"
}
```

### Response (same shape as `events_stream.json` item)

```json
{
  "event_id": "string",
  "timestamp": "string (ISO 8601)",
  "amount": "number",
  "payment_method": "string",
  "error_code": "string",
  "p_recoverable": "number (0.0-1.0)",
  "decision": "string | null",
  "reason": "string | null",
  "explanation": "object",
  "fallback_flag": "boolean",
  "outcome": "string (recovered | correctly_stopped | escalated_unresolved | no_action | validation_error)"
}
```

### Error Response (HTTP 400)

```json
{
  "error": "string",
  "details": "object"
}
```

### Implementation Notes

- Wraps `pipeline.process_event()` directly
- No persistence — runs in-memory, returns outcome
- Idempotency store is per-process (resets on restart)
- Handles malformed input gracefully via validation fallback logic

---

## Versioning

- **Contract version** increments on breaking changes (field rename, type change, enum value add/remove).
- **File version** embedded in each file's metadata if needed.
- Frontend should validate against this contract at build time.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-08-25 | Initial contract |