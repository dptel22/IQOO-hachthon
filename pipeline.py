"""
pipeline.py — Orchestration Pipeline (Phase 3)
==============================================

Single entry point: process_event(event: dict) -> dict

Internal helpers (module-private, prefixed _):
  _validate_event
  _check_idempotency
  _extract_features
  _predict_recoverability
  _decide_action
  _execute_intervention
  _escalate_to_manual_review
  _write_audit_log

State maintained per-event during simulation:
  - live_retry_count: pipeline's own counter (separate from prior_retry_count)
  - infra_retry_count: separate counter for infra timeouts (Failure Handler #2)
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import detection

# ── Audit Log Configuration ──────────────────────────────────────────────────

AUDIT_LOG_PATH = "logs/audit_log.jsonl"

# Global idempotency store (in-memory for this run)
_idempotency_store: dict[str, dict] = {}


# ── Helper: Write Audit Record ────────────────────────────────────────────────

def _write_audit_log(record: dict) -> None:
    """Append one JSON line to the audit log."""
    record["audit_timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Helper: Simulated Idempotency Key ────────────────────────────────────────

def _make_idempotency_key(event_id: str, attempt_number: int, action_type: str) -> str:
    """SHA256(event_id + str(attempt_number) + action_type)"""
    raw = f"{event_id}{attempt_number}{action_type}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Internal Helper 1: Validate Event ────────────────────────────────────────

_REQUIRED_FIELDS = [
    "event_id", "timestamp", "amount", "currency",
    "customer_id", "payment_method", "error_code",
    "error_reason", "hour_of_day", "prior_retry_count",
]

def _validate_event(event: dict) -> tuple[bool, Optional[str]]:
    """
    Check required fields present and well-formed.
    Returns (is_valid, error_message).
    On failure: writes validation_error audit record.
    """
    for field in _REQUIRED_FIELDS:
        if field not in event:
            _write_audit_log({
                "event_id": event.get("event_id", "unknown"),
                "module": "validation",
                "decision": "validation_error",
                "reason": f"missing_field_{field}",
                "input": {"event": {k: v for k, v in event.items() if k != "metadata"}},
                "output": {},
                "explanation": {},
                "idempotency_key": "",
                "fallback_flag": False,
            })
            return False, f"Missing required field: {field}"

    # Basic type checks
    try:
        amount = float(event["amount"])
        if amount <= 0:
            raise ValueError
    except (TypeError, ValueError):
        _write_audit_log({
            "event_id": event.get("event_id", "unknown"),
            "module": "validation",
            "decision": "validation_error",
            "reason": "invalid_amount",
            "input": {"event": {k: v for k, v in event.items() if k != "metadata"}},
            "output": {},
            "explanation": {},
            "idempotency_key": "",
            "fallback_flag": False,
        })
        return False, "Invalid amount"

    return True, None


# ── Internal Helper 2: Check Idempotency ────────────────────────────────────

def _check_idempotency(event: dict, attempt_number: int, action_type: str) -> Optional[dict]:
    """
    Compute idempotency key. If already executed, return prior recorded outcome.
    Otherwise return None (proceed to execute).
    """
    event_id = event["event_id"]
    key = _make_idempotency_key(event_id, attempt_number, action_type)

    if key in _idempotency_store:
        # Return cached outcome
        cached = _idempotency_store[key]
        _write_audit_log({
            "event_id": event_id,
            "module": "idempotency",
            "decision": cached["decision"],
            "reason": "idempotent_replay",
            "input": {"event_id": event_id, "attempt_number": attempt_number, "action_type": action_type},
            "output": cached["output"],
            "explanation": {},
            "idempotency_key": key,
            "fallback_flag": False,
        })
        return cached["output"]

    return None


def _record_idempotent_outcome(event_id: str, attempt_number: int, action_type: str,
                                decision: str, output: dict) -> None:
    """Record the outcome of an action for idempotency replay."""
    key = _make_idempotency_key(event_id, attempt_number, action_type)
    _idempotency_store[key] = {"decision": decision, "output": output}


# ── Internal Helper 3: Extract Features ──────────────────────────────────────

# Reuse detection's extract_features for consistency
from detection import extract_features as _detect_extract_features

def _extract_features(event: dict) -> tuple[dict, bool]:
    """
    Build feature vector per Phase 2 schema, applying fallback logic.
    Returns (features_dict, fallback_flag).
    This is Failure Handler #1: missing/malformed → neutral fallback, flagged, continue.
    """
    # detection.extract_features returns (np_array, fallback_flag)
    x, fallback_flag = _detect_extract_features(event)
    # Convert to dict for audit log
    features_dict = {name: float(x[i]) for i, name in enumerate(detection.FEATURE_NAMES)}
    return features_dict, fallback_flag


# ── Internal Helper 4: Predict Recoverability ────────────────────────────────

def _predict_recoverability(features: dict) -> dict:
    """
    Call the trained Detection model.
    If model unavailable: default to decision="no_action", reason="model_unavailable"
    EXCEPT if amount > 10000, still escalate regardless.
    """
    try:
        # Create a mock event for detection.predict_recoverability
        # It expects the original event format, so we reconstruct what it needs
        # But actually, detection.predict_recoverability takes a full event dict.
        # Let's call it directly with a reconstructed event containing just what's needed.
        import pickle
        with open("detector_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("detector_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Build feature array in correct order
        x = np.array([features[name] for name in detection.FEATURE_NAMES])
        x_scaled = scaler.transform(x.reshape(1, -1))
        p_recoverable = float(model.predict_proba(x_scaled)[0, 1])

        # Explanation: standardized feature * coefficient
        explanation = {}
        for i, name in enumerate(detection.FEATURE_NAMES):
            explanation[name] = round(float(x_scaled[0, i] * model.coef_[0, i]), 4)

        return {
            "p_recoverable": round(p_recoverable, 4),
            "explanation": explanation,
            "model_available": True,
        }

    except Exception as e:
        # Model unavailable — fallback
        return {
            "p_recoverable": 0.5,  # neutral
            "explanation": {},
            "model_available": False,
            "error": str(e),
        }


# Need numpy for the above
import numpy as np


# ── Internal Helper 5: Decide Action ────────────────────────────────────────

def _decide_action(p_recoverable: float, event: dict, live_retry_count: int) -> tuple[str, str]:
    """
    Decision logic (locked rules):
      if amount > 10000:           escalate, "amount_above_threshold"
      elif live_retry_count >= 3:  escalate, "max_retries_exhausted"
      elif p_recoverable >= 0.47:  retry,    "risk_score_above_threshold"
      else:                        no_action,"risk_score_below_threshold"
    """
    amount = float(event["amount"])

    if amount > 10000:
        return "escalate", "amount_above_threshold"
    elif live_retry_count >= 3:
        return "escalate", "max_retries_exhausted"
    elif p_recoverable >= 0.47:
        return "retry", "risk_score_above_threshold"
    else:
        return "no_action", "risk_score_below_threshold"


# ── Internal Helper 6: Execute Intervention ──────────────────────────────────

# Success probabilities per error category for retry action (from Phase 1 logic)
RETRY_SUCCESS_PROB = {
    "temporary":          0.65,
    "insufficient_funds": 0.35,
    "card_declined":      0.15,
    "auth_failed":        0.10,
    "expired_card":       0.02,
    "invalid_request":    0.00,
    "unknown":            0.10,
}

# Map error code to category (reuse from detection)
ERROR_CATEGORY = detection.ERROR_CATEGORY

def _execute_intervention(event: dict, action_type: str, idempotency_key: str,
                          infra_retry_count: int = 0) -> dict:
    """
    Simulate the action outcome. Returns dict with outcome details.
    Failure Handler #2: if simulated call "times out", retry the call itself
    up to 2 times with exponential backoff (500ms -> 1500ms) using a SEPARATE
    infra_retry_count — never added to live_retry_count.
    """
    max_infra_retries = 2
    base_delay = 0.5  # seconds

    # Check for simulated timeout (deterministic based on event hash for reproducibility)
    # In reality this would be a real API call that could time out.
    # We'll simulate: 5% chance of timeout on first attempt, decreasing on retries.
    timeout_chance = 0.05 * (0.5 ** infra_retry_count)

    # Deterministic "random" based on event_id + attempt
    event_hash = hash(event["event_id"] + action_type + str(infra_retry_count))
    rng_val = (event_hash % 10000) / 10000.0

    if rng_val < timeout_chance and infra_retry_count < max_infra_retries:
        # Simulated timeout — retry the infra call itself
        delay = base_delay * (2 ** infra_retry_count)
        time.sleep(min(delay, 0.01))  # small sleep for realism, capped
        return _execute_intervention(event, action_type, idempotency_key,
                                      infra_retry_count + 1)

    # If still timing out after max infra retries
    if rng_val < timeout_chance:
        _write_audit_log({
            "event_id": event["event_id"],
            "module": "execution",
            "decision": "execution_error",
            "reason": "infra_timeout_exhausted",
            "input": {"event_id": event["event_id"], "action_type": action_type,
                      "infra_retry_count": infra_retry_count},
            "output": {"success": False, "error": "infra_timeout"},
            "explanation": {},
            "idempotency_key": idempotency_key,
            "fallback_flag": False,
        })
        return {"success": False, "error": "infra_timeout", "routed_to_manual_review": True}

    # Normal execution — determine success based on error category
    category = ERROR_CATEGORY.get(event["error_code"], "unknown")
    success_prob = RETRY_SUCCESS_PROB[category]

    # Deterministic success based on event hash
    success_hash = hash(event["event_id"] + action_type + "success")
    success_rng = (success_hash % 10000) / 10000.0
    success = success_rng < success_prob

    outcome = {
        "success": success,
        "error": None if success else f"retry_failed_{category}",
        "action_type": action_type,
    }

    _write_audit_log({
        "event_id": event["event_id"],
        "module": "execution",
        "decision": "retry" if action_type == "retry" else action_type,
        "reason": "executed",
        "input": {"event_id": event["event_id"], "action_type": action_type,
                  "category": category, "success_prob": success_prob},
        "output": outcome,
        "explanation": {},
        "idempotency_key": idempotency_key,
        "fallback_flag": False,
    })

    return outcome


# ── Internal Helper 7: Escalate to Manual Review ────────────────────────────

def _escalate_to_manual_review(event: dict, reason: str) -> dict:
    """
    Write escalation audit record. Returns escalation outcome.
    Terminal state: escalated_unresolved (if no human resolution in batch window).
    """
    event_id = event["event_id"]
    amount = float(event["amount"])

    _write_audit_log({
        "event_id": event_id,
        "module": "escalation",
        "decision": "escalate",
        "reason": reason,
        "input": {"event_id": event_id, "amount": amount,
                  "error_code": event["error_code"]},
        "output": {"terminal_state": "escalated_unresolved", "escalation_reason": reason},
        "explanation": {},
        "idempotency_key": "",
        "fallback_flag": False,
    })

    return {
        "terminal_state": "escalated_unresolved",
        "escalation_reason": reason,
        "recovered": False,
    }


# ── Main Entry Point: process_event ──────────────────────────────────────────

def process_event(event: dict) -> dict:
    """
    Single orchestration function. Processes one payment failure event
    through the full recovery pipeline. Returns final outcome dict.
    """
    event_id = event["event_id"]
    amount = float(event["amount"])

    # ── Step 1: Validate ─────────────────────────────────────────────────
    is_valid, error = _validate_event(event)
    if not is_valid:
        return {"event_id": event_id, "final_state": "validation_error", "recovered": False, "error": error}

    # Initialize live retry counter from dataset's prior_retry_count
    # This is the STATIC TRAINING FEATURE (prior_retry_count) seeding the
    # pipeline's own LIVE counter. They are separate concepts — do not conflate.
    live_retry_count = int(event.get("prior_retry_count", 0))

    # ── Step 2: Feature Extraction (Failure Handler #1) ─────────────────
    features, fallback_flag = _extract_features(event)

    # ── Step 3: Predict Recoverability ──────────────────────────────────
    pred_result = _predict_recoverability(features)
    p_recoverable = pred_result["p_recoverable"]
    model_available = pred_result.get("model_available", True)

    # Write detection audit record
    _write_audit_log({
        "event_id": event_id,
        "module": "detection",
        "decision": "score_computed",
        "reason": "model_inference" if model_available else "model_unavailable_fallback",
        "input": {"features": features, "fallback_flag": fallback_flag},
        "output": {"p_recoverable": p_recoverable, "model_available": model_available},
        "explanation": pred_result.get("explanation", {}),
        "idempotency_key": "",
        "fallback_flag": fallback_flag,
    })

    # ── Step 4: Decision Loop (with mid-sequence re-check) ───────────────
    while True:
        # Check idempotency for this decision point
        decision, reason = _decide_action(p_recoverable, event, live_retry_count)

        if decision == "no_action":
            _write_audit_log({
                "event_id": event_id,
                "module": "decision",
                "decision": "no_action",
                "reason": reason,
                "input": {"p_recoverable": p_recoverable, "live_retry_count": live_retry_count,
                          "amount": amount},
                "output": {"terminal_state": "no_action", "recovered": False},
                "explanation": pred_result.get("explanation", {}),
                "idempotency_key": "",
                "fallback_flag": fallback_flag,
            })
            return {
                "event_id": event_id,
                "final_state": "no_action",
                "recovered": False,
                "live_retry_count": live_retry_count,
            }

        elif decision == "escalate":
            # Escalation is terminal — write audit and return
            result = _escalate_to_manual_review(event, reason)
            return {
                "event_id": event_id,
                "final_state": result["terminal_state"],
                "recovered": False,
                "escalation_reason": reason,
                "live_retry_count": live_retry_count,
            }

        elif decision == "retry":
            # Check idempotency for this retry attempt
            attempt_number = live_retry_count + 1  # 1-indexed attempt
            action_type = "retry"
            cached = _check_idempotency(event, attempt_number, action_type)
            if cached is not None:
                # Idempotent replay — return cached outcome
                return {
                    "event_id": event_id,
                    "final_state": "recovered" if cached["success"] else "retry_failed",
                    "recovered": cached["success"],
                    "live_retry_count": live_retry_count,
                }

            # Execute the retry
            idempotency_key = _make_idempotency_key(event_id, attempt_number, action_type)
            execution_result = _execute_intervention(event, action_type, idempotency_key)

            # Record for idempotency
            _record_idempotent_outcome(event_id, attempt_number, action_type,
                                        "retry", execution_result)

            if execution_result.get("routed_to_manual_review"):
                return {
                    "event_id": event_id,
                    "final_state": "escalated_unresolved",
                    "recovered": False,
                    "escalation_reason": "infra_timeout_exhausted",
                    "live_retry_count": live_retry_count,
                }

            if execution_result["success"]:
                # SUCCESS — mid-sequence re-check: stop immediately
                _write_audit_log({
                    "event_id": event_id,
                    "module": "decision",
                    "decision": "retry",
                    "reason": "success_on_retry",
                    "input": {"p_recoverable": p_recoverable, "live_retry_count": live_retry_count,
                              "attempt_number": attempt_number},
                    "output": {"terminal_state": "recovered", "recovered": True},
                    "explanation": pred_result.get("explanation", {}),
                    "idempotency_key": idempotency_key,
                    "fallback_flag": fallback_flag,
                })
                return {
                    "event_id": event_id,
                    "final_state": "recovered",
                    "recovered": True,
                    "recovery_attempt": attempt_number,
                    "live_retry_count": live_retry_count,
                }

            # FAILURE — increment live retry counter and loop (mid-sequence re-check)
            live_retry_count += 1

            _write_audit_log({
                "event_id": event_id,
                "module": "decision",
                "decision": "retry",
                "reason": "retry_failed_continue",
                "input": {"p_recoverable": p_recoverable, "live_retry_count": live_retry_count - 1,
                          "attempt_number": attempt_number},
                "output": {"terminal_state": "pending_next_retry", "recovered": False,
                           "live_retry_count": live_retry_count},
                "explanation": pred_result.get("explanation", {}),
                "idempotency_key": idempotency_key,
                "fallback_flag": fallback_flag,
            })

            # Loop continues — _decide_action will be called again with updated live_retry_count
            # If live_retry_count now >= 3, next iteration will escalate
            continue


# ── Batch Runner ──────────────────────────────────────────────────────────────

def run_batch(input_path: str = "data/synthetic_payment_failures.jsonl") -> list[dict]:
    """Run pipeline on all events in input file. Returns list of outcomes."""
    outcomes = []
    with open(input_path, "r") as f:
        for line in f:
            event = json.loads(line)
            outcome = process_event(event)
            outcomes.append(outcome)
    return outcomes


# ── Demo / CLI ─────────────────────────────────────────────────────────────────

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        print("Running full batch...")
        outcomes = run_batch()
        print(f"Processed {len(outcomes)} events.")
        # Summary
        recovered = sum(1 for o in outcomes if o.get("recovered"))
        escalated = sum(1 for o in outcomes if o.get("final_state", "").startswith("escalated"))
        no_action = sum(1 for o in outcomes if o.get("final_state") == "no_action")
        validation_err = sum(1 for o in outcomes if o.get("final_state") == "validation_error")
        print(f"  Recovered:      {recovered}")
        print(f"  Escalated:      {escalated}")
        print(f"  No action:      {no_action}")
        print(f"  Validation err: {validation_err}")
    else:
        # Quick demo on first 5 events
        print("Demo on first 5 events:")
        with open("data/synthetic_payment_failures.jsonl", "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                event = json.loads(line)
                print(f"\n--- Event {i+1}: {event['event_id']} ---")
                outcome = process_event(event)
                print(f"  Final state: {outcome['final_state']}")
                print(f"  Recovered:   {outcome.get('recovered', False)}")
                print(f"  Live retries: {outcome.get('live_retry_count', 0)}")


if __name__ == "__main__":
    main()