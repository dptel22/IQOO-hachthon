"""
measure.py — Measurement Report (Phase 5)
==========================================

Computes and reports THREE separate metric sections:
  A. Detection metrics (model quality, from Phase 2 holdout)
  B. Pipeline / business metrics (from batch run audit log)
  C. Sanity check: all events accounted for

Does NOT compute ROI — explicitly forbidden by spec.
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

AUDIT_LOG_PATH = "logs/audit_log.jsonl"
DATASET_PATH = "data/synthetic_payment_failures.jsonl"
TEST_DATASET_PATH = "data/synthetic_payment_failures_test.jsonl"
DETECTION_REPORT_PATH = "reports/measurement_report.md"
DETECTION_JSON_PATH = "reports/measurement_report.json"

MODEL_PATH = "models/detector_model.pkl"
SCALER_PATH = "models/detector_scaler.pkl"

# Dummy cost assumptions (explicitly stated as assumptions)
FP_COST = 10.0       # INR10 wasted-intervention cost per false positive
FN_COST_MULTIPLIER = 1.0  # Lost amount = full event amount per false negative


# ── Load Data ──────────────────────────────────────────────────────────────────

def load_audit_log():
    """Parse audit log and organize by event_id."""
    events = {}
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            eid = record["event_id"]
            if eid not in events:
                events[eid] = []
            events[eid].append(record)
    return events


def load_dataset():
    """Load original dataset for amount lookup."""
    events = {}
    with open(DATASET_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            events[ev["event_id"]] = ev
    return events


# ── Section A: Detection Metrics (from Phase 2 holdout) ────────────────────────

def compute_detection_metrics():
    """Recompute holdout metrics from detection.py training."""
    import pickle
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    import detection

    # Load model and scaler
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load test data
    X_test = []
    y_test = []
    with open(TEST_DATASET_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            x, _ = detection.extract_features(ev)
            X_test.append(x)
            y_test.append(ev["recovered"])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "test_size": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }


# ── Section B: Pipeline / Business Metrics ─────────────────────────────────────

def compute_pipeline_metrics(audit_events, dataset):
    """Compute pipeline metrics from audit log."""
    # Track final state per event by finding the last decision record
    final_states = {}
    for eid, records in audit_events.items():
        # Find the last record with a terminal_state in output
        terminal_state = None
        recovery_attempt = None
        escalation_reason = None
        live_retry_count = 0

        for rec in reversed(records):
            output = rec.get("output", {})
            if "terminal_state" in output:
                terminal_state = output["terminal_state"]
                if "recovery_attempt" in output:
                    recovery_attempt = output["recovery_attempt"]
                if "escalation_reason" in output:
                    escalation_reason = output["escalation_reason"]
                if "live_retry_count" in output:
                    live_retry_count = output["live_retry_count"]
                break

        # Fallback: check decision field
        if terminal_state is None:
            for rec in reversed(records):
                if rec.get("decision") in ("recovered", "no_action", "escalate", "validation_error"):
                    terminal_state = rec["decision"]
                    break

        if terminal_state is None:
            terminal_state = "unknown"

        final_states[eid] = {
            "terminal_state": terminal_state,
            "recovery_attempt": recovery_attempt,
            "escalation_reason": escalation_reason,
            "live_retry_count": live_retry_count,
        }

    # Counters
    recovered = 0
    recovered_amount = 0.0
    correctly_stopped = 0
    correctly_stopped_amount = 0.0
    escalated_unresolved = 0
    escalated_unresolved_amount = 0.0
    no_action = 0
    no_action_amount = 0.0
    validation_error = 0
    flagged_for_retry = 0  # events where decision was "retry"

    # Also track per-event for cost calculation
    false_positives = 0  # pipeline retried but didn't recover
    false_negatives = 0  # pipeline no_action/escalate but dataset label was recovered=1

    for eid, fs in final_states.items():
        ds_event = dataset.get(eid, {})
        amount = ds_event.get("amount", 0)
        ds_recovered = ds_event.get("recovered", 0)
        state = fs["terminal_state"]

        if state == "recovered":
            recovered += 1
            recovered_amount += amount
            if ds_recovered == 0:
                false_positives += 1  # pipeline recovered but dataset says not recoverable

        elif state == "no_action":
            no_action += 1
            no_action_amount += amount
            if ds_recovered == 1:
                false_negatives += 1  # pipeline gave up but dataset says recoverable

        elif state == "escalated_unresolved":
            escalated_unresolved += 1
            escalated_unresolved_amount += amount
            # Escalation is a "correctly stopped" bucket if it was due to rules
            reason = fs.get("escalation_reason", "")
            if reason in ("amount_above_threshold", "max_retries_exhausted"):
                correctly_stopped += 1
                correctly_stopped_amount += amount
            if ds_recovered == 1:
                false_negatives += 1  # escalated but dataset says recoverable

        elif state == "validation_error":
            validation_error += 1

        # Track "flagged for retry" = any event that had a retry decision
        # Check if any record has decision="retry" with a retry-related reason
        for rec in audit_events[eid]:
            if rec.get("decision") == "retry" and rec.get("reason") in (
                "risk_score_above_threshold", "executed", "success_on_retry", "retry_failed_continue"
            ):
                flagged_for_retry += 1
                break

    # Recovery rate = recovered / flagged_for_retry
    recovery_rate = recovered / flagged_for_retry if flagged_for_retry > 0 else 0.0

    # Cost calculation (dummy assumptions)
    fp_cost_total = false_positives * FP_COST
    fn_cost_total = 0.0
    for eid, fs in final_states.items():
        ds_event = dataset.get(eid, {})
        if ds_event.get("recovered") == 1 and fs["terminal_state"] in ("no_action", "escalated_unresolved"):
            fn_cost_total += ds_event.get("amount", 0) * FN_COST_MULTIPLIER

    total_cost = fp_cost_total + fn_cost_total

    return {
        "recovered": recovered,
        "recovered_amount": round(recovered_amount, 2),
        "correctly_stopped": correctly_stopped,
        "correctly_stopped_amount": round(correctly_stopped_amount, 2),
        "escalated_unresolved": escalated_unresolved,
        "escalated_unresolved_amount": round(escalated_unresolved_amount, 2),
        "no_action": no_action,
        "no_action_amount": round(no_action_amount, 2),
        "validation_error": validation_error,
        "flagged_for_retry": flagged_for_retry,
        "recovery_rate": round(recovery_rate, 4),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fp_cost": round(fp_cost_total, 2),
        "fn_cost": round(fn_cost_total, 2),
        "total_cost": round(total_cost, 2),
    }


# ── Section C: Sanity Check ────────────────────────────────────────────────────

def sanity_check(audit_events, dataset, pipeline_metrics):
    """Verify 100% of events accounted for, no silent drops."""
    audit_count = len(audit_events)
    dataset_count = len(dataset)
    accounted = sum([
        pipeline_metrics["recovered"],
        pipeline_metrics["correctly_stopped"],
        pipeline_metrics["escalated_unresolved"],
        pipeline_metrics["no_action"],
        pipeline_metrics["validation_error"],
    ])

    # Note: correctly_stopped is a SUBSET of escalated_unresolved (those with valid reasons)
    # So the true partition is:
    # recovered + escalated_unresolved + no_action + validation_error
    true_partition = (
        pipeline_metrics["recovered"] +
        pipeline_metrics["escalated_unresolved"] +
        pipeline_metrics["no_action"] +
        pipeline_metrics["validation_error"]
    )

    issues = []
    if audit_count != dataset_count:
        issues.append(f"Audit log has {audit_count} events, dataset has {dataset_count}")
    if true_partition != dataset_count:
        issues.append(f"Partition sum {true_partition} != dataset count {dataset_count}")

    # Check each event appears in audit log
    missing_in_audit = set(dataset.keys()) - set(audit_events.keys())
    if missing_in_audit:
        issues.append(f"{len(missing_in_audit)} events missing from audit log")

    return {
        "audit_events": audit_count,
        "dataset_events": dataset_count,
        "partition_sum": true_partition,
        "fully_accounted": len(issues) == 0,
        "issues": issues,
    }


# ── Report Generation ──────────────────────────────────────────────────────────

def generate_report(detection_metrics, pipeline_metrics, sanity):
    """Generate markdown and JSON reports."""
    timestamp = datetime.now().isoformat()

    md = f"""# Measurement Report

**Generated:** {timestamp}

---

## A. Detection Metrics (Model Quality — Phase 2 Holdout)

*Measured against dataset's `recovered` ground truth label. Independent of pipeline execution.*

| Metric | Value |
|--------|-------|
| Precision | {detection_metrics['precision']:.4f} |
| Recall | {detection_metrics['recall']:.4f} |
| F1 Score | {detection_metrics['f1']:.4f} |
| Test Set Size | {detection_metrics['test_size']} |
| Positive Rate (in test) | {detection_metrics['positive_rate']:.4f} |

> **Note:** These metrics evaluate the *classifier* in isolation. They are NOT the same as pipeline recovery rate (Section B). The classifier predicts "is this recoverable?" — the pipeline decides "should we retry?" and then executes. Conflating these is a category error.

---

## B. Pipeline / Business Metrics (Batch Run — 500 Events)

*Measured from the audit log of a full batch run through `process_event()`.*

| Metric | Count | Amount (INR) |
|--------|-------|------------|
| **Recovered** | {pipeline_metrics['recovered']} | {pipeline_metrics['recovered_amount']:,.2f} |
| **Correctly Stopped** (rules fired appropriately) | {pipeline_metrics['correctly_stopped']} | {pipeline_metrics['correctly_stopped_amount']:,.2f} |
| **Escalated Unresolved** (no human resolution in batch window) | {pipeline_metrics['escalated_unresolved']} | {pipeline_metrics['escalated_unresolved_amount']:,.2f} |
| **No Action** (score below threshold) | {pipeline_metrics['no_action']} | {pipeline_metrics['no_action_amount']:,.2f} |
| **Validation Error** | {pipeline_metrics['validation_error']} | — |

### Derived Metrics

| Metric | Value |
|--------|-------|
| Flagged for Retry (pipeline attempted retry) | {pipeline_metrics['flagged_for_retry']} |
| Recovery Rate (recovered ÷ flagged) | {pipeline_metrics['recovery_rate']:.4f} |
| Total Amount Recovered | INR{pipeline_metrics['recovered_amount']:,.2f} |

### Cost Analysis (Stated Assumptions — NOT Measured)

> **Assumptions:** False Positive = INR{FP_COST:.0f} wasted-intervention cost. False Negative = lost event amount (1×).

| Cost Component | Value |
|----------------|-------|
| False Positives (retry attempted, didn't recover) | {pipeline_metrics['false_positives']} × INR{FP_COST:.0f} = INR{pipeline_metrics['fp_cost']:,.2f} |
| False Negatives (no retry/escalated but was recoverable) | {pipeline_metrics['false_negatives']} events = INR{pipeline_metrics['fn_cost']:,.2f} |
| **Total Estimated Cost** | **INR{pipeline_metrics['total_cost']:,.2f}** |

> WARNING: These costs use explicit dummy assumptions. They are NOT measured from production data. No ROI computation is performed.

---

## C. Sanity Check

| Check | Result |
|-------|--------|
| Audit events match dataset | {sanity['audit_events']} vs {sanity['dataset_events']} — {'PASS' if sanity['audit_events'] == sanity['dataset_events'] else 'FAIL'} |
| Partition covers 100% of events | {sanity['partition_sum']} / {sanity['dataset_events']} — {'PASS' if sanity['fully_accounted'] else 'FAIL'} |
| No silent drops | {'PASS' if sanity['fully_accounted'] else 'FAIL'} |

"""
    if not sanity['fully_accounted']:
        md += "\n**Issues found:**\n"
        for issue in sanity['issues']:
            md += f"- {issue}\n"

    md += f"""

---

## Summary

- **Detection** (classifier quality): F1 = {detection_metrics['f1']:.4f}
- **Pipeline Recovery** (business outcome): {pipeline_metrics['recovered']} events, INR{pipeline_metrics['recovered_amount']:,.2f}
- **Recovery Rate** (of flagged): {pipeline_metrics['recovery_rate']:.4f}
- **Correctly Stopped** (rules working): {pipeline_metrics['correctly_stopped']} events
- **Escalated Unresolved** (honest accounting): {pipeline_metrics['escalated_unresolved']} events
- **Sanity**: {'ALL CHECKS PASS' if sanity['fully_accounted'] else 'ISSUES DETECTED'}

> These are SEPARATE metrics. Detection F1 != Pipeline Recovery Rate. They measure different things.
"""

    json_report = {
        "generated_at": timestamp,
        "detection_metrics": detection_metrics,
        "pipeline_metrics": pipeline_metrics,
        "sanity_check": sanity,
        "cost_assumptions": {
            "fp_cost_per_event": FP_COST,
            "fn_cost_multiplier": FN_COST_MULTIPLIER,
        },
    }

    return md, json_report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PHASE 5 — Measurement Report")
    print("=" * 65)

    print("\nLoading audit log...")
    audit_events = load_audit_log()
    print(f"  {len(audit_events)} events in audit log")

    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} events in dataset")

    print("\nComputing Section A: Detection Metrics...")
    detection_metrics = compute_detection_metrics()
    print(f"  Precision: {detection_metrics['precision']:.4f}")
    print(f"  Recall:    {detection_metrics['recall']:.4f}")
    print(f"  F1:        {detection_metrics['f1']:.4f}")

    print("\nComputing Section B: Pipeline Metrics...")
    pipeline_metrics = compute_pipeline_metrics(audit_events, dataset)
    print(f"  Recovered: {pipeline_metrics['recovered']} (INR{pipeline_metrics['recovered_amount']:,.2f})")
    print(f"  Correctly Stopped: {pipeline_metrics['correctly_stopped']}")
    print(f"  Escalated Unresolved: {pipeline_metrics['escalated_unresolved']}")
    print(f"  No Action: {pipeline_metrics['no_action']}")
    print(f"  Recovery Rate: {pipeline_metrics['recovery_rate']:.4f}")

    print("\nRunning Section C: Sanity Check...")
    sanity = sanity_check(audit_events, dataset, pipeline_metrics)
    if sanity['fully_accounted']:
        print("  [OK] All events accounted for")
    else:
        print("  [ERROR] ISSUES:")
        for issue in sanity['issues']:
            print(f"    - {issue}")

    print("\nGenerating reports...")
    md, json_report = generate_report(detection_metrics, pipeline_metrics, sanity)

    with open(DETECTION_REPORT_PATH, "w") as f:
        f.write(md)
    with open(DETECTION_JSON_PATH, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"  Markdown: {DETECTION_REPORT_PATH}")
    print(f"  JSON:     {DETECTION_JSON_PATH}")

    print("\n" + "=" * 65)
    print("MEASUREMENT REPORT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()