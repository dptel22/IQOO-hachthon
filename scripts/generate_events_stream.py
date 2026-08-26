#!/usr/bin/env python3
"""
Generate events_stream.json from audit log and dataset.
Each event includes: event_id, timestamp, amount, payment_method, error_code,
p_recoverable, decision, reason, explanation (feature→contribution),
fallback_flag, final outcome (recovered / correctly_stopped / escalated_unresolved / no_action)
"""

import json
from collections import defaultdict

AUDIT_LOG_PATH = "logs/audit_log.jsonl"
DATASET_PATH = "data/synthetic_payment_failures.jsonl"
OUTPUT_PATH = "data/events_stream.json"

def load_audit_log():
    """Parse audit log and organize by event_id."""
    events = defaultdict(list)
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            eid = record["event_id"]
            events[eid].append(record)
    return events

def load_dataset():
    """Load original dataset for event metadata."""
    events = {}
    with open(DATASET_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            events[ev["event_id"]] = ev
    return events

def get_final_outcome(audit_records, dataset_event):
    """
    Determine the final outcome for an event from its audit records.
    Returns one of: 'recovered', 'correctly_stopped', 'escalated_unresolved', 'no_action', 'validation_error'
    """
    # Find the last record with a terminal_state in output
    terminal_state = None
    reason = None
    escalation_reason = None
    recovery_attempt = None
    live_retry_count = 0
    p_recoverable = 0.0
    decision = None
    explanation = {}
    fallback_flag = False

    for rec in reversed(audit_records):
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

    # Also check for p_recoverable and explanation from detection module
    for rec in audit_records:
        if rec.get("module") == "detection" and rec.get("decision") == "score_computed":
            p_recoverable = rec.get("output", {}).get("p_recoverable", 0.0)
            explanation = rec.get("explanation", {})
            fallback_flag = rec.get("fallback_flag", False)
            break

    # Determine decision and reason from the decision module records
    for rec in audit_records:
        if rec.get("module") == "decision":
            decision = rec.get("decision")
            reason = rec.get("reason")
            # Use the last decision record's reason

    # Map terminal_state to outcome
    if terminal_state == "recovered":
        return "recovered"
    elif terminal_state == "no_action":
        return "no_action"
    elif terminal_state == "escalated_unresolved":
        # Check if it was correctly stopped (amount threshold or max retries)
        if escalation_reason in ("amount_above_threshold", "max_retries_exhausted"):
            return "correctly_stopped"
        return "escalated_unresolved"
    elif terminal_state == "validation_error":
        return "validation_error"
    elif terminal_state == "pending_next_retry":
        # This shouldn't be final, but if it is, check what happened
        return "escalated_unresolved"
    elif terminal_state is None:
        # Fallback: check decision field
        for rec in reversed(audit_records):
            if rec.get("decision") in ("recovered", "no_action", "escalate", "validation_error"):
                if rec["decision"] == "recovered":
                    return "recovered"
                elif rec["decision"] == "no_action":
                    return "no_action"
                elif rec["decision"] == "escalate":
                    if rec.get("reason") in ("amount_above_threshold", "max_retries_exhausted"):
                        return "correctly_stopped"
                    return "escalated_unresolved"
                elif rec["decision"] == "validation_error":
                    return "validation_error"

    return "unknown"

def main():
    print("Loading audit log...")
    audit_events = load_audit_log()
    print(f"  {len(audit_events)} events in audit log")

    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} events in dataset")

    output_events = []

    for event_id, audit_records in audit_events.items():
        ds_event = dataset.get(event_id, {})

        # Get base event info
        timestamp = ds_event.get("timestamp", "")
        amount = ds_event.get("amount", 0)
        payment_method = ds_event.get("payment_method", "")
        error_code = ds_event.get("error_code", "")

        # Get detection info
        p_recoverable = 0.0
        explanation = {}
        fallback_flag = False

        for rec in audit_records:
            if rec.get("module") == "detection" and rec.get("decision") == "score_computed":
                p_recoverable = rec.get("output", {}).get("p_recoverable", 0.0)
                explanation = rec.get("explanation", {})
                fallback_flag = rec.get("fallback_flag", False)
                break

        # Get final decision/reason
        decision = None
        reason = None
        for rec in audit_records:
            if rec.get("module") == "decision":
                decision = rec.get("decision")
                reason = rec.get("reason")

        # Determine final outcome
        outcome = get_final_outcome(audit_records, ds_event)

        # Build output event
        output_event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "amount": amount,
            "payment_method": payment_method,
            "error_code": error_code,
            "p_recoverable": round(p_recoverable, 4),
            "decision": decision,
            "reason": reason,
            "explanation": explanation,
            "fallback_flag": fallback_flag,
            "outcome": outcome
        }

        output_events.append(output_event)

    # Sort by timestamp for consistent ordering
    output_events.sort(key=lambda x: x["timestamp"])

    print(f"Writing {len(output_events)} events to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_events, f, indent=2)

    # Print summary
    outcome_counts = defaultdict(int)
    for ev in output_events:
        outcome_counts[ev["outcome"]] += 1

    print("\nOutcome distribution:")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcome}: {count}")

    print(f"\nDone! Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()