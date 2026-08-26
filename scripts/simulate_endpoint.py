#!/usr/bin/env python3
"""
Live endpoint for failure injection: POST /simulate-event
Wraps pipeline.process_event() and returns same shape as events_stream.json items.
"""

from flask import Flask, request, jsonify
import json
import uuid
from datetime import datetime, timezone

import pipeline

app = Flask(__name__)

@app.route("/simulate-event", methods=["POST"])
def simulate_event():
    """
    Accept a raw event JSON (possibly malformed) and return the processed outcome.
    Response shape matches events_stream.json items.
    """
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {str(e)}"}), 400

    # Build event dict with defaults for optional fields
    event = {
        "event_id": data.get("event_id", f"evt_{uuid.uuid4()}"),
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "amount": data.get("amount", 0),
        "currency": data.get("currency", "INR"),
        "customer_id": data.get("customer_id", f"cust_{uuid.uuid4().hex[:8]}"),
        "payment_method": data.get("payment_method", "card"),
        "error_code": data.get("error_code", "payment_failed"),
        "error_reason": data.get("error_reason", "Simulated failure"),
        "hour_of_day": data.get("hour_of_day", 12),
        "prior_retry_count": data.get("prior_retry_count", 0),
        # metadata is ignored by pipeline
    }

    # Run through pipeline
    outcome = pipeline.process_event(event)

    # Build response matching events_stream.json shape
    # We need to extract p_recoverable and explanation from the audit log
    # Since the audit log is written to file, we can read the latest detection record
    p_recoverable = 0.0
    explanation = {}
    fallback_flag = False
    decision = None
    reason = None

    # Read the audit log to get the detection record for this event
    try:
        with open("audit_log.jsonl", "r") as f:
            for line in reversed(f.readlines()):
                rec = json.loads(line)
                if rec.get("event_id") == event["event_id"]:
                    if rec.get("module") == "detection" and rec.get("decision") == "score_computed":
                        p_recoverable = rec.get("output", {}).get("p_recoverable", 0.0)
                        explanation = rec.get("explanation", {})
                        fallback_flag = rec.get("fallback_flag", False)
                        break
                    elif rec.get("module") == "decision":
                        decision = rec.get("decision")
                        reason = rec.get("reason")
    except Exception:
        pass

    # Determine outcome string from pipeline result
    final_state = outcome.get("final_state", "unknown")
    if final_state == "recovered":
        outcome_str = "recovered"
    elif final_state == "no_action":
        outcome_str = "no_action"
    elif final_state == "escalated_unresolved":
        # Check escalation reason
        esc_reason = outcome.get("escalation_reason", "")
        if esc_reason in ("amount_above_threshold", "max_retries_exhausted"):
            outcome_str = "correctly_stopped"
        else:
            outcome_str = "escalated_unresolved"
    elif final_state == "validation_error":
        outcome_str = "validation_error"
    else:
        outcome_str = "unknown"

    response = {
        "event_id": event["event_id"],
        "timestamp": event["timestamp"],
        "amount": event["amount"],
        "payment_method": event["payment_method"],
        "error_code": event["error_code"],
        "p_recoverable": round(p_recoverable, 4),
        "decision": decision,
        "reason": reason,
        "explanation": explanation,
        "fallback_flag": fallback_flag,
        "outcome": outcome_str
    }

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "simulate-endpoint"})


if __name__ == "__main__":
    print("Starting failure injection endpoint on http://localhost:8080")
    print("POST /simulate-event - simulate a payment failure event")
    print("GET  /health - health check")
    app.run(host="0.0.0.0", port=8080, debug=False)