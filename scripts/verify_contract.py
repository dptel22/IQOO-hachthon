"""
Final verification that all output files match the API_CONTRACT.md
"""
import json
from collections import Counter

def verify_events_stream():
    print("=" * 60)
    print("VERIFYING: events_stream.json")
    print("=" * 60)
    with open('events_stream.json', 'r') as f:
        events = json.load(f)

    print(f"Total events: {len(events)}")

    required = ['event_id', 'timestamp', 'amount', 'payment_method', 'error_code',
                'p_recoverable', 'decision', 'reason', 'explanation', 'fallback_flag', 'outcome']

    missing = []
    for i, e in enumerate(events):
        for field in required:
            if field not in e:
                missing.append((i, field))

    if missing:
        print('MISSING FIELDS:')
        for idx, field in missing[:10]:
            print('  Event', idx, events[idx].get('event_id'), ':', 'missing', field)
        return False
    else:
        print('PASS: All required fields present in all events')

    # Check types
    type_errors = []
    for i, e in enumerate(events):
        if not isinstance(e['event_id'], str):
            type_errors.append((i, 'event_id', 'string'))
        if not isinstance(e['timestamp'], str):
            type_errors.append((i, 'timestamp', 'string'))
        if not isinstance(e['amount'], (int, float)):
            type_errors.append((i, 'amount', 'number'))
        if not isinstance(e['payment_method'], str):
            type_errors.append((i, 'payment_method', 'string'))
        if not isinstance(e['error_code'], str):
            type_errors.append((i, 'error_code', 'string'))
        if not isinstance(e['p_recoverable'], (int, float)):
            type_errors.append((i, 'p_recoverable', 'number'))
        if e['decision'] is not None and not isinstance(e['decision'], str):
            type_errors.append((i, 'decision', 'string or null'))
        if e['reason'] is not None and not isinstance(e['reason'], str):
            type_errors.append((i, 'reason', 'string or null'))
        if not isinstance(e['explanation'], dict):
            type_errors.append((i, 'explanation', 'object'))
        if not isinstance(e['fallback_flag'], bool):
            type_errors.append((i, 'fallback_flag', 'boolean'))
        if not isinstance(e['outcome'], str):
            type_errors.append((i, 'outcome', 'string'))

    if type_errors:
        print('TYPE ERRORS:')
        for idx, field, expected in type_errors[:10]:
            print(f'  Event {idx}: {field} expected {expected}')
        return False
    else:
        print('PASS: All fields have correct types')

    # Check outcome enum
    valid_outcomes = {'recovered', 'correctly_stopped', 'escalated_unresolved', 'no_action', 'validation_error'}
    invalid_outcomes = set(e['outcome'] for e in events) - valid_outcomes
    if invalid_outcomes:
        print(f'FAIL: Invalid outcome values: {invalid_outcomes}')
        return False
    else:
        print('PASS: All outcome values are valid enum')

    # Check decision enum
    valid_decisions = {'retry', 'no_action', 'escalate', None}
    invalid_decisions = set(e['decision'] for e in events) - valid_decisions
    if invalid_decisions:
        print(f'FAIL: Invalid decision values: {invalid_decisions}')
        return False
    else:
        print('PASS: All decision values are valid enum')

    # Check p_recoverable range
    p_vals = [e['p_recoverable'] for e in events]
    if min(p_vals) < 0 or max(p_vals) > 1:
        print(f'FAIL: p_recoverable out of range [0,1]: {min(p_vals)} - {max(p_vals)}')
        return False
    else:
        print(f'PASS: p_recoverable in range [0,1]: {min(p_vals):.4f} - {max(p_vals):.4f}')

    # Print distributions
    outcomes = Counter(e['outcome'] for e in events)
    print('\nOutcome distribution:')
    for o, c in outcomes.most_common():
        print(f'  {o}: {c}')

    decisions = Counter(str(e['decision']) for e in events)
    print('\nDecision distribution:')
    for d, c in decisions.most_common():
        print(f'  {d}: {c}')

    fallback = Counter(e['fallback_flag'] for e in events)
    print('\nFallback flag:')
    for f, c in fallback.most_common():
        print(f'  {f}: {c}')

    return True


def verify_model_metadata():
    print("\n" + "=" * 60)
    print("VERIFYING: model_metadata.json")
    print("=" * 60)
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)

    required_top = ['model_name', 'version', 'description', 'features', 'feature_descriptions',
                    'coefficients', 'intercept', 'threshold', 'class_weight', 'metrics', 'scaler']
    for field in required_top:
        if field not in meta:
            print(f'FAIL: Missing top-level field: {field}')
            return False
    print('PASS: All top-level fields present')

    # Check features count
    if len(meta['features']) != 18:
        print(f'FAIL: Expected 18 features, got {len(meta["features"])}')
        return False
    print(f'PASS: 18 features present')

    # Check scaler
    if meta['scaler']['type'] != 'StandardScaler':
        print(f'FAIL: Expected StandardScaler, got {meta["scaler"]["type"]}')
        return False
    if len(meta['scaler']['mean']) != 18 or len(meta['scaler']['scale']) != 18:
        print(f'FAIL: Scaler arrays should have 18 elements')
        return False
    print('PASS: StandardScaler with 18 mean/scale values')

    # Check threshold
    if meta['threshold'] != 0.47:
        print(f'FAIL: Expected threshold 0.47, got {meta["threshold"]}')
        return False
    print('PASS: Threshold = 0.47')

    # Check class_weight
    if meta['class_weight'] != {'0': 1, '1': 3}:
        print(f'FAIL: Expected class_weight {{0: 1, 1: 3}}, got {meta["class_weight"]}')
        return False
    print('PASS: class_weight = {0: 1, 1: 3}')

    return True


def verify_measurement_report():
    print("\n" + "=" * 60)
    print("VERIFYING: measurement_report.json")
    print("=" * 60)
    with open('measurement_report.json', 'r') as f:
        report = json.load(f)

    required_sections = ['generated_at', 'detection_metrics', 'pipeline_metrics', 'sanity_check', 'cost_assumptions']
    for section in required_sections:
        if section not in report:
            print(f'FAIL: Missing section: {section}')
            return False
    print('PASS: All required sections present')

    # Check detection_metrics
    dm = report['detection_metrics']
    dm_fields = ['precision', 'recall', 'f1', 'test_size', 'positive_rate']
    for field in dm_fields:
        if field not in dm:
            print(f'FAIL: Missing detection_metrics field: {field}')
            return False
    print('PASS: detection_metrics complete')

    # Check pipeline_metrics
    pm = report['pipeline_metrics']
    pm_fields = ['recovered', 'recovered_amount', 'correctly_stopped', 'correctly_stopped_amount',
                 'escalated_unresolved', 'escalated_unresolved_amount', 'no_action', 'no_action_amount',
                 'validation_error', 'flagged_for_retry', 'recovery_rate', 'false_positives',
                 'false_negatives', 'fp_cost', 'fn_cost', 'total_cost']
    for field in pm_fields:
        if field not in pm:
            print(f'FAIL: Missing pipeline_metrics field: {field}')
            return False
    print('PASS: pipeline_metrics complete')

    # Check sanity_check
    sc = report['sanity_check']
    sc_fields = ['audit_events', 'dataset_events', 'partition_sum', 'fully_accounted', 'issues']
    for field in sc_fields:
        if field not in sc:
            print(f'FAIL: Missing sanity_check field: {field}')
            return False
    if not sc['fully_accounted']:
        print('FAIL: fully_accounted is false')
        return False
    print('PASS: sanity_check complete and fully_accounted=true')

    print(f"  Detection F1: {dm['f1']:.4f}")
    print(f"  Pipeline recovered: {pm['recovered']}")
    print(f"  Recovery rate: {pm['recovery_rate']:.4f}")

    return True


def verify_audit_log():
    print("\n" + "=" * 60)
    print("VERIFYING: audit_log.jsonl")
    print("=" * 60)

    event_ids = set()
    modules = Counter()
    decisions = Counter()
    line_count = 0

    with open('audit_log.jsonl', 'r') as f:
        for line in f:
            line_count += 1
            try:
                rec = json.loads(line)
                if 'event_id' in rec:
                    event_ids.add(rec['event_id'])
                if 'module' in rec:
                    modules[rec['module']] += 1
                if 'decision' in rec:
                    decisions[rec['decision']] += 1
            except json.JSONDecodeError:
                pass

    print(f"Total lines: {line_count}")
    print(f"Unique event_ids: {len(event_ids)}")
    print("\nModule distribution:")
    for m, c in modules.most_common():
        print(f'  {m}: {c}')
    print("\nDecision distribution:")
    for d, c in decisions.most_common():
        print(f'  {d}: {c}')

    if len(event_ids) != 500:
        print(f'FAIL: Expected 500 unique events, got {len(event_ids)}')
        return False
    print('PASS: 500 unique events in audit log')

    return True


def verify_simulate_endpoint():
    print("\n" + "=" * 60)
    print("VERIFYING: simulate_endpoint.py (live endpoint)")
    print("=" * 60)

    with open('simulate_endpoint.py', 'r') as f:
        code = f.read()

    checks = [
        ('Flask import', 'from flask import Flask'),
        ('POST /simulate-event', '@app.route("/simulate-event", methods=["POST"])'),
        ('request.get_json', 'request.get_json()'),
        ('pipeline.process_event', 'pipeline.process_event'),
        ('jsonify response', 'jsonify('),
        ('health endpoint', '/health'),
    ]

    all_pass = True
    for name, pattern in checks:
        if pattern in code:
            print(f'PASS: {name}')
        else:
            print(f'FAIL: {name} - pattern not found')
            all_pass = False

    # Check response shape matches events_stream.json
    if '"event_id"' in code and '"p_recoverable"' in code and '"outcome"' in code:
        print('PASS: Response shape includes key fields')
    else:
        print('FAIL: Response shape missing key fields')
        all_pass = False

    return all_pass


def verify_api_contract():
    print("\n" + "=" * 60)
    print("VERIFYING: API_CONTRACT.md")
    print("=" * 60)

    with open('API_CONTRACT.md', 'r') as f:
        contract = f.read()

    # Check all 5 sections documented
    sections = [
        'events_stream.json',
        'audit_log.jsonl',
        'measurement_report.json',
        'model_metadata.json',
        'POST /simulate-event'
    ]

    all_pass = True
    for section in sections:
        if section in contract:
            print(f'PASS: Section "{section}" documented')
        else:
            print(f'FAIL: Section "{section}" NOT documented')
            all_pass = False

    # Check enum values documented
    if 'recovered' in contract and 'correctly_stopped' in contract and 'escalated_unresolved' in contract:
        print('PASS: Outcome enum values documented')
    else:
        print('FAIL: Outcome enum values not fully documented')
        all_pass = False

    if 'retry' in contract and 'no_action' in contract and 'escalate' in contract:
        print('PASS: Decision enum values documented')
    else:
        print('FAIL: Decision enum values not fully documented')
        all_pass = False

    return all_pass


def main():
    print("\n" + "#" * 60)
    print("# FINAL CONTRACT VERIFICATION")
    print("#" * 60)

    results = []
    results.append(("events_stream.json", verify_events_stream()))
    results.append(("model_metadata.json", verify_model_metadata()))
    results.append(("measurement_report.json", verify_measurement_report()))
    results.append(("audit_log.jsonl", verify_audit_log()))
    results.append(("simulate_endpoint.py", verify_simulate_endpoint()))
    results.append(("API_CONTRACT.md", verify_api_contract()))

    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n*** ALL VERIFICATIONS PASSED ***")
        print("The backend contract is complete and matches the documentation.")
    else:
        print("\n*** SOME VERIFICATIONS FAILED ***")
        print("Review the failures above.")

    return all_ok


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)