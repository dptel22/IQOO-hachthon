import json

with open('events_stream.json', 'r') as f:
    events = json.load(f)

print('Total events:', len(events))

# Check all required fields present
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
else:
    print('All required fields present in all events!')

# Check outcome distribution
from collections import Counter
outcomes = Counter(e['outcome'] for e in events)
print('Outcome distribution:')
for o, c in outcomes.most_common():
    print('  ', o, ':', c)

# Check decision distribution
decisions = Counter(str(e['decision']) for e in events)
print('Decision distribution:')
for d, c in decisions.most_common():
    print('  ', d, ':', c)

# Check fallback_flag
fallback = Counter(e['fallback_flag'] for e in events)
print('Fallback flag:')
for f, c in fallback.most_common():
    print('  ', f, ':', c)

# Check explanation presence
expl = Counter(len(e['explanation']) > 0 for e in events)
print('Has explanation (non-empty):')
for h, c in expl.most_common():
    print('  ', h, ':', c)

# Check p_recoverable range
p_vals = [e['p_recoverable'] for e in events]
print('p_recoverable range:', min(p_vals), '-', max(p_vals))

# Check enum values for payment_method
pm = Counter(e['payment_method'] for e in events)
print('payment_method values:')
for v, c in pm.most_common():
    print('  ', v, ':', c)

# Check error_code values
ec = Counter(e['error_code'] for e in events)
print('error_code values:')
for v, c in ec.most_common():
    print('  ', v, ':', c)