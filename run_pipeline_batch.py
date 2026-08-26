#!/usr/bin/env python3
"""
Run the pipeline batch processing to generate fresh audit log.
"""
import pipeline

def main():
    outcomes = pipeline.run_batch()
    print(f'Processed {len(outcomes)} events')
    recovered = sum(1 for o in outcomes if o.get('recovered'))
    escalated = sum(1 for o in outcomes if o.get('final_state', '').startswith('escalated'))
    no_action = sum(1 for o in outcomes if o.get('final_state') == 'no_action')
    validation_err = sum(1 for o in outcomes if o.get('final_state') == 'validation_error')
    print(f'  Recovered: {recovered}')
    print(f'  Escalated: {escalated}')
    print(f'  No action: {no_action}')
    print(f'  Validation err: {validation_err}')

if __name__ == '__main__':
    main()