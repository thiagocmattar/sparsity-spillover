"""Retain every attempt; derive best-so-far only from qualified measurements."""
import csv
from common import HERE, read_json


def rows(results):
    best = 1.0
    for result in results:
        key = result.get('candidate_key', 'candidate')
        timing = result.get('timing', {})
        candidate = timing.get(key, {})
        speedup = result.get('primary_speedup')
        if result.get('qualified') and speedup is not None:
            best = max(best, speedup)
        logical = result.get('canonical_logical_products', {}).get('measured', {})
        yield {'iteration': result['attempt'], 'idea': result['idea'],
               'status': result['status'], 'qualified': result.get('qualified', False),
               'quality_scope': result.get('quality_scope'), 'dense_comparator': result.get('selected_dense'),
               'latency_ms': candidate.get('median_host_ms'),
               'dense_latency_ms': timing.get(result.get('selected_dense'), {}).get('median_host_ms'),
               'R_model': result.get('canonical_R_model'),
               'zero_products': logical.get('block_zero_product_count'),
               'model_products': logical.get('model_product_count'),
               'speedup_matched_dense': speedup,
               'speedup_eager': candidate.get('paired_geomean_speedup'),
               'best_so_far_matched_dense': best, 'elapsed_seconds': result.get('elapsed_seconds'),
               'error': result.get('error', '')}


if __name__ == '__main__':
    output = list(rows(read_json(p) for p in sorted((HERE/'artifacts').glob('*/result.json'))))
    if output:
        with (HERE/'progress.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print(f'{len(output)} iterations recorded')
