"""Untimed full-coverage operands, exactness, and independently checked work counts."""
import time
import torch
from io_utils import RUN, module, write
from frozen_replay import R28, scaffold
from mechanism_reference import projection_work, same_bits
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics

legacy = module('run056_frozen_operand_formulas', R28 / '115_hybrid_diagnostics.py')


def add_counts(target, values):
    for key, value in values.items():
        target[key] = target.get(key, 0) + int(value)


def collect(model, frozen, native, validation, dest, emit, architecture, *, blocks=338, counts=True):
    started = time.monotonic()
    accumulator = ActivationAccumulator((0., .001, .01))
    weights = weight_statistics(native) if counts else None
    operations, mechanisms, histograms, attention_counts = {}, {}, {}, {}
    originals, exact_checks, frozen_values = [], [], {}
    state = {'block': 0, 'completed': 0}
    status = 'failed'

    def add_op(op, values):
        add_counts(operations.setdefault(op, {}), values)

    with ActivationCapture(model, ['a', 'm'], torch=torch) as capture, torch.inference_mode():
        class Joint:
            def __init__(self, original, index, reference):
                self.original, self.index, self.reference = original, index, reference

            def __call__(self, h, z, residual):
                original = self.original
                work = {}
                checks = {}
                for site, value, gate, threshold in [
                        ('h', h, original.gh, original.th), ('z', z, original.gz, original.tz)]:
                    name = f'{site}.layer_{self.index}'
                    value = value.masked_fill(value < threshold, 0) if gate else value
                    if self.reference:
                        frozen_values[name] = value.clone()
                        continue
                    reference = frozen_values[name]
                    checks[site] = {'values_bitwise': same_bits(value, reference),
                                    'zero_masks_equal': torch.equal(value == 0, reference == 0)}
                    if counts:
                        capture.activations[name] = value
                        row = projection_work(value, tile=getattr(original, 'tile', True),
                            short=getattr(original, 'short', True), fast_weights=original.fast_weights)
                        work[site] = row
                        add_counts(mechanisms.setdefault(name, {}), row['counts'])
                if not self.reference:
                    exact_checks.append({'block': state['block'], 'layer': self.index, 'sites': checks})
                    if not all(all(check.values()) for check in checks.values()):
                        raise ValueError(f'Changed h/z operand bits at block {state["block"]}, layer {self.index}')
                old_count = original.count
                original.count = counts and not self.reference
                try:
                    result = original(h, z, residual)
                finally:
                    original.count = old_count
                if counts and not self.reference:
                    stat = original.stats.sum((0, 1)).cpu().tolist()
                    hc, zc = [work[s]['legacy_counters'] for s in ('h', 'z')]
                    expected = [hc[0], hc[1], zc[0], zc[1], hc[2], zc[2]]
                    if stat != expected:
                        raise ValueError(f'Instrumented work mismatch: {stat} != {expected}')
                    early = int((work['h']['completed_groups'] & work['z']['completed_groups']).sum())
                    add_counts(mechanisms.setdefault(f'joint.layer_{self.index}', {}),
                               {'joint_short_early_returns': early})
                    for site, op in [('h', 'mlp_w2'), ('z', 'attention_output_projection')]:
                        row = work[site]['counts']
                        add_op(op, {'issued_mmas': row['mma_issued'],
                                    'skipped_mmas': row['mma_potential']-row['mma_issued'],
                                    'simt_products': row['scalar_products']})
                return result

        class Attention:
            def __init__(self, original, index):
                self.original, self.index = original, index

            def __call__(self, q, k, v, scale):
                output = self.original(q, k, v, scale, count=True)
                for site, value in [('q_post', q), ('k_post', k), ('v', v)]:
                    capture.activations[f'{site}.layer_{self.index}'] = value
                stat = self.original.stats.sum((0, 1, 2, 3)).cpu().tolist()
                if self.original.prefix_stats.sum().item() != 0:
                    raise ValueError('Frozen no-prefix attention policy changed')
                if stat[0]+stat[1] != 147456 or stat[2]+stat[3] != 147456:
                    raise ValueError('Attention matrix-work conservation failed')
                target = attention_counts.setdefault(str(self.index), [0]*4)
                for j, value in enumerate(stat):
                    target[j] += int(value)
                for op, row in legacy.attention_opportunities(q, k, v).items():
                    add_op(op, row)
                return output

        for is_frozen, current in [(True, frozen), (False, model)]:
            for i, layer in enumerate(current.gpt_neox.layers):
                originals.append((layer, '_run026_joint', layer._run026_joint))
                layer._run026_joint = Joint(layer._run026_joint, i, is_frozen)
                if counts and not is_frozen:
                    originals.append((layer.attention, '_run028_attention', layer.attention._run028_attention))
                    layer.attention._run028_attention = Attention(layer.attention._run028_attention, i)
        try:
            for index in range(blocks):
                state['block'] = index
                ids = torch.tensor(validation[index*2048:(index+1)*2048].copy(),
                                   device='cuda', dtype=torch.long)[None]
                frozen_values.clear()
                scaffold.forward(frozen, ids)
                scaffold.forward(model, ids)
                if len(frozen_values) != architecture['layers']*2:
                    raise ValueError('Missing frozen h/z operand coverage')
                if counts:
                    expected_names = {f'{s}.layer_{i}' for s in ['a','m','h','z','q_post','k_post','v']
                                      for i in range(architecture['layers'])}
                    if set(capture.activations) != expected_names:
                        raise ValueError('Incomplete seven-site diagnostic capture')
                    accumulator.update(capture.activations, torch=torch)
                    for name, value in capture.activations.items():
                        site = name.split('.')[0]
                        width = value.shape[-1]
                        hist = torch.bincount(torch.count_nonzero(value.reshape(-1, width), dim=-1),
                                              minlength=width+1)
                        histograms.setdefault(name, torch.zeros_like(hist)).add_(hist)
                        if site in {'a','m','h','z'}:
                            op, n = {'a': ('qkv_projection',384), 'm': ('mlp_w1',512),
                                     'h': ('mlp_w2',128), 'z': ('attention_output_projection',128)}[site]
                            row = legacy.projection_counts(value, n)
                            if site in {'h','z'}:
                                row = {key: row[key] for key in ['product_count','zero_product_count']}
                            add_op(op, row)
                capture.clear()
                state['completed'] += 1
                if (index+1) % 32 == 0:
                    elapsed = time.monotonic()-started
                    emit('diagnostics', blocks=index+1, input_tokens_per_second=(index+1)*2048/elapsed,
                         remaining_seconds=(blocks-index-1)*elapsed/(index+1))
            if len(exact_checks) != blocks*architecture['layers']:
                raise ValueError('Missing exact operand checks')
            if counts:
                for op, denominator in architecture['per_block_operation_products'].items():
                    if operations[op]['product_count'] != denominator*architecture['layers']*blocks:
                        raise ValueError(f'Logical denominator changed: {op}')
            status = 'complete'
        finally:
            for obj, name, original in reversed(originals):
                setattr(obj, name, original)
            report = {
                'status': status, 'counts_collected': counts,
                'coverage': {'blocks': state['completed'], 'target_blocks': blocks,
                    'documents': 500 if blocks == 338 else None, 'input_tokens': state['completed']*2048,
                    'excluded_tail_tokens': 1444 if blocks == 338 else None},
                'operand_bitwise': status == 'complete' and all(all(all(s.values()) for s in row['sites'].values())
                                                                    for row in exact_checks),
                'exact_operand_checks': exact_checks,
                'elapsed_seconds': time.monotonic()-started}
            if counts:
                pooled = {}
                for name, row in mechanisms.items():
                    add_counts(pooled.setdefault(name.split('.')[0], {}), row)
                numerator = sum(row.get('zero_product_count',0) for row in operations.values())
                denominator = architecture['model_product_count']*state['completed']
                report.update(
                    mechanisms_by_layer=mechanisms, mechanisms_pooled=pooled,
                    mechanism_counter_method='Operand-derived categories checked against instrumented CUDA issued/bypassed/scalar totals; S whole-group elimination takes precedence.',
                    weight_request_unit='BF16 source operand requests, not cache misses or DRAM bytes',
                    bf16_scalar_opportunity_lower_bound={'zero_products': numerator, 'model_products': denominator,
                        'fraction': numerator/denominator if denominator else None, 'per_operation': operations,
                        'exclusions': 'weight zeros and probability underflow; not canonical R_model'},
                    per_site_layer=accumulator.rows(), pooled_by_site=accumulator.pooled_by_site(),
                    active_features_per_row={n: h.cpu().tolist() for n,h in histograms.items()},
                    native_weight_statistics=weights, attention_counts_by_layer=attention_counts)
            write(dest, report)
    return report
