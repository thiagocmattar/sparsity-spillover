"""Untimed operand statistics; native instruction counts remain unmeasured."""
import time
import torch
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics
from frozen_diagnostics import projection_counts, hybrid_counts, attention_opportunities
from io_utils import write


def collect(model, native, validation, dest, emit, architecture, *, blocks=338):
    accumulator = ActivationAccumulator((0., .001, .01))
    histograms, counts, hybrid, attention = {}, {}, {}, {}
    input_instructions, tile_occupancy = {}, {}
    originals = []
    started = time.monotonic()

    def add(key, values):
        target = counts.setdefault(key, {})
        for name, value in values.items():
            target[name] = target.get(name, 0) + int(value)

    with ActivationCapture(model, ['a', 'm', 'h', 'z', 'q_post', 'k_post', 'v'], torch=torch) as capture:
        class Joint:
            def __init__(self, original, index): self.original, self.index = original, index

            def __call__(self, h, z, residual):
                op = self.original
                for site, value, gated, threshold in [('h', h, op.gh, op.th), ('z', z, op.gz, op.tz)]:
                    capture.activations[f'{site}.layer_{self.index}'] = value.masked_fill(value < threshold, 0) if gated else value
                if getattr(op, 'native', False):
                    return op(h, z, residual)
                old = op.count
                op.count = True
                try: output = op(h, z, residual)
                finally: op.count = old
                stat = op.stats.sum(tuple(range(op.stats.ndim - 1))).cpu().tolist()
                assert stat[0] + stat[1] == 2097152 and stat[2] + stat[3] == 524288
                for site, offset, scalar in [('h', 0, 4), ('z', 2, 5)]:
                    expected = hybrid_counts(capture.activations[f'{site}.layer_{self.index}'], op.fast_weights, op.skip)
                    assert [stat[offset], stat[offset + 1], stat[scalar]] == expected
                pooled = hybrid.setdefault(str(self.index), [0] * 6)
                for i, value in enumerate(stat): pooled[i] += int(value)
                return output

        class Attention:
            def __init__(self, original, index): self.original, self.index = original, index

            def __call__(self, q, k, v, scale):
                op = self.original
                for site, value in [('q_post', q), ('k_post', k), ('v', v)]:
                    capture.activations[f'{site}.layer_{self.index}'] = value
                if getattr(op, 'native', False):
                    return op(q, k, v, scale)
                output = op(q, k, v, scale, count=True)
                stat = op.stats.sum((0, 1, 2, 3)).cpu().tolist()
                assert stat[0] + stat[1] == stat[2] + stat[3] == 557056
                assert op.prefix_stats.sum().item() == 0
                pooled = attention.setdefault(str(self.index), [0] * 4)
                for i, value in enumerate(stat): pooled[i] += int(value)
                return output

        for i, layer in enumerate(model.gpt_neox.layers):
            joint = getattr(layer, '_run026_joint', None)
            attn = getattr(layer.attention, '_run028_attention', None)
            originals.append((layer, joint, attn))
            if joint is not None: layer._run026_joint = Joint(joint, i)
            if attn is not None: layer.attention._run028_attention = Attention(attn, i)
        try:
            with torch.inference_mode():
                for block in range(blocks):
                    ids = torch.tensor(validation[block * 2048:(block + 1) * 2048].copy(), device='cuda', dtype=torch.long)[None]
                    model(input_ids=ids, use_cache=False)
                    assert len(capture.activations) == 42
                    accumulator.update(capture.activations, torch=torch)
                    for name, value in capture.activations.items():
                        site, suffix = name.split('.')
                        width = value.shape[-1]
                        hist = torch.bincount(value.reshape(-1, width).count_nonzero(-1), minlength=width + 1)
                        histograms.setdefault(name, torch.zeros_like(hist)).add_(hist)
                        if site in ('a', 'm', 'h', 'z'):
                            operation, n = {'a': ('qkv_projection', 1536), 'm': ('mlp_w1', 2048),
                                            'h': ('mlp_w2', 512), 'z': ('attention_output_projection', 512)}[site]
                            row = projection_counts(value, n)
                            add(operation, {k: row[k] for k in ('product_count', 'zero_product_count')})
                            # Keep tile opportunity separate from actually issued native instructions.
                            add('16x16_opportunity_' + site, {'potential_mmas': row['issued_mmas'] + row['skipped_mmas'],
                                                            'empty_tile_mmas': row['skipped_mmas']})
                            zeros = value.reshape(-1, width) == 0
                            for tile_rows in (8, 16):
                                tiled = zeros.reshape(zeros.shape[0] // tile_rows, tile_rows, width // 16, 16).all(1).all(-1)
                                tile = tile_occupancy.setdefault(f'{name}:{tile_rows}x16', {'empty': 0, 'total': 0})
                                tile['empty'] += int(tiled.sum())
                                tile['total'] += tiled.numel()
                            if site in ('a', 'm'):
                                layer = model.gpt_neox.layers[int(suffix.split('_')[1])]
                                linear = layer.attention.query_key_value if site == 'a' else layer.mlp.dense_h_to_4h
                                projection = getattr(linear, '_run028_projection', None)
                                if projection is not None and not getattr(linear, '_run040_native', False):
                                    target = input_instructions.setdefault(name, {'issued': 0, 'bypassed': 0})
                                    skipped = row['skipped_mmas'] if projection.skip else 0
                                    target['bypassed'] += skipped
                                    target['issued'] += row['issued_mmas'] + row['skipped_mmas'] - skipped
                    for i in range(6):
                        operands = [capture.activations[f'{site}.layer_{i}'] for site in ('q_post', 'k_post', 'v')]
                        for op, values in attention_opportunities(*operands).items(): add(op, values)
                    capture.clear()
                    if (block + 1) % 32 == 0:
                        elapsed = time.monotonic() - started
                        emit('diagnostics', blocks=block + 1, input_tokens_per_second=(block + 1) * 2048 / elapsed,
                             remaining_seconds=(blocks - block - 1) * elapsed / (block + 1))
        finally:
            for layer, joint, attn in originals:
                if joint is not None: layer._run026_joint = joint
                if attn is not None: layer.attention._run028_attention = attn
    for op, expected in architecture['per_block_operation_products'].items():
        assert counts[op]['product_count'] == expected * architecture['layers'] * blocks
    write(dest, {'status': 'complete', 'coverage': {'blocks': blocks, 'documents': 500 if blocks == 338 else None,
           'input_tokens': blocks * 2048, 'excluded_tail_tokens': 1444 if blocks == 338 else None},
           'precision': 'BF16 actual operands, not the canonical FP16 model sparsity',
           'logical_counter_limits': 'Activation zeros only; excludes weight zeros and softmax-probability underflow. '
                                     'PV uses V zeros as a lower bound, not full model-wide sparsity.',
           'hybrid_fields': ['h_issued', 'h_bypassed', 'z_issued', 'z_bypassed', 'h_scalar_products', 'z_scalar_products'],
           'hybrid_counts_by_layer': hybrid, 'attention_counts_by_layer': attention,
           'attention_fields': ['qk_issued', 'qk_bypassed', 'pv_issued', 'pv_bypassed'],
           'native_instruction_counts': 'unmeasured; absent entries are not zeros',
           'input_projection_instructions_by_layer': input_instructions,
           'input_projection_counter_method': 'Operand-derived exact16x16 test with the actual projection skip flag',
           'empty_tile_counts': tile_occupancy,
           'logical_and_opportunity_counts': counts, 'per_site_layer': accumulator.rows(),
           'pooled_by_site': accumulator.pooled_by_site(),
           'active_features_per_row': {k: v.cpu().tolist() for k, v in histograms.items()},
           'weight_statistics': weight_statistics(native), 'elapsed_seconds': time.monotonic() - started})
