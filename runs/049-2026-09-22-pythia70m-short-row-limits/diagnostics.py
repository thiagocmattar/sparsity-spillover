"""Untimed operand statistics; native instruction counts remain unmeasured."""
import time
import torch
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator, weight_statistics
from frozen_diagnostics import projection_counts, hybrid_counts, attention_opportunities
from io_utils import write
from tile_oracle import counts as instruction_counts
from parallel_work_oracle import extra_scalar_products


def collect(model, native, validation, dest, emit, architecture, *, blocks=338):
    accumulator = ActivationAccumulator((0., .001, .01))
    histograms, counts, hybrid, attention = {}, {}, {}, {}
    input_instructions, tile_occupancy, actual_layouts = {}, {}, {}
    joint_occupancy = {}
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
                operands = [capture.activations[f'{site}.layer_{self.index}'].reshape(-1, width)
                            for site, width in [('h', 2048), ('z', 512)]]
                active = [(x != 0).sum(-1) for x in operands]
                safe = [((x == 0) | ((x.abs() >= 2.**-50) & (x.abs() <= 2.**50))).all(-1) for x in operands]
                for limit in (8, 16, 32, 64):
                    short = safe[0] & safe[1] & (active[0] <= limit) & (active[1] <= limit)
                    grouped = short.reshape(-1, 8)
                    all_short = grouped.all(-1)
                    duplicate = short & (~all_short).repeat_interleave(8)
                    group = joint_occupancy.setdefault(f'layer_{self.index}:limit_{limit}',
                        dict(rows=0, paired_short_rows=0, groups=0, all_paired_short_groups=0, mixed_groups=0,
                             repeated_h_scalar_products_if_prepass=0, repeated_z_scalar_products_if_prepass=0))
                    group['rows'] += short.numel()
                    group['paired_short_rows'] += int(short.sum())
                    group['groups'] += len(grouped)
                    group['all_paired_short_groups'] += int(all_short.sum())
                    group['mixed_groups'] += int((grouped.any(-1) & ~all_short).sum())
                    group['repeated_h_scalar_products_if_prepass'] += int(active[0][duplicate].sum()) * 512
                    group['repeated_z_scalar_products_if_prepass'] += int(active[1][duplicate].sum()) * 512
                if getattr(op, 'native', False):
                    return op(h, z, residual)
                old = op.count
                op.count = True
                try: output = op(h, z, residual)
                finally: op.count = old
                stat = op.stats.sum(tuple(range(op.stats.ndim - 1))).cpu().tolist()
                tile_rows = h.reshape(-1,2048).shape[0] // op.stats.shape[0]
                actual_layouts[str(self.index)] = {'row_group':tile_rows,'instruction_shape':[16,8,16],
                    'padded_rows_per_instruction':16-tile_rows,'short_row_limit':getattr(op,'short_limit',2),
                    'duplicate_prepass_work_counted':getattr(op,'parallel_prepass_duplicates',False)}
                extra = extra_scalar_products(capture.activations[f'h.layer_{self.index}'],
                    capture.activations[f'z.layer_{self.index}'], rows=tile_rows,
                    limit=getattr(op,'short_limit',2),fast_weights=op.fast_weights,skip=op.skip
                    ) if getattr(op,'parallel_prepass_duplicates',False) else [0,0]
                for site, offset, scalar in [('h', 0, 4), ('z', 2, 5)]:
                    expected = instruction_counts(capture.activations[f'{site}.layer_{self.index}'], tile_rows, op.fast_weights, op.skip, getattr(op,'short_limit',2))
                    expected[2] += extra[0 if site == 'h' else 1]
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
                                if projection is not None and not getattr(linear, '_run042_native', False):
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
           'actual_output_projection_layouts': actual_layouts,
           'cross_layout_counter_caution':'Compare absolute issued instructions and scalar products; M8 and M16 have different potential-instruction denominators.',
           'attention_fields': ['qk_issued', 'qk_bypassed', 'pv_issued', 'pv_bypassed'],
           'native_instruction_counts': 'unmeasured; absent entries are not zeros',
           'input_projection_instructions_by_layer': input_instructions,
           'input_projection_counter_method': 'Operand-derived exact16x16 test with the actual projection skip flag',
           'empty_tile_counts': tile_occupancy,
           'logical_and_opportunity_counts': counts, 'per_site_layer': accumulator.rows(),
           'pooled_by_site': accumulator.pooled_by_site(),
           'active_features_per_row': {k: v.cpu().tolist() for k, v in histograms.items()},
           'joint_occupancy_by_layer_limit': joint_occupancy,
           'joint_occupancy_note': 'Operand-derived safe row/group eligibility; duplicate-work fields are counterfactual for limits not used by the current backend. Actual instruction/scalar counts remain separate.',
           'weight_statistics': weight_statistics(native), 'elapsed_seconds': time.monotonic() - started})
