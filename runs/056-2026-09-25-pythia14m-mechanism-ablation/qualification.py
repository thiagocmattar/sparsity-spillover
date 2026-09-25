"""Eager-anchored numerical checks plus exact candidate/frozen graph agreement."""
import math
import torch
from torch.nn import functional as F
from mechanism_reference import same_bits


def compare_inputs(runners, inputs, bounds, numerical_gate, progress=lambda **fields: None):
    totals = {mode: 0. for mode in runners}
    gates = {mode: [] for mode in runners if mode != 'native'}
    exact = []
    prediction_tokens = 0
    with torch.inference_mode():
        for index, ids in enumerate(inputs):
            runners['native'].stage(ids)
            reference = runners['native']()
            outputs = {}
            for mode, runner in runners.items():
                if mode == 'native':
                    actual = reference
                else:
                    runner.stage(ids)
                    actual = runner()
                    gate = numerical_gate(reference, actual,
                        relative_l2=bounds['logit_relative_l2'],
                        atol=bounds['logit_atol'], rtol=bounds['logit_rtol'])
                    gates[mode].append({'input_index': index, **gate})
                outputs[mode] = actual
                loss = F.cross_entropy(actual[:, :-1].float().reshape(-1, actual.shape[-1]),
                                       ids[:, 1:].reshape(-1), reduction='sum')
                totals[mode] += float(loss)
                del actual, loss
            exact.append({'input_index': index,
                          'bitwise': same_bits(outputs['candidate_graph'], outputs['frozen_graph'])})
            prediction_tokens += ids.shape[0]*(ids.shape[1]-1)
            del outputs, reference
            if (index+1) % 16 == 0:
                progress(blocks=index+1, prediction_tokens=prediction_tokens,
                         loss={m: v/prediction_tokens for m, v in totals.items()})
    if not prediction_tokens:
        raise ValueError('Qualification needs at least one input')
    losses = {m: v/prediction_tokens for m, v in totals.items()}
    valid = {m: all(r['pass'] for r in rows) and math.isfinite(losses[m])
                and abs(losses[m]-losses['native']) <= bounds['validation_loss_atol']
             for m, rows in gates.items()}
    valid['native'] = math.isfinite(losses['native'])
    valid['candidate_frozen_bitwise'] = all(r['bitwise'] for r in exact)
    return {'prediction_tokens': prediction_tokens, 'loss': losses, 'gates': gates,
            'pass': valid, 'exact_graph_checks': exact,
            'loss_delta': {m: v-losses['native'] for m, v in losses.items()}}
