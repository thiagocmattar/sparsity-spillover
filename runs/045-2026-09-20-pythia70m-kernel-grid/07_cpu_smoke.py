"""Real checkpoint/adapter smoke without CUDA execution or timing claims."""
import argparse
import copy
import json
from io_utils import RUN, read, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    import replay
    import torch
    import transformers
    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
    from sparsity_research.sites import FixedOneSidedThreshold
    torch.set_num_threads(2)
    checkpoint = next(c for c in read(RUN/'provenance/inputs.json')['checkpoints'] if c['id'] == args.condition)
    for row in checkpoint['files']:
        verify(row)
    model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, RUN/checkpoint['checkpoint'], torch=torch)
    model.eval()
    before = topology_metadata(model)
    if checkpoint['family'] == 'HZ+OL1@h':
        assert before['topology_id'] == 'HZ'
        for layer in model.gpt_neox.layers:
            assert isinstance(layer.mlp.act, FixedOneSidedThreshold)
            assert isinstance(layer.attention.z_gate, FixedOneSidedThreshold)
            assert layer.mlp.act.kappa == layer.attention.z_gate.kappa == checkpoint['dose']
            assert not hasattr(layer, 'a_gate') and not hasattr(layer, 'm_gate')
    with torch.inference_mode():
        anchor = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits
        assert anchor.shape == (1, 4, 50304) and torch.isfinite(anchor).all()
        for mode in ('legacy', 'opt073'):
            candidate = copy.deepcopy(model).to(dtype=torch.bfloat16)
            metadata = replay.install(candidate, mode)
            assert topology_metadata(candidate) == before
            assert all(hasattr(layer, '_run026_joint') for layer in candidate.gpt_neox.layers)
            for layer in candidate.gpt_neox.layers:
                operation = layer._run026_joint
                if checkpoint['family'] == 'HZ+OL1@h':
                    assert operation.gh and operation.gz
                    assert operation.th == operation.tz == checkpoint['dose']
            del candidate
    print(json.dumps({'status': 'passed', 'condition': args.condition, 'topology': before,
                      'native_cpu_logits': list(anchor.shape),
                      'specialized_execution': 'installation only; CUDA qualification remains required'}))


if __name__ == '__main__':
    main()
