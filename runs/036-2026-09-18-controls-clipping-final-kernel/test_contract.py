"""Threshold semantics, exact ports, and source/condition integrity."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from clipping_adapter import clip, install, validate_thresholds

HERE=Path(__file__).resolve().parent


class Contract(unittest.TestCase):
    def test_signed_equality_and_deferred_relu(self):
        for dtype in (torch.float32,torch.bfloat16):
            x=torch.tensor([-2.,-.5,-.1,0.,.1,.5,2.],dtype=dtype)
            for threshold in (0.,.1,.5):
                expected=x.masked_fill(x.abs()<=threshold,0.)
                self.assertTrue(torch.equal(clip(x,threshold),expected))
                h=torch.relu(x)
                self.assertTrue(torch.equal(clip(x,threshold,deferred_relu=True),h.masked_fill(h.abs()<=threshold,0.)))

    def test_threshold_coverage(self):
        complete={f'{s}.layer_0':0. for s in ('a','m','h','z')}
        validate_thresholds(complete,1)
        for wrong in ({'h.layer_0':0.},{**complete,'a.layer_0':float('nan')},{**complete,'a.layer_0':-1.}):
            with self.assertRaises(ValueError):validate_thresholds(wrong,1)

    def test_real_pythia_hook_placement_and_relu_identity(self):
        from transformers import GPTNeoXConfig,AutoModelForCausalLM
        from sparsity_research.pythia import expose_attention_sites
        cfg=GPTNeoXConfig(hidden_size=32,intermediate_size=64,num_attention_heads=4,num_hidden_layers=1,vocab_size=64)
        model=AutoModelForCausalLM.from_config(cfg).eval()
        expose_attention_sites(model,torch=torch)
        layer=model.gpt_neox.layers[0]
        layer.mlp.act=torch.nn.ReLU()
        thresholds={f'{s}.layer_0':.5 for s in ('a','m','h','z')}
        handles=install(model,thresholds)
        x=torch.tensor([[-1.,-.5,.25,.5,1.]])
        self.assertTrue(torch.equal(layer.mlp.act(x),torch.tensor([[0.,0.,0.,0.,1.]])))
        self.assertTrue(torch.equal(layer.attention.z_site(x),torch.tensor([[-1.,0.,0.,0.,1.]])))
        self.assertEqual(len(handles),4)
        for h in handles:h.remove()
        # The final kernel defers ReLU; candidate hook must restore its effect.
        layer.mlp.act.forward=lambda x:x
        install(model,thresholds,candidate=True)
        self.assertTrue(torch.equal(layer.mlp.act(x),torch.tensor([[0.,0.,0.,0.,1.]])))

    def test_retained_grid_and_checkpoint_identity(self):
        manifest=json.loads((HERE/'provenance/inputs.json').read_text())
        conditions=manifest['conditions']
        self.assertEqual(len(conditions),40)
        self.assertEqual(len({c['id'] for c in conditions}),40)
        for cp in manifest['checkpoints']:
            rows=[c for c in conditions if c['checkpoint_id']==cp['id']]
            self.assertEqual([c['p'] for c in rows],[i/10 for i in range(10)])
            for c in rows:
                validate_thresholds(c['retained_point']['thresholds_by_site_layer'],6)
                self.assertEqual(c['retained_point']['coverage']['sequences'],338)
                self.assertEqual(c['candidate'],'k050' if cp['scale']=='14M' else 'k050-70m-v2')


if __name__=='__main__':unittest.main()
