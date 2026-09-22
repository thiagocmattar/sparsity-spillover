"""Verify that the grid preserves the requested checkpoint and kernel identities."""
import json,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())

def test_complete_matched_grid():
 rows=read(RUN/'provenance/inputs.json')['checkpoints']
 assert len(rows)==11 and len({r['id'] for r in rows})==11
 for family,topology in [('HZ+OL1@h','HZ'),('A7+OL1@all','A7-Z-POST')]:
  subset=[r for r in rows if r['family']==family]
  assert sorted(r['dose'] for r in subset)==[0,.01,.05,.1,.5]
  assert all(r['topology']['topology_id']==topology for r in subset)
  assert all(r['topology']['hidden_size']==512 and r['topology']['num_hidden_layers']==6 for r in subset)
  for r in subset:
   source=r['source_condition']
   assert source['pressure_method']=='orthogonal_l1'
   assert source['pressure_sites']==(['h'] if family=='HZ+OL1@h' else ['a','m','h','q_post','k_post','v','z'])

def test_frozen_kernel_and_policy():
 old=RUN.parent/'050-2026-09-22-pythia70m-kernel-families'
 assert (RUN/'provenance/selection-final.json').read_bytes()==(old/'provenance/selection-final.json').read_bytes()
 for row in read(RUN/'provenance/kernel-reuse.json')['files']:
  raw=(RUN/row['path']).read_bytes()
  assert raw==(old/row['path']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==row['sha256']
 selection=read(RUN/'provenance/selection-final.json')
 assert sum(v.startswith('c_') for k,v in selection['policies']['sparse_c'].items())==5
 assert all(v=='dense_dot' for k,v in selection['policies']['sparse_c'].items() if k.startswith('z.'))

def test_checkpoint_provenance_and_validation():
 manifest=read(RUN/'provenance/inputs.json')
 old=RUN.parent/'049-2026-09-22-pythia70m-short-row-limits'
 assert manifest['validation']==read(old/'provenance/inputs.json')['validation']
 for row in manifest['checkpoints']:
  for item in row['provenance']:
   raw=(RUN/item['path']).read_bytes()
   assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
 cfg=read(RUN/'config.json')
 assert cfg['conditions']==[r['id'] for r in manifest['checkpoints']]
 assert cfg['validation_blocks']==338 and cfg['process_replicates']==3
