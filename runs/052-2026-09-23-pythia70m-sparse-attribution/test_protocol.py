"""Scientific selection, contrast signs and metadata conservation; CPU only."""
import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

RUN=Path(__file__).resolve().parent


def load(name,file):
    sys.path.insert(0,str(RUN))
    spec=importlib.util.spec_from_file_location(name,RUN/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_raw_delta_can_pass_while_dense_adjusted_delta_fails():
    e=load('r052_effects','effects.py')
    values={'14_sparse05':.60,'14_sparse10':.58,'14_dense05':.65,'14_dense10':.65,
            '70_sparse05':1.20,'70_sparse10':1.17,'70_dense05':1.30,'70_dense10':1.28}
    arrays={k:np.full((3,8),v) for k,v in values.items()}
    result=e.interval(arrays,draws=100)
    assert result['point']['delta70_minus14_ms']==pytest.approx(.01)
    assert result['point']['adjusted70_minus14_ms']==pytest.approx(-.01)
    assert result['crossed_process_input_95']['adjusted70_minus14_ms']==pytest.approx([-.01,-.01])


def test_contrast_rejects_unmatched_or_invalid_timings():
    e=load('r052_effects_invalid','effects.py')
    with pytest.raises(ValueError):e.interval({'x':np.ones((3,2)),'y':np.ones((2,2))})
    with pytest.raises(ValueError):e.interval({'x':np.zeros((3,2))})


def test_failed_dense_control_does_not_hide_qualified_raw_threshold_drop():
    e=load('r052_partial_qualification','effects.py')
    arrays={size+'_'+kind+dose:np.full((3,4),value)
            for size in ('14','70') for kind,value in [('sparse',1.),('dense',1.2)] for dose in ('05','10')}
    accepted={key:True for key in arrays};accepted['70_dense10']=False
    result=e.interval(arrays,qualified=accepted,draws=100)
    assert result['qualified']['delta70_minus14_ms'] is True
    assert result['crossed_process_input_95']['delta70_minus14_ms']==pytest.approx([0.,0.])
    assert result['crossed_process_input_95']['adjusted70_minus14_ms'] is None


def test_control_effects_are_paired_and_failed_controls_cannot_qualify():
    e=load('r052_controls','effects.py')
    jitter=np.array([[.8,1.,1.2],[.9,1.1,1.3],[1.,1.2,1.4]])
    arrays={'sparse':jitter,'dense':jitter*1.2,'Base':jitter*1.5,'off':jitter*1.1}
    result=e.paired_intervals(arrays,{'dense_gain':{'dense':1,'sparse':-1},'skip_gain':{'off':1,'sparse':-1}},
                              ratios={'Base_speedup':('Base','sparse')},qualified={'sparse':True,'dense':True,'Base':True,'off':False},draws=200)
    assert result['crossed_process_input_95']['Base_speedup']==pytest.approx([1.5,1.5])
    assert result['crossed_process_input_95']['dense_gain'][0]>0
    assert result['qualified']['skip_gain'] is False
    assert result['crossed_process_input_95']['skip_gain'] is None
    assert result['point']['skip_gain']>0


def screen_fixture():
    return {f'{cid}:{split}:{site}.0':{
        n:{'qualified':True,'host_ms':v} for n,v in
        [('dense_native',1.2),('dense_fused',1.1),('dense_dot',1.),('c_prior',.8),('e_new',.7),('f_one_kappa',.6 if cid=='c24' else 1.3)]}
        for cid in ('c24','c25') for split in ('development','confirmation') for site in ('h','z')}


def test_selection_requires_both_kappas_and_both_training_splits():
    s=load('r052_select','05_select.py');rows=screen_fixture();prior={'h.0':'c_prior','z.0':'dense_dot'}
    result=s.select(rows,prior)
    assert result['policies']['candidate']=={'h.0':'e_new','z.0':'e_new'}
    assert result['ablations']['candidate_h_off']=={'h.0':'e_new_noskip','z.0':'e_new'}
    rows['c25:confirmation:z.0']['e_new']['qualified']=False
    assert s.select(rows,prior)['policies']['candidate']['z.0']=='dense_dot'
    rows['c24:development:h.0']['e_new']['host_ms']=.79
    assert s.select(rows,prior)['policies']['candidate']['h.0']=='c_prior'


def test_new_counts_are_actual_union_counts_and_padding_is_explicit():
    w=load('r052_work','work_counters.py')
    x=torch.zeros(64,512,dtype=torch.bfloat16);x[:,0:3]=.5
    op=SimpleNamespace(spec={'id':'e_test','family':'e','gm':4,'bn':64},n=512,count=torch.full((16,8),3,dtype=torch.int32))
    work=w.Work();work.add('z.0',op,x);r=work.result()['per_site_layer']['z.0']
    assert r['weight_values_requested']==16*3*512
    assert r['padded_tensorcore_product_positions']==16*16*16*512
    assert r['products_in_row_padding']==16*16*12*512
    assert r['activation_scan_values']==64*512*8
    op.spec['no_skip']=True;work=w.Work();work.add('z.0',op,x)
    assert work.result()['per_site_layer']['z.0']['weight_values_requested']==16*512*512
    op.count[0,2]=2
    with pytest.raises(AssertionError):w.Work().add('z.0',op,x)


def test_warp_counts_and_disabled_work_preserve_gate_semantics():
    w=load('r052_work_warp','work_counters.py')
    x=torch.zeros(64,512,dtype=torch.bfloat16);x[:,0:3]=.5
    op=SimpleNamespace(spec={'id':'f_test','family':'f','v':4},n=512,count=torch.full((64,4),3,dtype=torch.int32))
    work=w.Work();work.add('z.0',op,x)
    assert work.result()['per_site_layer']['z.0']['scalar_products_executed']==64*3*512
    op.spec['no_skip']=True;work=w.Work();work.add('z.0',op,x)
    assert work.result()['per_site_layer']['z.0']['scalar_products_executed']==64*512*512


def test_training_prefix_and_reference_checkpoint_identity():
    import json,hashlib
    p=json.loads((RUN/'provenance/inputs.json').read_text());raw=(RUN/p['training']['path']).read_bytes()
    assert len(raw)==128*2048*4 and hashlib.sha256(raw).hexdigest()==p['training']['sha256']
    assert p['development_indices']==list(range(64)) and p['confirmation_indices']==list(range(64,128))
    references=json.loads((RUN/'provenance/reference14.json').read_text())['checkpoints']
    assert [(r['id'],r['dose']) for r in references]==[('d05',.05),('d10',.1)]
    for r in references:
        assert r['source_condition']['active_sites']==['h','z'] and r['source_condition']['pressure_sites']==['h']


def test_archive_verifies_actual_member_bytes_and_rejects_wrong_inventory(tmp_path):
    import tarfile,json,hashlib
    v=load('r052_archive','14_verify_archive.py');source=tmp_path/'source.txt';source.write_bytes(b'retained evidence')
    archive=tmp_path/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:stream.add(source,arcname='artifacts/source.txt')
    record={'path':'artifacts/source.txt','bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'files':[record]}))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
    assert v.verify(archive,inventory,receipt)['files']==1
    record['sha256']='0'*64;inventory.write_text(json.dumps({'files':[record]}))
    with pytest.raises(AssertionError):v.verify(archive,inventory,receipt)
