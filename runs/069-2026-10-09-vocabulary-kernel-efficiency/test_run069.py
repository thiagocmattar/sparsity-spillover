"""Certificate precision, independent masks and immutable input receipts."""
from pathlib import Path
import torch
import pytest
import common


def test_certificate_precedes_half_rounding():
    threshold=torch.tensor(0.1,dtype=torch.float32)
    raw=torch.zeros(32,128)
    raw[0,0]=torch.nextafter(threshold,torch.tensor(float('inf')))
    raw[16,0]=threshold
    bits=common.support(raw,float(threshold))
    assert bits.tolist()==[1,0]
    assert raw[0,0].half()==raw[16,0].half()


def test_row_groups_keep_independent_feature_support():
    raw=torch.zeros(32,128)
    raw[0,0]=1;raw[16,16]=1
    bits=common.support(raw,0.1)
    assert bits.tolist()==[1,2]
    weights=torch.arange(128*8,dtype=torch.float32).reshape(128,8)
    values=torch.ones(32,128)
    mask=raw[::16,::16]>0.1
    gated=(values.reshape(2,16,8,16)*mask[:,None,:,None]).reshape(32,128)
    result=gated@weights
    assert torch.equal(result[:16],weights[:16].sum(0).expand(16,-1))
    assert torch.equal(result[16:],weights[16:32].sum(0).expand(16,-1))


def test_identity_receipt_detects_changed_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'REPO',tmp_path)
    path=tmp_path/'receipt-source.json';common.write(path,{'count':338,'tail':1444})
    row=common.record(path)
    assert common.verify(row)==path
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='Identity mismatch'):common.verify(row)


def test_selection_keeps_separate_dense_and_sparse_winners():
    def timing(dense,sparse):
        return {'no_skip':{'cuda_ms':{'geomean':dense}},'sparse':{'cuda_ms':{'geomean':sparse}}}
    data={'legacy_n256':timing(3,2),'a':timing(1,1.5),'b':timing(2,1)}
    assert common.shortlist(data,2)==['a','b']


@pytest.mark.parametrize('defect,expected',[
    ('coverage','Incomplete validation'),('loss','Pooled loss mismatch'),('samples','Incomplete timing samples')])
def test_endpoint_verifier_rejects_incomplete_evidence(tmp_path,monkeypatch,defect,expected):
    import importlib.util
    spec=importlib.util.spec_from_file_location('run069_verify',Path(__file__).with_name('08_verify.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUN',tmp_path)
    monkeypatch.setattr(module,'sources',lambda:{'checkpoints':[{'id':'fixture'}]})
    cfg=common.config();folder=tmp_path/'artifacts/endpoints/fixture-r1'
    result={'status':'complete','qualified':True,'evidence':'valid','blocks':338,'predictions':691886,
            'runtime':{'gpu':'NVIDIA GeForce RTX 5090','gpu_uuid':'fixture'},
            'timings':{'full_forward':{'native':{'count':448}}}}
    quality={'excluded_tail_tokens':1444,'bounds':cfg['numerical_bounds'],
             'checks':{'native':[{'pass':True}]*338},'loss':{'native':1.,'anchor':1.}}
    if defect=='coverage':result['blocks']=2
    if defect=='loss':quality['loss']['native']=1.1
    if defect=='samples':result['timings']['full_forward']['native']['count']=447
    common.write(folder/'result.json',result);common.write(folder/'quality.json',quality)
    with pytest.raises(ValueError,match=expected):module.main()
