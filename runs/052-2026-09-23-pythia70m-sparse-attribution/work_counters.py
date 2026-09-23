"""Validate actual per-CTA counts; requested loads are not DRAM transactions."""
import torch
from legacy_work import Work as PreviousWork


class Work(PreviousWork):
    def add(self,key,op,x):
        if op is None or op.spec['family'] not in ('e','f'):
            return super().add(key,op,x)
        m,k=x.shape;n=op.n;s=op.spec
        row=self.rows.setdefault(key,{'backend':s['id'],'blocks':0})
        def add(name,value):row[name]=row.get(name,0)+int(value)
        active=x!=0
        if s['family']=='e':
            gm=s['gm'];assert m%gm==0
            expected=active.reshape(m//gm,gm,k).any(1).sum(-1).int()
            nt=(n+s['bn']-1)//s['bn']
            assert torch.equal(op.count,expected[:,None].expand(-1,nt)),'CTA union counts disagree'
            used=torch.full_like(expected,k) if s.get('no_skip',False) else expected
            iterations=int(((used+15)//16).sum())
            add('padded_tensorcore_product_positions',iterations*16*16*n)
            add('products_in_row_padding',iterations*16*(16-gm)*n)
            add('weight_values_requested',int(used.sum())*n)
            add('consumer_tile_iterations',iterations*nt)
            add('activation_scan_values',m*k*nt)
            add('active_indices_in_local_storage',int(expected.sum())*nt)
        else:
            expected=active.sum(-1).int();nt=(n+32*s['v']-1)//(32*s['v'])
            assert torch.equal(op.count,expected[:,None].expand(-1,nt)),'Warp counts disagree'
            used=m*k if s.get('no_skip',False) else int(expected.sum())
            add('weight_values_requested',used*n);add('scalar_products_executed',used*n)
            add('activation_scan_values',m*k*nt)
        add('blocks',1);add('input_values',m*k);add('active_values',int(active.sum()))
        add('dense_products',m*k*n);add('nonzero_activation_products',int(active.sum())*n)
        add('count_words_written',op.count.numel())
        row['note']='CTA-local/warp counts checked against independent gated masks; tensor-core work is padded product positions, not directly measured instructions.'
