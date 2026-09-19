"""Pool retained integer diagnostics; no new GPU measurements or timing selection."""
from io_utils import RUN, read, write, record


def main():
    source=RUN/'results/am-load-avoidance.json'
    report=read(source)
    assert report['status']=='qualified' and report['logical_counts_match_frozen']
    baseline=report['diagnostics']['frozen']
    port=report['diagnostics']['port-am']
    rows={}
    mapping={'a':('qkv_projection',128),'m':('mlp_w1',128),
             'h':('mlp_w2',512),'z':('attention_output_projection',128)}
    for site,(operation,width) in mapping.items():
        layers=[v for k,v in baseline['active_features_per_row'].items() if k.startswith(site+'.layer_')]
        assert len(layers)==6 and all(len(v)==width+1 for v in layers)
        hist=[sum(v[i] for v in layers) for i in range(width+1)]
        total_rows=sum(hist);nonzeros=sum(i*v for i,v in enumerate(hist))
        assert total_rows==338*2048*6
        row={'rows':total_rows,'elements':total_rows*width,
             'zeros':total_rows*width-nonzeros,
             'sparsity':1-nonzeros/(total_rows*width),
             'rows_at_most_two_nonzeros':sum(hist[:3]),
             'fraction_rows_at_most_two_nonzeros':sum(hist[:3])/total_rows,
             'mean_nonzeros_per_row':nonzeros/total_rows}
        old=baseline['bf16_scalar_opportunity_lower_bound']['per_operation'][operation]
        new=port['bf16_scalar_opportunity_lower_bound']['per_operation'][operation]
        n={'a':384,'m':512,'h':128,'z':128}[site]
        assert row['zeros']*n==old['zero_product_count']
        assert row['elements']*n==old['product_count']
        row['frozen_mma']={k:old[k] for k in ['issued_mmas','skipped_mmas']}
        row['port_mma']={k:new[k] for k in ['issued_mmas','skipped_mmas']}
        row['port_simt_products']=new.get('simt_products',0)
        row['port_to_frozen_issued_mma_ratio']=new['issued_mmas']/old['issued_mmas']
        if site in {'a','m'}:
            tiles=[v for k,v in baseline['input_tile_occupancy'].items() if k.startswith(site+'.layer_')]
            assert len(tiles)==6
            row['tiles']={k:sum(v[k] for v in tiles) for k in ['empty8','total8','empty16','total16']}
            for size in [8,16]:
                row['tiles'][f'empty_fraction_{size}x16']=row['tiles']['empty'+str(size)]/row['tiles']['total'+str(size)]
            requested=new['mma_weight_elements_requested']+new['scalar_weight_elements_requested']
            row['source_weight_element_requests']=requested
            row['padded_dense_source_weight_element_requests']=new['dense_layout_weight_elements']
            row['fraction_weight_requests_avoided_within_port_layout']=1-requested/new['dense_layout_weight_elements']
        rows[site]=row
    output={'source':record(source),'coverage':baseline['coverage'],'sites':rows,
            'limits':'BF16 executed operands; raw integer counts pooled before division. Source weight requests are not measured DRAM traffic. Bypass includes short-row scalar substitution. Port uses twice as many padded M16 opportunities at a/m as the frozen M16 layout.'}
    write(RUN/'results/mechanism.json',output)
    lines=['# Pooled structure and executed work','','| Site | Sparsity (%) | Rows with <=2 nonzeros (%) | Mean nonzeros/row | Frozen issued MMA | Port issued MMA |',
           '|---|---:|---:|---:|---:|---:|']
    for site,r in rows.items():
        lines.append(f"| {site} | {100*r['sparsity']:.4f} | {100*r['fraction_rows_at_most_two_nonzeros']:.4f} | {r['mean_nonzeros_per_row']:.4f} | {r['frozen_mma']['issued_mmas']:,} | {r['port_mma']['issued_mmas']:,} |")
    lines+=['','| Site | Empty 16x16 tiles (%) | Empty 8x16 tiles (%) | Weight requests avoided within port layout (%) |',
            '|---|---:|---:|---:|']
    for site in ['a','m']:
        r=rows[site];t=r['tiles']
        lines.append(f"| {site} | {100*t['empty_fraction_16x16']:.4f} | {100*t['empty_fraction_8x16']:.4f} | {100*r['fraction_weight_requests_avoided_within_port_layout']:.4f} |")
    lines+=['',output['limits'],'']
    (RUN/'results/mechanism.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
