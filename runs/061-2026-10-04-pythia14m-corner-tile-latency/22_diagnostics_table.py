"""Pool retained integer site/tile and issued-work counts for this benchmark."""
from io_utils import RUN, read, write, record, verify


def main():
    result=read(RUN/'results/results.json');rows=[];sources=[]
    for pair in result['pairs']:
        for family,source in pair['diagnostics'].items():
            diagnostics=read(verify(source));sources.append(source)
            pooled={r['name']:r for r in diagnostics['pooled_by_site']}
            sites={}
            for site in ['a','m','h','z']:
                activation=pooled[site]
                tiles=[r for name,r in diagnostics['tiles8x16_by_site_layer'].items() if name.startswith(site+'.')]
                empty=sum(r['empty'] for r in tiles);total=sum(r['total'] for r in tiles)
                sites[site]={'exact_zero_count':activation['exact_zero_count'],'activation_count':activation['total'],
                             'exact_zero_fraction':activation['exact_zero_count']/activation['total'],
                             'empty_tiles':empty,'total_tiles':total,'empty_tile_fraction':empty/total}
            work={}
            for site,operation in [('h','mlp_w2'),('z','attention_output_projection')]:
                counts=diagnostics['bf16_scalar_opportunity_lower_bound']['per_operation'][operation]
                issued,bypassed=counts['issued_mmas'],counts['skipped_mmas']
                work[site]={'issued_mmas':issued,'bypassed_mmas':bypassed,'simt_products':counts['simt_products'],
                            'issued_fraction':issued/(issued+bypassed)}
            rows.append({'family':family,'kappa':pair['kappa'],'sites':sites,'work':work})
    write(RUN/'results/diagnostic-summary.json',{'rows':rows,'source_files':sources,'source_script':record(__file__),
          'precision':'actual BF16 sparse candidate operands; all338 complete validation blocks; integer counts pooled before fractions',
          'limits':'issued/bypassed MMA atoms include padded rows and SIMT substitution; no DRAM-byte or canonical FP16 R_model claim'})
    lines=['# BF16 site structure and executed h/z work','',
           'Each site cell is exact-zero % / empty8x16-tile %. Pool all integer counts across338 blocks and six layers before division.','',
           '| Gate | κ | A zeros / tiles % | M zeros / tiles % | H zeros / tiles % | Z zeros / tiles % | H MMA issued % | Z MMA issued % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cells=[f"{100*row['sites'][s]['exact_zero_fraction']:.3f} / {100*row['sites'][s]['empty_tile_fraction']:.3f}" for s in ['a','m','h','z']]
        lines.append(f"| {row['family']} | {row['kappa']:g} | "+' | '.join(cells)+
                     f" | {100*row['work']['h']['issued_fraction']:.3f} | {100*row['work']['z']['issued_fraction']:.3f} |")
    lines+=['','MMA percentages use the padded M8 consumer potential as denominator. Bypassed atoms include short-row SIMT substitution; integer SIMT work is retained in the machine summary.',
            'Tile gates retain fewer element zeros but more empty tiles and bypass more h/z MMA atoms. This accompanies measured full-model gains; it does not measure avoided DRAM bytes or separately attribute the gain to H versus Z.']
    (RUN/'results/diagnostic-tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
