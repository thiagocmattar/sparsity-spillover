"""Full-forward native-base comparison; incomplete and failed trials stay visible."""
from collections import defaultdict
from statistics import median
from io_utils import RUN,read,record,write


def main():
    groups=defaultdict(list);development=[];failures=[];sources=[]
    for path in sorted((RUN/'artifacts/attempts').glob('*/result.json')):
        row=read(path);a=row['arguments'];sources.append(record(path))
        if row['status']!='complete' or not row.get('qualified'):
            failures.append({'attempt':path.parent.name,'condition':row['condition'],
                             'candidate':row['candidate'],'status':row['status'],'error':row.get('error')})
            continue
        if a.get('smoke'):continue
        if a.get('development'):
            development.append({'attempt':path.parent.name,'candidate':row['candidate'],
                'condition':row['condition'],'ms':row['timing']['candidate_graph']['median_host_ms'],
                'loss':row['loss']['candidate_graph']})
            continue
        assert row['validation_blocks']==338
        groups[(row['condition'],row['candidate'])].append((path,row))
    table=[]
    for (condition,candidate),items in sorted(groups.items()):
        values=[r['timing']['candidate_graph']['median_host_ms'] for _,r in items]
        native=[r['timing']['native_graph']['median_host_ms'] for _,r in items]
        table.append({'condition':condition,'candidate':candidate,'processes':len(items),
                      'candidate_ms':median(values),'candidate_process_range_ms':[min(values),max(values)],
                      'native_same_checkpoint_ms':median(native),
                      'native_process_range_ms':[min(native),max(native)],
                      'maximum_absolute_loss_delta':max(abs(r['loss_delta']['candidate_graph']) for _,r in items),
                      'all_processes_complete':len(items)>=3})
    comparison=None
    selection=RUN/'provenance/final-selection.json'
    if selection.exists():
        selected=read(selection)['candidate']
        keyed={(r['condition'],r['candidate']):r for r in table}
        keys=[('m14-c01','full'),('m14-c20','full'),('c00',selected),('c21',selected)]
        if all(k in keyed and keyed[k]['all_processes_complete'] for k in keys):
            base14,best14,base70,best70=[keyed[k] for k in keys]
            b14=base14['native_same_checkpoint_ms'];b70=base70['native_same_checkpoint_ms']
            l14=best14['candidate_ms'];l70=best70['candidate_ms']
            s14=b14/l14;s70=b70/l70
            # This conservative range uses process extrema; it is not a confidence interval.
            conservative70=base70['native_process_range_ms'][0]/best70['candidate_process_range_ms'][1]
            optimistic14=base14['native_process_range_ms'][1]/best14['candidate_process_range_ms'][0]
            comparison={'model14':{'native_base_ms':b14,'candidate_ms':l14,'speedup':s14,'latency_reduction_percent':100*(1-l14/b14)},
                        'model70':{'native_base_ms':b70,'candidate_ms':l70,'speedup':s70,'latency_reduction_percent':100*(1-l70/b70)},
                        'higher_fractional_gain_than_fresh14':s70>s14,
                        'higher_than_historical14':s70>1.42650079,
                        'separated_process_ranges':conservative70>optimistic14,
                        'range_interpretation':'Extrema across three process medians, not a statistical confidence interval.',
                        'reference':'Each size uses native T0/P0 full-logit graph latency on this same GPU; never the sparse implementation at T0/P0.'}
    write(RUN/'results/summary.json',{'full_validation':table,'development':development,
        'failures':failures,'comparison':comparison,'sources':sources,'script':record(__file__),
        'limits':'Development timings use16 training blocks; final timing uses64 inputs x7 passes x3 fresh processes. Profile durations and skip toggles do not replace the native-base comparison.'})
    print({'completed_groups':len(table),'development_results':len(development),'failures':len(failures),'comparison':comparison})

if __name__=='__main__':main()
