"""Separate the same-checkpoint native gap from the conditional h/z contribution."""
import math
from io_utils import RUN, read, write

data = read(RUN/'results/hz-latency.json')
times = {mode: row['host_geomean_ms'] for mode, row in data['latency'].items()}
a,b,c,d,native,frozen = (times[m] for m in ['A_graph','B_graph','C_graph','D_graph','native_graph','frozen_graph'])
total = native-a
sparse = d-a
other = native-d
assert total > 0 and math.isclose(total, sparse+other, rel_tol=1e-14)
result = {
    'source': 'results/hz-latency.json',
    'same_checkpoint_native_ms':native,
    'total_native_to_A_saved_ms':total,
    'total_native_to_A_reduction_percent':100*total/native,
    'native_over_A_speedup':native/a,
    'joint_hz_saved_ms':sparse,
    'hz_share_of_same_checkpoint_native_gap_percent':100*sparse/total,
    'net_other_implementation_saved_ms':other,
    'z_saved_with_h_disabled_ms':d-b,
    'h_saved_with_z_disabled_ms':d-c,
    'untouched_K050_minus_control_A_ms':frozen-a,
    'untouched_K050_over_control_A_ratio':frozen/a,
    'joint_saved_relative_to_untouched_K050_ms':d-frozen,
    'joint_reduction_relative_to_untouched_K050_percent':100*(d-frozen)/d,
    'interpretation': 'The native gap is for this same thresholded checkpoint and session. Other includes all differences retained in D; it is not a separately identified dense-optimization effect. h and z conditional effects are not additive.'
}
write(RUN/'results/engineering-comparison.json', result)
print(result)
