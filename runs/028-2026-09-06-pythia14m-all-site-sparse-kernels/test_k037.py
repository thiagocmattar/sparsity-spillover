"""The warp-empty return preserves the selected MMA sequence and counts."""
import itertools


def test_empty_fragment_guard_preserves_mma_order_and_counts():
    for m_count,n_count in [(1,4),(2,3)]:
        for values in itertools.product([False,True],repeat=m_count+n_count):
            a,b=values[:m_count],values[m_count:]
            old=[(m,n_count-1-n if m%2 else n) for m in range(m_count) for n in range(n_count)
                 if a[m] and b[n_count-1-n if m%2 else n]]
            new=[] if not any(a) or not any(b) else [(m,n_count-1-n if m%2 else n)
                 for m in range(m_count) for n in range(n_count) if a[m] and b[n_count-1-n if m%2 else n]]
            assert new==old
            assert m_count*n_count-len(new)==m_count*n_count-sum(a)*sum(b)
