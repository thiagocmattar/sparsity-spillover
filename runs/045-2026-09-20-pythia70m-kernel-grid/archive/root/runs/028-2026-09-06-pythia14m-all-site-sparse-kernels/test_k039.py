"""Four/eight N atoms retain bijective output coverage and K16 MMA ordering."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('atoms,candidate', [(4, 'k039'), (8, 'k040')])
def test_wider_joint_mapping(atoms, candidate):
    coordinates = []
    for bx in range(2):
        for by in range(16 // atoms):
            for thread in range(64):
                lane, warp = thread % 32, thread // 32
                row = bx * 32 + warp * 16 + lane // 4
                for atom in range(atoms):
                    for element in range(4):
                        coordinates.append((row + (element // 2) * 8,
                            by * atoms * 8 + atom * 8 + (lane % 4) * 2 + element % 2))
    assert len(coordinates) == len(set(coordinates)) == 64 * 128
    assert set(coordinates) == {(row, col) for row in range(64) for col in range(128)}
    source = (Path(__file__).parent / 'candidates' / candidate / 'joint.cu').read_text()
    assert f'col=blockIdx.y*{atoms * 8}' in source
    assert f'dim3(m/32,{16 // atoms})' in source
    assert f'accumulate<512,Skip,{atoms}>' in source
    assert f'accumulate<128,Skip,{atoms}>' in source
    assert 'for(int base=0;base<K;base+=16)' in source
