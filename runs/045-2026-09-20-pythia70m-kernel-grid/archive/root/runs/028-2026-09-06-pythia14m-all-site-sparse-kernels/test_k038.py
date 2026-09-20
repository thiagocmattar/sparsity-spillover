"""The wider N tile maps every output element exactly once."""
def test_m32n16_output_mapping():
    coordinates=[]
    for bx in range(2):
        for by in range(8):
            for thread in range(64):
                lane,warp=thread%32,thread//32
                row=bx*32+warp*16+lane//4
                for atom in range(2):
                    for e in range(4):
                        coordinates.append((row+(e//2)*8,by*16+atom*8+(lane%4)*2+e%2))
    assert len(coordinates)==64*128
    assert len(set(coordinates))==len(coordinates)
    assert set(coordinates)=={(row,col) for row in range(64) for col in range(128)}
