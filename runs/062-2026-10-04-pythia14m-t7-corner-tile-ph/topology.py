"""Register only the approved Run062 operator; A7-Z-POST is already canonical."""
def register():
    from tile_gate import register as register_corner_gate
    register_corner_gate()
