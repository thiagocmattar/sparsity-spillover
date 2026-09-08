"""Draw all 14M training variants with the existing A0-only clipping reference."""
import json

from evidence import HERE
from plots import configure, overview


if __name__ == '__main__':
    data = json.loads((HERE / 'figure_data.json').read_text(encoding='utf-8'))
    configure()
    overview(data, all_variants=True)
