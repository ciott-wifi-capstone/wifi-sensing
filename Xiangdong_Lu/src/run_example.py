"""Run the actual fixed estimator on a public SYNTHETIC covariance fixture."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
import replay as r


def load_covariance(path):
    values = np.empty((6, 30, 3, 3), dtype=complex)
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or ()) != {'state','tone','row','col','real','imag'}:
            raise ValueError('Expected state,tone,row,col,real,imag CSV fields')
        for line in reader:
            state = line['state']
            if state not in r.STATES:
                raise ValueError('Unknown RF state: ' + state)
            key = (r.STATES.index(state),int(line['tone']),int(line['row']),int(line['col']))
            if not (0 <= key[1] < 30 and 0 <= key[2] < 3 and 0 <= key[3] < 3):
                raise ValueError('Covariance index out of range')
            if key in seen:
                raise ValueError('Duplicate covariance entry')
            seen.add(key)
            values[key] = complex(float(line['real']), float(line['imag']))
    if len(seen) != values.size or not np.isfinite(values).all():
        raise ValueError('Missing or non-finite covariance entries')
    if not np.allclose(values, values.swapaxes(-1,-2).conj(), atol=1e-10):
        raise ValueError('Covariance must be Hermitian')
    if np.linalg.eigvalsh(values).min() < -1e-9:
        raise ValueError('Covariance must be positive semidefinite')
    return values


def estimate(covariance):
    # Diagonal-only synthetic empty-room background. It is not measured CSI.
    empty = np.broadcast_to(np.eye(3) * .05, covariance.shape).copy()
    datasets = {
        'run1': {'cov':empty, 'alt':empty},
        'run2': {'cov':covariance, 'alt':covariance},
        'run3': {'cov':empty * 1.01, 'alt':empty * 1.01},
    }
    tables = defaultdict(list)
    r.estimate_point('SYNTHETIC', datasets, None, None, tables)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tables = estimate(load_covariance(args.input))
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use an empty output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    selected = [dict(row, synthetic=True) for row in tables['blind_predictions']
                if row['mapping']=='cov' and row['run']=='run2']
    r.csvout(args.out.resolve()/'estimates.csv', selected)
    r.csvout(args.out.resolve()/'spectra.csv', [dict(row,synthetic=True) for row in tables['spectra']
              if row['mapping']=='cov' and row['run']=='run2'])
    for row in selected:
        print(f"SYNTHETIC {row['branch']}: {row['method']} = {row['angle_deg']:+.0f} deg")


if __name__=='__main__':
    main()
