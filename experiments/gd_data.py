"""UCI Gas Sensor Array Drift (Vergara et al. 2012): 13,910 samples, 16 metal-oxide sensors x 8
features = 128, 6 gases, 10 batches over 36 months.

Protocol of AMDS-PFFA (arXiv 2409.13167) and the methods in its Table: batches 1 and 2 labelled
(source); each of batches 3..10 is a target whose LABELS are hidden but whose FEATURES may be
used during training (unsupervised domain adaptation); accuracy is reported per target batch.
"""
import os

import numpy as np

DIR = os.path.join(os.environ.get('GD_DATA', 'C:/Users/luckf/dq_data/gasdrift'), 'Dataset')
PUBLISHED = {   # per target batch 3..10, from AMDS-PFFA Table (sources: batches 1, 2)
    'AMDS-PFFA': [99.05, 88.46, 98.44, 93.31, 86.17, 89.24, 71.06, 66.83],
    'DAELM': [96.53, 82.61, 81.47, 84.97, 71.89, 78.10, 87.02, 57.42],
    'DRCA': [92.69, 87.58, 95.90, 86.52, 60.25, 62.24, 72.34, 52.00],
    'DANP': [95.80, 71.02, 95.20, 91.02, 65.54, 78.82, 77.14, 54.24],
    'DCLSL': [94.76, 81.98, 94.92, 77.00, 63.36, 85.03, 68.08, 63.05],
    'TDACNN': [83.83, 77.64, 75.63, 74.36, 62.08, 75.10, 60.85, 50.88],
}


def load_batch(k):
    X, y = [], []
    for line in open(os.path.join(DIR, 'batch%d.dat' % k)):
        parts = line.split()
        if not parts:
            continue
        y.append(int(parts[0].split(';')[0]) - 1)
        row = np.zeros(128)
        for p in parts[1:]:
            i, v = p.split(':')
            row[int(i) - 1] = float(v)
        X.append(row)
    return np.array(X), np.array(y)


def load_all():
    return {k: load_batch(k) for k in range(1, 11)}


if __name__ == '__main__':
    B = load_all()
    for k, (X, y) in B.items():
        print('batch %2d: %4d samples, gases %s' % (k, len(y), np.bincount(y, minlength=6)))
