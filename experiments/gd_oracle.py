"""Idea-8 oracle (results/tesla/PREREG_oracles.md): is gas-sensor drift low-dimensional?

Classifier: logistic regression on batches 1+2, features z-scored with source statistics, then frozen.
For each target batch, the ORACLE fits (with the target labels, diagnosis only) a small drift
correction applied to the raw target features before the frozen classifier:
  sensor-gain   16 numbers: one multiplicative gain per sensor, shared by its 8 features
  sensor-g+o    32 numbers: per-sensor gain on all 8 features plus an offset on the 2 steady-state features
  feature-aff   256 numbers: an independent gain and offset per feature (much more flexible)
If even the 16-number oracle cannot pass 83.21% weighted, drift is not low-dimensional per sensor.
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from gd_data import PUBLISHED, load_all

warnings.filterwarnings('ignore')


def main():
    B = load_all()
    Xs = np.vstack([B[1][0], B[2][0]])
    ys = np.concatenate([B[1][1], B[2][1]])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    lr = LogisticRegression(max_iter=5000).fit((Xs - mu) / sd, ys)
    W = torch.tensor(lr.coef_, dtype=torch.float32)
    b = torch.tensor(lr.intercept_, dtype=torch.float32)
    M, S = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    sens = torch.arange(128) // 8
    steady = torch.tensor([(i % 8) in (0, 1) for i in range(128)])
    sizes, res = [], {k: [] for k in ('no correction', 'sensor-gain', 'sensor-g+o', 'feature-aff')}
    for t in range(3, 11):
        X = torch.tensor(B[t][0], dtype=torch.float32)
        y = torch.tensor(B[t][1])
        sizes.append(len(y))
        for kind in res:
            lg = torch.zeros(16, requires_grad=True)
            off = torch.zeros(16, requires_grad=True)
            fa = torch.zeros(128, requires_grad=True)
            fo = torch.zeros(128, requires_grad=True)

            def correct():
                if kind == 'sensor-gain':
                    return X * torch.exp(lg[sens])
                if kind == 'sensor-g+o':
                    return X * torch.exp(lg[sens]) + steady * off[sens] * S
                if kind == 'feature-aff':
                    return X * torch.exp(fa) + fo * S
                return X
            params = {'sensor-gain': [lg], 'sensor-g+o': [lg, off], 'feature-aff': [fa, fo]}.get(kind, [])
            if params:
                opt = torch.optim.Adam(params, lr=0.03)
                for _ in range(1500):
                    logits = ((correct() - M) / S) @ W.T + b
                    loss = nn.functional.cross_entropy(logits, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                acc = 100 * float((((((correct() - M) / S) @ W.T + b).argmax(-1)) == y).float().mean())
            res[kind].append(acc)
    sizes = np.array(sizes)
    for kind, a in res.items():
        print('%-14s ' % kind + ' '.join('%6.2f' % v for v in a) + '   weighted %.2f' % np.average(a, weights=sizes))
    p = PUBLISHED['AMDS-PFFA']
    print('%-14s ' % 'AMDS-PFFA' + ' '.join('%6.2f' % v for v in p) + '   weighted %.2f' % np.average(p, weights=sizes))


if __name__ == '__main__':
    main()
