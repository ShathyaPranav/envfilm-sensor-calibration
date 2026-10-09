"""Sensor-level drift layer, fitted without labels on each target batch.

A frozen classifier is trained on the labelled source batches (features z-scored with source
statistics). For a target batch, a drift layer with few parameters maps raw target features back
toward the source sensors' behaviour:
  sensor-gain  one multiplicative gain per sensor on all 8 of its features (16 numbers)
  sensor-g+o   that gain plus an additive offset on the sensor's 2 steady-state features (32)
  feature-aff  an independent gain and offset per feature (256; the most flexible)
The parameters are fitted on the UNLABELLED target batch by one of:
  moments   match the per-feature mean and spread of the corrected target to the source
  entropy   low prediction entropy plus a diversity term (mean prediction not collapsed)
  both      sum of the two
Usage: python gd_drift.py dev  (source 1 -> target 2; selection happens here)
       python gd_drift.py final <param> <objective>  (sources 1+2 -> targets 3..10)
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from gd_data import PUBLISHED, load_all

warnings.filterwarnings('ignore')
torch.manual_seed(0)


def classifier(kind, Z, y, seed=0):
    if kind == 'LR':
        m = LogisticRegression(max_iter=5000).fit(Z, y)
        W = torch.tensor(m.coef_, dtype=torch.float32)
        b = torch.tensor(m.intercept_, dtype=torch.float32)
        return lambda z: z @ W.T + b
    m = MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, random_state=seed).fit(Z, y)
    Ws = [torch.tensor(w, dtype=torch.float32) for w in m.coefs_]
    bs = [torch.tensor(c, dtype=torch.float32) for c in m.intercepts_]

    def f(z):
        h = torch.relu(z @ Ws[0] + bs[0])
        return h @ Ws[1] + bs[1]
    return f


def adapt(X, mu, sd, f, param, objective, steps=1500):
    X = torch.tensor(X, dtype=torch.float32)
    M, S = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
    sens = torch.arange(128) // 8
    steady = torch.tensor([(i % 8) in (0, 1) for i in range(128)], dtype=torch.float32)
    lg, off = torch.zeros(16, requires_grad=True), torch.zeros(16, requires_grad=True)
    fa, fo = torch.zeros(128, requires_grad=True), torch.zeros(128, requires_grad=True)

    def z():
        if param == 'sensor-gain':
            Xc = X * torch.exp(lg[sens])
        elif param == 'sensor-g+o':
            Xc = X * torch.exp(lg[sens]) + steady * off[sens] * S
        else:
            Xc = X * torch.exp(fa) + fo * S
        return (Xc - M) / S
    params = {'sensor-gain': [lg], 'sensor-g+o': [lg, off], 'feature-aff': [fa, fo]}[param]
    opt = torch.optim.Adam(params, lr=0.03)
    for _ in range(steps):
        zz = z()
        loss = 0.0
        if objective in ('moments', 'both'):
            loss = loss + (zz.mean(0) ** 2).mean() + ((zz.std(0) - 1) ** 2).mean()
        if objective in ('entropy', 'both'):
            p = torch.softmax(f(zz), -1)
            ent = -(p * torch.log(p + 1e-9)).sum(-1).mean()
            pm = p.mean(0)
            div = (pm * torch.log(pm + 1e-9)).sum()            # negative entropy of the mean prediction
            loss = loss + ent + div
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return f(z()).argmax(-1).numpy()


def evaluate(sources, targets, param, objective, kind):
    B = load_all()
    Xs = np.vstack([B[k][0] for k in sources])
    ys = np.concatenate([B[k][1] for k in sources])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    f = classifier(kind, (Xs - mu) / sd, ys)
    out = {}
    for t in targets:
        pred = adapt(B[t][0], mu, sd, f, param, objective)
        out[t] = 100 * float((pred == B[t][1]).mean())
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    if mode == 'dev':
        for kind in ('LR', 'MLP'):
            for param in ('sensor-gain', 'sensor-g+o', 'feature-aff'):
                for obj in ('moments', 'entropy', 'both'):
                    a = evaluate([1], [2], param, obj, kind)[2]
                    print('DEV 1->2  %-4s %-12s %-8s %.2f' % (kind, param, obj, a), flush=True)
        return
    param, obj, kind = sys.argv[2], sys.argv[3], sys.argv[4]
    B = load_all()
    sizes = np.array([len(B[k][1]) for k in range(3, 11)])
    r = evaluate([1, 2], list(range(3, 11)), param, obj, kind)
    a = [r[k] for k in range(3, 11)]
    print('FINAL %s %s %s: ' % (kind, param, obj) + ' '.join('%6.2f' % v for v in a) +
          '  weighted %.2f' % np.average(a, weights=sizes))
    p = PUBLISHED['AMDS-PFFA']
    print('AMDS-PFFA: ' + ' '.join('%6.2f' % v for v in p) + '  weighted %.2f; batches better: %d of 8'
          % (np.average(p, weights=sizes), int((np.array(a) > np.array(p)).sum())))


if __name__ == '__main__':
    main()
