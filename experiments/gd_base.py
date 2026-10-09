"""Headroom on the gas-drift benchmark: simple baselines vs the published table.

Sources: batches 1 and 2 (labelled). Each target batch 3..10: features available, labels hidden.
  src-z    classifier on features z-scored with source statistics
  batch-z  every batch z-scored with its OWN statistics (label-free, uses the target features)
  CORAL    target features whitened and re-coloured to the source covariance (label-free)
Classifiers: logistic regression, RBF SVM, MLP.
"""
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from gd_data import PUBLISHED, load_all

warnings.filterwarnings('ignore')


def zfit(X):
    return X.mean(0), X.std(0) + 1e-9


def coral(Xs, Xt, eps=1.0):
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])

    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-9, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    return (Xt - Xt.mean(0)) @ msqrt(Ct, inv=True) @ msqrt(Cs) + Xs.mean(0)


def clfs():
    return {'LR': lambda: LogisticRegression(max_iter=3000, C=1.0),
            'SVM': lambda: SVC(C=10, gamma='scale'),
            'MLP': lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0)}


def main():
    B = load_all()
    Xs = np.vstack([B[1][0], B[2][0]])
    ys = np.concatenate([B[1][1], B[2][1]])
    sizes = np.array([len(B[k][1]) for k in range(3, 11)])
    res = {}
    for norm in ('src-z', 'batch-z', 'CORAL'):
        for cn, mk in clfs().items():
            accs = []
            if norm == 'src-z':
                mu, sd = zfit(Xs)
                m = mk().fit((Xs - mu) / sd, ys)
            elif norm == 'batch-z':
                Zs = np.vstack([(B[k][0] - zfit(B[k][0])[0]) / zfit(B[k][0])[1] for k in (1, 2)])
                m = mk().fit(Zs, ys)
            else:
                mu, sd = zfit(Xs)
                m = mk().fit((Xs - mu) / sd, ys)
            for k in range(3, 11):
                Xt, yt = B[k]
                if norm == 'src-z':
                    Z = (Xt - mu) / sd
                elif norm == 'batch-z':
                    Z = (Xt - zfit(Xt)[0]) / zfit(Xt)[1]
                else:
                    Z = (coral((Xs - mu) / sd, (Xt - mu) / sd))
                accs.append(100 * float((m.predict(Z) == yt).mean()))
            res['%s %s' % (norm, cn)] = accs
            print('%-12s ' % ('%s %s' % (norm, cn)) + ' '.join('%6.2f' % a for a in accs) +
                  '   mean %.2f  weighted %.2f' % (np.mean(accs), np.average(accs, weights=sizes)), flush=True)
    print('\npublished (targets 3..10):')
    for n, a in PUBLISHED.items():
        print('%-12s ' % n + ' '.join('%6.2f' % v for v in a) + '   mean %.2f  weighted %.2f'
              % (np.mean(a), np.average(a, weights=sizes)))


if __name__ == '__main__':
    main()
