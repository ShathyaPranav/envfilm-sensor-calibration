"""SensorJEPA for gas-sensor drift.

Every sample is 16 sensors x 8 features. Each sensor becomes a token (its 8 features, embedded,
plus a learned sensor-identity embedding); a small transformer reads the 16 tokens.

Pretraining (no labels, all batches given to it): a random subset of sensors is hidden from the
context encoder; a predictor must produce, for each hidden sensor, the embedding that a slowly
updated copy of the encoder (EMA, no gradient) gives that sensor when it sees all 16. Drift hits
sensors one by one, so a representation that can stand in for any sensor from the others is
built on the relations between sensors rather than on any single sensor's absolute response.
VICReg variance/covariance terms keep the embedding from collapsing.

Fine-tuning: a linear-plus-MLP head on the pooled embedding, trained on the labelled source.
Input preprocessing: each batch z-scored with its own statistics (label-free; best simple baseline).

Usage: python gd_model.py dev|final [variant]
  dev    source batch 1 -> target batch 2 (the only labels of a target ever looked at)
  final  sources 1+2 -> targets 3..10, frozen config, 5 seeds
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from gd_data import PUBLISHED, load_all

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.join(os.path.dirname(_EXP), 'results', 'gasdrift')
CFG = dict(D=64, layers=2, heads=4, mask=0.5, pre_steps=3000, pre_lr=1e-3, ft_epochs=150, ft_lr=1e-3,
           enc_lr_mult=0.3, batch=256, pretrain=1, drop=0.1)
CFG.update(json.loads(os.environ.get('GD_CFG', '{}')))


def bz(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


class Encoder(nn.Module):
    def __init__(self, D, layers, heads, drop):
        super().__init__()
        self.inp = nn.Linear(8, D)
        self.sid = nn.Parameter(torch.randn(16, D) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(D))
        el = nn.TransformerEncoderLayer(D, heads, 2 * D, drop, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, layers)
        self.norm = nn.LayerNorm(D)

    def forward(self, x, keep=None):
        t = self.inp(x.view(-1, 16, 8))
        if keep is not None:
            t = torch.where(keep[..., None], t, self.mask_tok.expand_as(t))
        return self.norm(self.tr(t + self.sid))           # B, 16, D


def vicreg(z):
    z = z.reshape(-1, z.shape[-1])
    var = torch.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
    c = z - z.mean(0)
    cov = (c.T @ c) / (len(c) - 1)
    off = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / cov.shape[0]
    return var + 0.04 * off


def pretrain(X, seed):
    torch.manual_seed(seed)
    enc = Encoder(CFG['D'], CFG['layers'], CFG['heads'], CFG['drop']).to(DEV)
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(CFG['D'], 2 * CFG['D']), nn.GELU(), nn.Linear(2 * CFG['D'], CFG['D'])).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=CFG['pre_lr'], weight_decay=1e-4)
    X = torch.tensor(X, dtype=torch.float32, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for step in range(CFG['pre_steps']):
        i = torch.randint(0, len(X), (CFG['batch'],), generator=g).to(DEV)
        x = X[i]
        keep = torch.rand(len(x), 16, generator=g).to(DEV) > CFG['mask']
        ctx = enc(x, keep)
        with torch.no_grad():
            z = tgt(x)
            z = (z - z.mean((0, 1))) / (z.std((0, 1)) + 1e-4)
        p = pred(ctx)
        hid = ~keep
        loss = nn.functional.smooth_l1_loss(p[hid], z[hid]) + vicreg(ctx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            m = 0.996 + 0.004 * step / CFG['pre_steps']
            for a, b in zip(tgt.parameters(), enc.parameters()):
                a.mul_(m).add_(b, alpha=1 - m)
    return enc


class Classifier(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        D = CFG['D']
        self.head = nn.Sequential(nn.Linear(2 * D, 128), nn.GELU(), nn.Dropout(CFG['drop']), nn.Linear(128, 6))

    def forward(self, x):
        h = self.enc(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], -1))


def finetune(enc, Xs, ys, seed):
    torch.manual_seed(seed)
    m = Classifier(enc).to(DEV)
    opt = torch.optim.AdamW([{'params': m.enc.parameters(), 'lr': CFG['ft_lr'] * CFG['enc_lr_mult']},
                             {'params': m.head.parameters(), 'lr': CFG['ft_lr']}], weight_decay=1e-4)
    X = torch.tensor(Xs, dtype=torch.float32, device=DEV)
    y = torch.tensor(ys, device=DEV)
    g = torch.Generator().manual_seed(seed)
    for _ in range(CFG['ft_epochs']):
        m.train()
        perm = torch.randperm(len(X), generator=g).to(DEV)
        for j in range(0, len(X), CFG['batch']):
            k = perm[j:j + CFG['batch']]
            loss = nn.functional.cross_entropy(m(X[k]), y[k])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def accuracy(m, X, y):
    m.eval()
    p = m(torch.tensor(X, dtype=torch.float32, device=DEV)).argmax(-1).cpu().numpy()
    return 100 * float((p == y).mean())


def run(sources, targets, seed):
    B = load_all()
    Z = {k: bz(B[k][0]) for k in B}
    Xs = np.vstack([Z[k] for k in sources])
    ys = np.concatenate([B[k][1] for k in sources])
    if CFG['pretrain']:
        pool = range(1, 11) if CFG.get('pretrain_all', 0) else sorted(set(sources) | set(targets))
        enc = pretrain(np.vstack([Z[k] for k in pool]), seed)        # features only, never labels
    else:
        enc = Encoder(CFG['D'], CFG['layers'], CFG['heads'], CFG['drop'])
    head = CFG.get('head', 'finetune')
    if head == 'finetune':
        m = finetune(enc, Xs, ys, seed)
        return {k: accuracy(m, Z[k], B[k][1]) for k in targets}
    # frozen encoder, linear probe: logistic regression on [embedding] or [raw features, embedding]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    enc = enc.to(DEV).eval()

    @torch.no_grad()
    def emb(X):
        h = enc(torch.tensor(X, dtype=torch.float32, device=DEV))
        return torch.cat([h.mean(1), h.amax(1)], -1).cpu().numpy()

    def feats(X):
        e = emb(X)
        return np.hstack([X, e]) if head == 'probe+raw' else e
    sc = StandardScaler().fit(feats(Xs))
    lr = LogisticRegression(max_iter=5000, C=CFG.get('C', 1.0)).fit(sc.transform(feats(Xs)), ys)
    return {k: 100 * float((lr.predict(sc.transform(feats(Z[k]))) == B[k][1]).mean()) for k in targets}


def main():
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    tag = os.environ.get('GD_TAG', 'run')
    seeds = int(os.environ.get('GD_SEEDS', '3' if mode == 'dev' else '5'))
    t = time.time()
    if mode == 'dev':
        a = [run([1], [2], s)[2] for s in range(seeds)]
        print('DEV 1->2  %s  cfg %s  acc %s  mean %.2f  (%.0f s)' % (tag, json.dumps(CFG), ' '.join('%.2f' % v for v in a),
                                                                    np.mean(a), time.time() - t), flush=True)
        return
    B = load_all()
    sizes = np.array([len(B[k][1]) for k in range(3, 11)])
    rows = [run([1, 2], list(range(3, 11)), s) for s in range(seeds)]
    A = np.array([[r[k] for k in range(3, 11)] for r in rows])
    m = A.mean(0)
    print('FINAL %s cfg %s' % (tag, json.dumps(CFG)))
    print('  per batch 3..10: ' + ' '.join('%6.2f' % v for v in m))
    print('  weighted %.2f (seed sd %.2f)  mean %.2f' % (np.average(m, weights=sizes),
                                                        np.std([np.average(a, weights=sizes) for a in A]), m.mean()))
    pub = np.array(PUBLISHED['AMDS-PFFA'])
    print('  AMDS-PFFA weighted %.2f; batches better than AMDS-PFFA: %d of 8' % (np.average(pub, weights=sizes),
                                                                              int((m > pub).sum())))
    json.dump(dict(cfg=CFG, acc=A.tolist()), open(os.path.join(OUT, 'final_%s.json' % tag), 'w'))


if __name__ == '__main__':
    main()
