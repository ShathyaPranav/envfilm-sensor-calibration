"""A calibration network for a sensor fleet: shared body, unit identity, label-free context.

Same protocol and test as sec_protocol.py (targets' second co-location, odd days). The
network is trained on every unit's first co-location plus the anchors' deployment labels.

  input at hour t for unit u
    x        the unit's channels now, aligned by its first-co-location mean and spread
    C        the same channels over the past week (168 hours, with a missing-value mask)
    u        the unit's identity
  context encoder   dilated 1-D convolutions over C, then mean and max pooling -> c
  unit embedding    e_u, 8 numbers, learned (targets: only from their first co-location);
                    dropped at random during training (p = 0.3) so the body does not lean on it
  body              MLP on x, with every hidden layer scaled and shifted by FiLM([c, e_u])
  loss              mean squared error on the reference

The context encoder sees only raw sensor readings, never a reference, so it is available
for every unit at every hour. What it may learn from the anchors is how a sensor's recent
behaviour (its baselines, its range) tells the calibration to shift.

Hyperparameters are fixed in advance, not tuned on the test. Early stopping uses the last
30 days of each anchor's deployment, held out from training.

PASS MARK, fixed before the first run: the network beats the best simple deployable arm
of sec_protocol.py (per task, the better of own-first in any model class and the pooled
+ctx arms) by at least 5% in median RMSE and on at least two thirds of target units, on at
least two pollutants in Antwerp; then the same code, unchanged, must hold in Oslo and
Zagreb. Ablations (no context, no unit embedding) are reported next to it.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import TASKS
from sec_protocol import OUT, build, report, score

warnings.filterwarnings('ignore')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN = 168
HOLD = pd.Timedelta(days=30)
CFG = dict(hidden=128, ctx=32, emb=8, emb_drop=0.3, lr=2e-3, wd=1e-4, batch=512, epochs=60, patience=8, cosine=0,
           residual=0)
CFG.update(eval(os.environ.get('SEC_CFG', '{}')))      # variants are logged in results/sec/DESIGN_LOG.md


class Net(nn.Module):
    def __init__(self, f, n_units, use_ctx=True, use_emb=True):
        super().__init__()
        H, K, E = CFG['hidden'], CFG['ctx'], CFG['emb']
        self.use_ctx, self.use_emb = use_ctx, use_emb
        self.enc = nn.Sequential(
            nn.Conv1d(2 * f, K, 5, padding=2), nn.GELU(),
            nn.Conv1d(K, K, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(K, K, 5, padding=8, dilation=4), nn.GELU())
        self.emb = nn.Embedding(n_units, E)
        nn.init.zeros_(self.emb.weight)
        cond = 2 * K * use_ctx + E * use_emb
        self.l1, self.l2, self.l3 = nn.Linear(f, H), nn.Linear(H, H), nn.Linear(H, H)
        self.film = nn.Linear(max(cond, 1), 6 * H)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.out = nn.Linear(H, 1)
        self.act = nn.GELU()

    def forward(self, x, ctx, unit, drop_emb=False):
        parts = []
        if self.use_ctx:
            h = self.enc(ctx)
            parts += [h.mean(-1), h.amax(-1)]
        if self.use_emb:
            e = self.emb(unit)
            if drop_emb:
                e = e * (torch.rand(e.shape[0], 1, device=e.device) > CFG['emb_drop']).float()
            parts.append(e)
        cond = torch.cat(parts, -1) if parts else torch.zeros(x.shape[0], 1, device=x.device)
        g = self.film(cond).chunk(6, -1)
        h = x
        for i, lin in enumerate((self.l1, self.l2, self.l3)):
            h = self.act(lin(h) * (1 + g[2 * i]) + g[2 * i + 1])
        return self.out(h).squeeze(-1)


def grids(D):
    """Per unit: aligned channels on a continuous hourly grid, and its start hour."""
    G = {}
    for n, raw in D['raw'].items():
        mu, sd = D['stat'][n]
        idx = pd.date_range(raw.index.min(), raw.index.max(), freq='h')
        a = ((raw.reindex(idx) - mu) / sd).to_numpy(np.float32)
        a = np.clip(a, -20, 20)
        G[n] = (idx[0], a)
    return G


def samples(D, G, units, frames):
    """(unit index, grid row) of every labelled hour in `frames`, and its target."""
    U, R, Y = [], [], []
    for n, df in frames.items():
        t0, a = G[n]
        rows = ((df.hour - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        U.append(np.full(len(rows), units[n]))
        R.append(rows)
        Y.append(df[D['ref']].to_numpy(np.float32))
    return np.concatenate(U), np.concatenate(R), np.concatenate(Y)


class Batcher:
    def __init__(self, G, order):
        f = next(iter(G.values()))[1].shape[1]
        self.pad = {}
        for n in order:
            a = G[n][1]
            p = np.concatenate([np.full((WIN, f), np.nan, np.float32), a])
            self.pad[n] = torch.tensor(p, device=DEV)
        self.order = order
        self.big = torch.nn.utils.rnn.pad_sequence([self.pad[n] for n in order], batch_first=True,
                                                   padding_value=float('nan'))

    def __call__(self, u, r):
        u = torch.as_tensor(u, device=DEV)
        r = torch.as_tensor(r, device=DEV) + WIN
        x = self.big[u, r]
        off = torch.arange(-WIN, 0, device=DEV)
        c = self.big[u[:, None], r[:, None] + off[None, :]]
        m = torch.isfinite(c)
        c = torch.where(m, c, torch.zeros_like(c))
        ctx = torch.cat([c, m.float()], -1).transpose(1, 2)
        x = torch.nan_to_num(x, 0.0)
        return x, ctx, u


def base_grids(D, G):
    """Each unit's own first-co-location calibration (the standard practice), evaluated on its
    whole hourly grid. On the first co-location itself the values are out-of-fold (5 folds by
    day), so the residual the network learns there is an honest one."""
    from sec_protocol import make
    feats, ref = D['feats'], D['ref']
    out = {}
    for n, (t0, a) in G.items():
        raw = D['raw'][n]
        idx = pd.date_range(t0, periods=len(a), freq='h')
        X = raw.reindex(idx)
        okx = X.notna().all(axis=1).to_numpy()
        b = np.full(len(idx), np.nan, np.float32)
        f = D['first'][n]
        b[okx] = make('mlp').fit(f[feats], f[ref]).predict(X[okx])
        fold = (f.hour.dt.dayofyear % 5).to_numpy()
        rows = ((f.hour - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        for k in range(5):
            m = fold == k
            b[rows[m]] = make('mlp').fit(f[feats][~m], f[ref][~m]).predict(f[feats][m])
        out[n] = b
    return out


def train(D, use_ctx=True, use_emb=True, seed=0, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    G = grids(D)
    base = None
    if CFG['residual']:
        todo = {n: g for n, g in G.items() if (D['task'], n) not in _BASE}
        if todo:
            for n, b in base_grids(D, todo).items():
                _BASE[(D['task'], n)] = b
        base = {n: _BASE[(D['task'], n)] for n in G}
        sc = float(pd.concat([f[D['ref']] for f in D['first'].values()]).std())
        G = {n: (t0, np.concatenate([a, (base[n] / sc)[:, None]], 1)) for n, (t0, a) in G.items()}
    order = sorted(G)
    units = {n: i for i, n in enumerate(order)}
    fit_frames = dict(D['first'])
    val_frames = {}
    for a, d in D['anchor'].items():
        cut = d.hour.max() - HOLD
        fit_frames[a + '|dep'] = d[d.hour < cut]
        val_frames[a] = d[d.hour >= cut]
    G2 = dict(G)
    for k in list(fit_frames):
        if '|dep' in k:
            G2[k] = G[k.split('|')[0]]
            units[k] = units[k.split('|')[0]]
    u_tr, r_tr, y_tr = samples(D, G2, units, fit_frames)
    u_va, r_va, y_va = samples(D, G, units, val_frames)
    ok = (r_tr >= 0)
    u_tr, r_tr, y_tr = u_tr[ok], r_tr[ok], y_tr[ok]
    b_tr = b_va = 0.0
    if base is not None:
        bb = [base[n] for n in order]
        b_tr = np.array([bb[u][r] for u, r in zip(u_tr, r_tr)], np.float32)
        b_va = np.array([bb[u][r] for u, r in zip(u_va, r_va)], np.float32)
        k = np.isfinite(b_tr)
        u_tr, r_tr, y_tr, b_tr = u_tr[k], r_tr[k], y_tr[k], b_tr[k]
        b_va = np.nan_to_num(b_va, nan=float(np.nanmean(b_va)))
        y_tr = y_tr - b_tr
    ys, ym = float(y_tr.std()), float(y_tr.mean())
    B = Batcher(G, order)
    f = G[order[0]][1].shape[1]
    net = Net(f, len(order), use_ctx, use_emb).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    yt = torch.tensor((y_tr - ym) / ys, device=DEV)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['epochs']) if CFG['cosine'] else None
    best, best_state, bad = np.inf, None, 0
    for ep in range(CFG['epochs']):
        net.train()
        perm = np.random.permutation(len(y_tr))
        for i in range(0, len(perm), CFG['batch']):
            j = perm[i:i + CFG['batch']]
            x, ctx, u = B(u_tr[j], r_tr[j])
            loss = ((net(x, ctx, u, drop_emb=True) - yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if sched:
            sched.step()
        v = np.sqrt(np.mean((predict(net, B, u_va, r_va) * ys + ym + b_va - y_va) ** 2))
        if verbose:
            print('    epoch %2d  val RMSE %.3f' % (ep, v), flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG['patience']:
                break
    net.load_state_dict(best_state)
    preds = {}
    for n, t in D['test'].items():
        _, rr, _ = samples(D, G, units, {n: t})
        bt = 0.0 if base is None else np.nan_to_num(base[n][rr], nan=float(np.nanmean(base[n])))
        preds[n] = predict(net, B, np.full(len(rr), units[n]), rr) * ys + ym + bt
    return preds, best


_BASE = {}


@torch.no_grad()
def predict(net, B, u, r):
    net.eval()
    out = []
    for i in range(0, len(u), 4096):
        x, ctx, uu = B(u[i:i + 4096], r[i:i + 4096])
        out.append(net(x, ctx, uu).cpu().numpy())
    return np.concatenate(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    cities = [a for a in sys.argv[1:] if a in ('Antwerp', 'Oslo', 'Zagreb')] or ['Antwerp']
    tasks = [a for a in sys.argv[1:] if a in TASKS] or ['NO', 'NO2', 'PM2.5']
    seeds = int(os.environ.get('SEC_SEEDS', '3'))
    variants = {'net': (True, True), 'net -ctx': (False, True), 'net -emb': (True, False)}
    keep = os.environ.get('SEC_VARIANTS')
    if keep:
        variants = {k: v for k, v in variants.items() if k in keep.split(',')}
    tag = os.environ.get('SEC_TAG', 'v1')
    allrows = []
    for city in cities:
        for task in tasks:
            D = build(city, task)
            if len(D['test']) < 3 or not D['anchor']:
                print('%s %s: %d targets, %d anchors, skipped' % (city, task, len(D['test']), len(D['anchor'])))
                continue
            P = {}
            for name, (c, e) in variants.items():
                runs = []
                for s in range(seeds):
                    t = time.time()
                    p, v = train(D, c, e, seed=s)
                    runs.append(p)
                    print('  %s %s %-9s seed %d  anchor-holdout RMSE %.3f  (%.0f s)' % (city, task, name, s, v,
                                                                                     time.time() - t), flush=True)
                P[name] = {u: np.mean([r[u] for r in runs], axis=0) for u in D['test']}
            S = score(D, P)
            report(D, S, base=next(iter(P)))
            allrows.append(S.assign(city=city, task=task))
    pd.concat(allrows).to_csv(os.path.join(OUT, 'net_%s_%s_%s.csv' % (tag, '_'.join(cities), '_'.join(tasks))),
                              index=False)


if __name__ == '__main__':
    main()
