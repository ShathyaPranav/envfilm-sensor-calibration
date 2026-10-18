"""Smoke test for intel_trustnet.py on SYNTHETIC sensors, before the expensive real run.

Synthetic sensors mimic what Intel Lab showed: slowly varying quantized
temperature, battery voltage declining over the deployment, saturation at
122.153 C that becomes likelier below a SENSOR-SPECIFIC onset voltage, and a few
plausible-but-wrong readings in the transition band. Onsets differ per sensor so
the per-sensor heterogeneity measured on the real data is present.

This checks MECHANICS only -- that nothing crashes, that the telemetry prior stays
monotone through training, and that the model can learn the intended behaviour at
all. It does not test the non-circularity argument: synthetic fault labels here
are generated from voltage by construction, unlike the real evaluation's
voltage-independent labels.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intel_trustnet as T


def synth(seed, onset):
    rng = np.random.default_rng(seed)
    n = 9000
    t = np.arange(n)
    temp = 22 + 1.5 * np.sin(2 * np.pi * t / 1440) + rng.normal(0, 0.06, n)
    temp = np.round(temp / 0.0098) * 0.0098
    v = np.linspace(2.66, 2.12, n) + rng.normal(0, 0.004, n)
    sat = rng.random(n) < 1 / (1 + np.exp((v - onset) / 0.015))
    wrong = (v > onset) & (v < onset + 0.08) & (rng.random(n) < 0.03)
    x = temp.copy()
    x[wrong] = rng.uniform(1.0, 4.0, wrong.sum())
    x[sat] = 122.153
    return x, v, sat | wrong | (x < 0) | (x > 50)


def build(seed, onset):
    x, v, lab = synth(seed, onset)
    feat, scale = T.causal_features(x, v)
    refs = T.window_refs(x, lab, v)
    return dict(x=x, v=v, lab=lab, feat=feat, scale=scale, refs=refs,
                xt=torch.tensor(x, dtype=torch.float32), ft=torch.tensor(feat),
                rmu=torch.tensor(np.nan_to_num(refs['ref_mu']), dtype=torch.float32),
                rsd=torch.tensor(np.nan_to_num(refs['ref_sd']), dtype=torch.float32),
                valid=torch.tensor(refs['valid']), fault=torch.tensor(refs['fault']))


checks = []


def check(name, cond, detail=''):
    checks.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


print("1. feature and reference construction")
onsets = [2.22, 2.26, 2.30, 2.33, 2.28, 2.24, 2.31]
data = {i: build(i, o) for i, o in enumerate(onsets)}
d0 = data[0]
check("features finite", np.isfinite(d0['feat']).all(), f"shape {d0['feat'].shape}")
check("references exist", d0['refs']['valid'].sum() > 200, f"valid windows {d0['refs']['valid'].sum()}")
check("threshold baseline never NaN (fallback works)",
      np.isfinite(d0['refs']['thr_sd']).all() and np.isfinite(d0['refs']['thr_mu']).all(),
      f"fallback used in {d0['refs']['thr_empty'].sum()} windows")
check("fault labels present", d0['lab'].mean() > 0.05, f"fault share {d0['lab'].mean():.3f}")

print("2. telemetry prior is monotone at initialization")
m0 = T.TrustNet(True, True, True)
with torch.no_grad():
    for name, mod, lo, hi in (('g_v', m0.g_v, 2.0, 2.8), ('g_vrel', m0.g_vrel, 0.6, 1.2)):
        g = mod(torch.linspace(lo, hi, 400)).numpy()
        check(f"{name} non-decreasing", np.all(np.diff(g) >= -1e-6))

print("3. training runs and learns (train on 6 sensors, test on 1 held out)")
T.EPOCHS = 25
tr, te = [0, 1, 2, 3, 4, 5], [6]
model = T.train(tr, data, 'full_mono')
with torch.no_grad():
    for name, mod, lo, hi in (('g_v', model.g_v, 2.0, 2.8), ('g_vrel', model.g_vrel, 0.6, 1.2)):
        g = mod(torch.linspace(lo, hi, 400)).numpy()
        check(f"{name} still non-decreasing after training", np.all(np.diff(g) >= -1e-6))
res = T.evaluate(model, te, data, 'full_mono', 0)
f = res[res.fault]
check("held-out evaluation produced rows", len(res) > 0, f"{len(res)} windows, {len(f)} with faults")
if len(f):
    trust, std, madn = np.median(f.e_trust_sd), np.median(f.e_std), np.median(f.e_madn)
    check("TRUST-Net beats plain std on fault windows", trust < std,
          f"trust {trust:.4f}  std {std:.4f}  MADn {madn:.4f}")
if 'weight_auc' in res:
    check("weights separate clean from faulty readings", res['weight_auc'].iloc[0] > 0.9,
          f"AUC {res['weight_auc'].iloc[0]:.4f}")

print("4. every ablation config trains and evaluates")
T.EPOCHS = 3
for cfg in T.CFG:
    try:
        mm = T.train(tr, data, cfg)
        rr = T.evaluate(mm, te, data, cfg, 0)
        check(f"config {cfg}", len(rr) > 0 and np.isfinite(rr.e_trust_sd).all())
    except Exception as e:  # report, don't hide, a crashing config
        check(f"config {cfg}", False, repr(e))

n_fail = sum(not ok for _, ok in checks)
print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
sys.exit(1 if n_fail else 0)
