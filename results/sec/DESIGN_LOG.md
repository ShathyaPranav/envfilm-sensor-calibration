# SensEURCity model: design log

Every variant tried is listed here, with the number used to choose it. Choices are made on
the anchor hold-out (last 30 days of each anchor's deployment), never on the test units'
second co-location. Test numbers are recorded for each variant so nothing is hidden, but
they do not drive changes.

| date | variant | change | anchor hold-out RMSE (NO / NO2 / PM2.5) | test median RMSE (NO / NO2 / PM2.5) | decision |
|---|---|---|---|---|---|
| 2026-10-03 | v1 | first version as in sec_net.py docstring (CFG hidden 128, ctx 32, emb 8, lr 2e-3, 3 seeds averaged) | 2.7-4.0 / 2.3-2.4 / 1.5-1.6 (seed spread large on NO) | 6.89 / 3.91 / not finished | NO loses to simple pooled+ctx (6.04); NO2 beats every simple arm; training noisy, run interrupted by user pause |
| 2026-10-03 | v1 PM2.5 | v1 on PM2.5 | 1.52-1.65 | - / - / 3.23 (own-first 3.00) | loses on PM2.5 |
| 2026-10-03 | v2 | lr 5e-4, cosine, 40 epochs, patience 12 | 3.77-4.63 / 2.45-2.68 / 1.72-1.80 (all worse than v1) | 7.05 / 4.47 / 3.02 | REJECTED on hold-out |
| 2026-10-03 | v3 (planned) | residual: net learns a correction on top of each unit's own first-co-location MLP (out-of-fold on the first co-location), base value also given as an input channel; otherwise v1 | | | motivated by v1 losing to own-first on PM2.5 |
| 2026-10-03 | v3 | residual on own-first MLP (see above) | 1.83-2.06 / 2.81-3.03 / 1.40-1.44 (NO, PM2.5 better than v1; NO2 worse) | 5.86 / 4.22 / 2.71 | vs best simple arm per unit: NO -8.7% (8/21, fail), NO2 +19.6% (17/21, PASS), PM2.5 +7.6% (13/22, fails the 2/3 unit count). Pre-set bar (2 pollutants) NOT met in Antwerp. No-context ablation: NO 5.46, NO2 4.87, PM2.5 2.89 |
| 2026-10-03 | v1 NO2 rerun | same code as v1 | - | NO2 4.23 (first run gave 3.91: run-to-run noise ~0.3) | earlier 3.91 was partly luck |
| 2026-10-03 | v3 replication, code unchanged | Oslo, Zagreb | Oslo hold-out ranks variants opposite to the test on NO2 | Oslo NO 2.99 / NO2 4.04 / PM2.5 3.31 vs best simple 2.43 / 3.22 / 2.59 (LOSES all three); Zagreb NO2 4.96 vs 5.27 (+6%, 8 units), PM2.5 5.60 vs 7.57 (+26%, 8 units); Zagreb NO has no anchors | replication FAILS in Oslo; holds in Zagreb |
| 2026-10-03 | joint v3 (planned) | v3 unchanged, trained once on all three cities' first co-locations + anchors; plus city-blind setting. Pass mark in sec_joint.py docstring, fixed before the run | | | user chose option 1 (model paper only) |
| 2026-10-03 | joint v3 result (NO, NO2) | as planned | NO 4.45-4.81 | vs pre-set comparator: Antwerp NO -29% (6/21), Antwerp NO2 +18.9% (20/21, pass), Oslo NO +10.2% median gain but 13/21 units, Oslo NO2 +0.7%, Zagreb NO -11.8%, Zagreb NO2 -1.5% | PASS MARK FAILS (two losses > 5%); PM2.5 and O3 still running for completeness. Context now helps the net in all three cities on NO |
