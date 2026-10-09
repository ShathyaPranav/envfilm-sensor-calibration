"""Apply the pre-set pass mark of sec_joint.py to its results."""
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'sec')


def main():
    j = pd.read_csv(os.path.join(R, 'joint_v3.csv'))
    single = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ('protocol_simple_Antwerp.csv',
                                                                 'protocol_simple_Oslo_Zagreb.csv')])
    single['arm'] = single['kind'] + ' single-city ' + single['arm']
    simple = pd.concat([single[['city', 'task', 'unit', 'arm', 'rmse']],
                        j[~j.arm.str.contains('net')][['city', 'task', 'unit', 'arm', 'rmse']]])
    simple = simple[~simple.arm.str.contains('oracle')]
    rows = []
    for (c, t), g in j[j.arm == 'joint net'].groupby(['city', 'task']):
        s = simple[(simple.city == c) & (simple.task == t)]
        if s.empty:
            continue
        w = s.pivot_table(index='unit', columns='arm', values='rmse')
        best = w.median().idxmin()
        net = g.set_index('unit').rmse.reindex(w.index)
        gain = 100 * (1 - net / w[best])
        blind = j[(j.city == c) & (j.task == t) & (j.arm == 'city-blind net')].set_index('unit').rmse.reindex(w.index)
        rows.append(dict(city=c, task=t, units=len(w), net=net.median(), comparator=best, comp=w[best].median(),
                         gain=gain.median(), wins=int((net < w[best]).sum()), blind=blind.median(),
                         passes=bool(gain.median() >= 5 and (net < w[best]).sum() >= 2 * len(w) / 3),
                         big_loss=bool(gain.median() < -5)))
    r = pd.DataFrame(rows)
    print(r.round(2).to_string(index=False))
    prim = r[r.task.isin(['NO', 'NO2', 'PM2.5'])]
    ok_cities = sum(int(g.passes.sum() >= 2) for _, g in prim.groupby('city'))
    print('\ncities with >= 2 primary pollutants passing: %d of 3; city-pollutant pairs losing by > 5%%: %d'
          % (ok_cities, int(prim.big_loss.sum())))
    print('PASS MARK:', 'PASS' if ok_cities >= 2 and not prim.big_loss.any() else 'FAIL')


if __name__ == '__main__':
    main()
