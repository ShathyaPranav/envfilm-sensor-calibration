"""The TESLA benchmark (Ahn et al., AAAI 2025) rebuilt from SensEURCity v03.

Protocol, read from the official code (github.com/circleAhn/TESLA-tf, main.py,
lbt/utils/data_util.py, lbt/datasets/senseurcity/metadata.py):
  * first co-location of each unit only; rows with a missing value or a quality flag on
    the six PM columns dropped ("alldrop");
  * input: ONE Plantower channel (5310CAT, 5325CAT or 5301CAT for PM10, PM2.5, PM1) over
    the previous 360 rows; target: the matching Alphasense OPC-N3 value (OPCN3PM10/25/1)
    at the LAST row of the window; windows spanning gaps are still accepted (their
    accept_window_interval is effectively unlimited);
  * units in the order of their *_FIRST_COL_LIST: all but the last two train, the
    second-to-last validates, the last tests; RMSE in ug/m3.
Their preprocessed CSVs are on Google Drive; this file rebuilds them from the raw data,
so small differences are expected and are checked against their published "Raw" row.
"""
import os

import numpy as np
import pandas as pd

DATA = os.environ.get('SEC_DATA', 'C:/Users/luckf/dq_data/senseurcity')
UNITS = os.path.join(DATA, 'units')
OUT = os.path.join(DATA, 'tesla')
META = os.path.join(DATA, 'peek', 'metadata', 'metadata_dates.csv')

LISTS = {
    'Antwerp': ['402B00', '4043AE', '4043B1', '4047CD', '4047DD', '4047E0', '4047E7', '4049A6', '4065D0',
                '4065D3', '4065DA', '4065DD', '4065E0', '4065E3', '4065EA', '4067B0', '4067B3', '4067BA', '4067BD',
                '40623F', '40641B', '40642B', '402723', '406246', '406249', '406424', '408165', '408168', '408175',
                '408178'],
    'Oslo': ['64A291', '64CB6D', '64CB70', '64CB78', '64E9C5', '425FB3', '647D5A', '648B91', '651EF5', '651EFC',
             '652A32', '652D3A', '652FA4', '652FAF', '6517DD', '40817F', '40642E', '426178', '649312', '649526',
             '653257'],
    'Zagreb': ['64C52B', '64C225', '652A38', '652FA1', '4047D0', '40641E', '42816D', '64876B', '64876C', '427906',
               '428164', '648157', '648169', '649738'],
}
PMS = {'PM10': '5310CAT', 'PM2.5': '5325CAT', 'PM1': '5301CAT'}
OPC = {'PM10': 'OPCN3PM10', 'PM2.5': 'OPCN3PM25', 'PM1': 'OPCN3PM1'}
# published TESLA Table 1 (N = 360), RMSE
PUBLISHED = {
    'Raw': {'PM10': (23.05, 23.64, 15.41), 'PM2.5': (16.41, 8.46, 5.98), 'PM1': (13.21, 6.48, 5.58)},
    'Linear': {'PM10': (18.40, 20.51, 14.71), 'PM2.5': (12.33, 9.38, 3.34), 'PM1': (6.27, 5.19, 1.45)},
    'DLinear': {'PM10': (18.35, 20.45, 14.73), 'PM2.5': (12.22, 9.62, 3.35), 'PM1': (6.26, 5.21, 1.49)},
    'Transformer': {'PM10': (15.34, 17.49, 15.11), 'PM2.5': (8.49, 8.30, 3.40), 'PM1': (3.62, 5.22, 1.43)},
    'Informer': {'PM10': (15.47, 16.36, 14.86), 'PM2.5': (7.52, 8.14, 3.26), 'PM1': (4.43, 4.69, 1.34)},
    'PatchTST': {'PM10': (14.60, 16.90, 14.87), 'PM2.5': (7.66, 8.29, 3.33), 'PM1': (6.14, 4.86, 2.57)},
    'iTransformer': {'PM10': (16.54, 16.40, 14.03), 'PM2.5': (8.62, 7.88, 3.23), 'PM1': (5.10, 4.22, 1.49)},
    'TESLA': {'PM10': (14.58, 15.79, 13.87), 'PM2.5': (7.54, 7.90, 3.03), 'PM1': (4.78, 3.63, 1.14)},
}
CITIES = ('Antwerp', 'Oslo', 'Zagreb')
# dataset time span per city from TESLA Table A1 (applied on top of each unit's own window)
SPAN = {'Antwerp': ('2020-04-03 16:12', '2020-06-15 07:07'), 'Oslo': ('2020-09-01 00:00', '2020-10-14 08:56'),
        'Zagreb': ('2020-05-22 16:06', '2020-07-08 12:00')}
# station filter; Oslo is left open because the published row count (810,314) matches only
# without it
COLOC = {'Antwerp': 'ANT_REF_R801', 'Oslo': None, 'Zagreb': 'ZAG_REF_IMI'}
COLS = list(PMS.values()) + list(OPC.values())


def first_col_window(m, ase):
    r = m.loc[ase]
    a = pd.Timestamp(str(r['First_Col_Start'])[:16].replace('T', ' '))
    b = pd.Timestamp(str(r['First_Col_End'])[:16].replace('T', ' '))
    return a, b


def unit_frame(city, ase, m):
    dst = os.path.join(OUT, '%s_%s.parquet' % (city, ase))
    if os.path.exists(dst):
        return pd.read_parquet(dst)
    d = pd.read_parquet(os.path.join(UNITS, '%s_%s.parquet' % (city, ase)),
                        columns=['date', 'Location.ID'] + COLS + [c + '_flag' for c in COLS])
    a, b = pd.Timestamp(SPAN[city][0]), pd.Timestamp(SPAN[city][1])
    d = d[(d.date >= a) & (d.date <= b)]
    if COLOC[city]:
        d = d[d['Location.ID'].astype(str) == COLOC[city]]
    ok = d[COLS].notna().all(axis=1) & d[[c + '_flag' for c in COLS]].isna().all(axis=1)
    d = d.loc[ok, ['date'] + COLS].reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(dst, index=False)
    return d


ORIG = os.path.join(os.path.dirname(DATA), 'tesla_orig', 'finedust_concentration_alldrop_firstcol_dataset')
SOURCE = os.environ.get('TSL_SOURCE', 'orig')        # 'orig' = the authors' own preprocessed CSVs


def orig_frame(city, ase):
    pre = {'Antwerp': 'Antwerp_', 'Oslo': 'oslo_', 'Zagreb': 'Zagreb_'}[city]
    path = os.path.join(ORIG, '%s%s_FirstCol.csv' % (pre, ase))
    if not os.path.exists(path):
        path = os.path.join(ORIG, '%s_%s_FirstCol.csv' % (city, ase))
    d = pd.read_csv(path)
    d['date'] = pd.to_datetime(d['date'], utc=True).dt.tz_localize(None)
    return d


EXTRA = [c for c in os.environ.get('TSL_EXTRA', '').split(',') if c]   # extra on-board channels, e.g. SHT31HE


def with_extra(city, ase, d):
    """Attach on-board channels (from the unit's raw file, same minute) to the authors' rows."""
    if not EXTRA:
        return d
    u = pd.read_parquet(os.path.join(UNITS, '%s_%s.parquet' % (city, ase)), columns=['date'] + EXTRA)
    u = u.groupby('date').mean()
    d = d.join(u, on='date')
    d[EXTRA] = d[EXTRA].interpolate(limit_direction='both')
    return d


REF10 = os.path.join(DATA, 'ref10')


def ref10_frame(city, ase):
    """Second benchmark: same units and periods as the authors' first-co-location files, but the
    target is the station's REFERENCE instrument (Ref.PM10/2.5/1), and everything is averaged to
    10 minutes (the references are not all minute-resolved). Stored in the authors' column names
    (the OPCN3* columns hold the reference) so the rest of the code is unchanged."""
    dst = os.path.join(REF10, '%s_%s.parquet' % (city, ase))
    if os.path.exists(dst):
        return pd.read_parquet(dst)
    o = orig_frame(city, ase)
    lo, hi = o.date.min(), o.date.max()
    cols = ['date', 'Location.ID', '5310CAT', '5325CAT', '5301CAT', 'Ref.PM10', 'Ref.PM2.5', 'Ref.PM1',
            '5310CAT_flag', '5325CAT_flag', '5301CAT_flag'] + EXTRA
    d = pd.read_parquet(os.path.join(UNITS, '%s_%s.parquet' % (city, ase)), columns=cols)
    d = d[(d.date >= lo) & (d.date <= hi)]
    for c in ('5310CAT', '5325CAT', '5301CAT'):
        d.loc[d[c + '_flag'].notna(), c] = np.nan
    d['t10'] = d.date.dt.floor('10min')
    num = ['5310CAT', '5325CAT', '5301CAT', 'Ref.PM10', 'Ref.PM2.5', 'Ref.PM1'] + EXTRA
    g = d.groupby('t10')[num]
    a = g.mean().where(g.count() >= 5)
    a = a.dropna(subset=['5310CAT', '5325CAT', '5301CAT', 'Ref.PM10', 'Ref.PM2.5', 'Ref.PM1'])
    a = a.rename(columns={'Ref.PM10': 'OPCN3PM10', 'Ref.PM2.5': 'OPCN3PM25', 'Ref.PM1': 'OPCN3PM1'})
    a = a.reset_index().rename(columns={'t10': 'date'})
    if EXTRA:
        a[EXTRA] = a[EXTRA].interpolate(limit_direction='both')
    os.makedirs(REF10, exist_ok=True)
    a.to_parquet(dst, index=False)
    return a


def load(city):
    if SOURCE == 'ref10':
        return [(ase, ref10_frame(city, ase)) for ase in LISTS[city]]
    if SOURCE == 'orig':
        return [(ase, with_extra(city, ase, orig_frame(city, ase))) for ase in LISTS[city]]
    m = pd.read_csv(META, encoding='latin-1').drop_duplicates('ASEs').set_index('ASEs')
    return [(ase, unit_frame(city, ase, m)) for ase in LISTS[city]]


def main():
    print('raw RMSE (uncalibrated Plantower vs OPC-N3) on the test unit, ours vs published:')
    for i, city in enumerate(CITIES):
        U = load(city)
        ase, t = U[-1]
        n = sum(len(f) for _, f in U)
        msg = []
        for pm in PMS:
            r = float(np.sqrt(np.mean((t[PMS[pm]] - t[OPC[pm]]) ** 2)))
            msg.append('%s %.2f (pub %.2f)' % (pm, r, PUBLISHED['Raw'][pm][i]))
        print('  %-8s units %2d rows %8d  test %s rows %6d | %s' % (city, len(U), n, ase, len(t), '  '.join(msg)))


if __name__ == '__main__':
    main()
