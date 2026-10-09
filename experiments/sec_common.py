"""Shared loading for the SensEURCity experiments (hourly, flag-screened)."""
import glob
import os

import numpy as np
import pandas as pd

DATA = os.environ.get('SEC_DATA', 'C:/Users/luckf/dq_data/senseurcity')
UNITS = os.path.join(DATA, 'units')
CACHE = os.path.join(DATA, 'hourly')
META = os.path.join(DATA, 'peek', 'metadata', 'metadata_dates.csv')

GAS = ['NO2_B43F_P1', 'NO_B4_P1', 'OX_A431_P1', 'CO_A4_P1', 'D300']
MET = ['SHT31TE', 'SHT31HE', 'SHT31TI', 'SHT31HI', 'BMP280']
PMS = ['5301CAT', '5325CAT', '5310CAT', '53PT003', '53PT005', '53PT010', '53PT025', '53PT050', '53PT100']
OPC = ['OPCN3PM1', 'OPCN3PM25', 'OPCN3PM10']
REF = ['Ref.NO2', 'Ref.NO', 'Ref.O3', 'Ref.CO_ppm', 'Ref.PM2.5', 'Ref.PM10', 'Ref.PM1']
FLAGGED = GAS + ['5301CAT', '5325CAT', '5310CAT'] + OPC + ['BMP280', 'SHT31HE', 'SHT31HI', 'SHT31TE', 'SHT31TI']
# target -> (reference column, input channels)
TASKS = {
    'NO2': ('Ref.NO2', GAS + MET),
    'NO': ('Ref.NO', GAS + MET),
    'O3': ('Ref.O3', GAS + MET),
    'PM2.5': ('Ref.PM2.5', PMS + OPC + MET),
    'PM10': ('Ref.PM10', PMS + OPC + MET),
}
MIN_PER_HOUR = 45


def meta():
    m = pd.read_csv(META, encoding='latin-1')
    for c in m.columns:
        if c.endswith('_Start') or c.endswith('_End'):
            m[c] = pd.to_datetime(m[c], utc=True, errors='coerce').dt.tz_localize(None)
    return m.set_index('ASEs')


def unit_names(city=None):
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(UNITS, '*.parquet')))
    names = [n for n in names if 'REF' not in n]
    return [n for n in names if city is None or n.startswith(city)]


def hourly(name):
    """Hourly means of one unit. A channel reading is dropped when its quality flag is set;
    an hour needs at least MIN_PER_HOUR valid minutes of a channel to count."""
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, name + '.parquet')
    if os.path.exists(dst):
        return pd.read_parquet(dst)
    cols = ['date', 'Location.ID'] + GAS + MET + PMS + OPC + REF
    d = pd.read_parquet(os.path.join(UNITS, name + '.parquet'))
    for c in FLAGGED:
        f = c + '_flag'
        if f in d.columns:
            d.loc[d[f].notna(), c] = np.nan
    d = d[[c for c in cols if c in d.columns]].copy()
    d['hour'] = d['date'].dt.floor('h')
    num = [c for c in d.columns if c not in ('date', 'Location.ID', 'hour')]
    g = d.groupby('hour')
    h = g[num].mean()
    n = g[num].count()
    h = h.where(n >= MIN_PER_HOUR)
    h['site'] = g['Location.ID'].agg(lambda s: s.iloc[0]).astype(str)
    h = h.reset_index()
    h.to_parquet(dst, index=False)
    return h


def periods(name, m=None):
    """(first co-location, deployment, second co-location) time bounds of a unit."""
    m = meta() if m is None else m
    r = m.loc[name.split('_')[1]]
    return ((r['First_Col_Start'], r['First_Col_End'], r['First_Col_LocationID']),
            (r['Deployment_Start'], r['Deployment_End'], r['Deployment_LocationID']),
            (r['Second_Col_Start'], r['Second_Col_End'], r['Second_Col_LocationID']))
