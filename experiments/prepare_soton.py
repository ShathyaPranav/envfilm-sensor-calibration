"""Prepare the two Southampton files the loaders read (soton_onboard.py, soton_matched.py).

Inputs, downloaded by hand into $DQ_DATA (default: <repo>/data):
  southampton/lcs.rds, df_pm_2min.csv, df_weather_2min.csv   Zenodo 7198378 (Bulot et al.)
  southampton_onboard/rh_temperature.csv                       Zenodo 6637781 (Bulot)

Outputs:
  southampton/lcs.parquet                  the low-cost sensor table, converted from R's .rds (needs pyreadr)
  southampton_onboard/onboard_2min.parquet each monitor's SHT35 humidity and temperature, 10 s readings averaged
                                           to the 2-minute grid of the other files
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get('DQ_DATA', os.path.join(ROOT, 'data'))


def lcs():
    dst = os.path.join(BASE, 'southampton', 'lcs.parquet')
    if os.path.exists(dst):
        return dst
    import pyreadr
    d = next(iter(pyreadr.read_r(os.path.join(BASE, 'southampton', 'lcs.rds')).values()))
    d.to_parquet(dst, index=False)
    return dst


def onboard():
    dst = os.path.join(BASE, 'southampton_onboard', 'onboard_2min.parquet')
    if os.path.exists(dst):
        return dst
    r = pd.read_csv(os.path.join(BASE, 'southampton_onboard', 'rh_temperature.csv'),
                    usecols=['median_humidity', 'median_temperature', 'site', 'date'], parse_dates=['date'])
    r['t2'] = r['date'].dt.floor('2min')
    d = r.groupby(['site', 't2'])[['median_humidity', 'median_temperature']].mean().reset_index()
    d.to_parquet(dst, index=False)
    return dst


if __name__ == '__main__':
    print(lcs())
    print(onboard())
