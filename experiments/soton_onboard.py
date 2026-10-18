"""Unit rotation on Southampton with each monitor's on-board humidity and temperature
(results/tesla/PREREG_southampton_onboard.md). Wraps tsl_rotate with a Southampton loader."""
import os
import sys

import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXP)
os.environ['TSL_EXTRA'] = 'rh_onboard,t_onboard,rh,temperature'
import tsl_bench as B
import tsl_rotate as R

BASE = os.environ.get('DQ_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
GROUP = {'SotonPMS': 'PMS5003', 'SotonSPS': 'SPS030'}


def frames(group):
    cache = os.path.join(BASE, 'southampton_onboard', 'frames_%s.parquet' % group)
    if not os.path.exists(cache):
        d = pd.read_parquet(os.path.join(BASE, 'southampton', 'lcs.parquet'),
                            columns=['sensor', 'site', 'cutDate', 'median_PM10', 'median_PM25', 'median_PM1'])
        d = d[d.sensor.str.startswith(GROUP[group])].copy()
        d['date'] = pd.to_datetime(d.cutDate.astype(str))
        d['site'] = d.site.str.lower()
        p = pd.read_csv(os.path.join(BASE, 'southampton', 'df_pm_2min.csv'), parse_dates=['datetime_cut']).drop(columns=['date'])
        w = pd.read_csv(os.path.join(BASE, 'southampton', 'df_weather_2min.csv'), parse_dates=['datetime_cut']).drop(columns=['date'])
        o = pd.read_parquet(os.path.join(BASE, 'southampton_onboard', 'onboard_2min.parquet'))
        o['site'] = o.site.str.lower()
        o = o.rename(columns={'t2': 'date', 'median_humidity': 'rh_onboard', 'median_temperature': 't_onboard'})
        p = p.rename(columns={'datetime_cut': 'date', 'PM10': 'OPCN3PM10', 'PM2.5': 'OPCN3PM25', 'PM1': 'OPCN3PM1'})
        w = w.rename(columns={'datetime_cut': 'date'})
        d = d.rename(columns={'median_PM10': '5310CAT', 'median_PM25': '5325CAT', 'median_PM1': '5301CAT'})
        d = d.merge(o, on=['site', 'date']).merge(p[['date', 'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1']], on='date').merge(
            w[['date', 'rh', 'temperature']], on='date')
        d = d[d.date < pd.Timestamp('2020-12-31')].dropna().sort_values(['sensor', 'date'])
        d[['sensor', 'date', '5310CAT', '5325CAT', '5301CAT', 'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1', 'rh_onboard',
           't_onboard', 'rh', 'temperature']].to_parquet(cache, index=False)
    d = pd.read_parquet(cache)
    return [(s, g.drop(columns='sensor').reset_index(drop=True)) for s, g in d.groupby('sensor')]


B.load = frames
B.EXTRA_OVERRIDE = True
R.CITIES = tuple(GROUP)

if __name__ == '__main__':
    R.main()
