"""Matched control for soton_onboard.py (results/tesla/PREREG_southampton_matched.md): the same rows, period,
window, folds and seeds, but the four state channels are the site weather station's humidity, temperature, dew
point and pressure (the channels of the earlier weather-only run) instead of the monitors' on-board sensors."""
import os
import sys

import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXP)
os.environ['TSL_EXTRA'] = 'rh,temperature,dew_point_temperature,air_pressure'
import tsl_bench as B
import tsl_rotate as R

BASE = 'C:/Users/luckf/dq_data'
GROUP = {'SotonPMS': 'PMS5003', 'SotonSPS': 'SPS030'}


def frames(group):
    cache = os.path.join(BASE, 'southampton_onboard', 'frames_%s.parquet' % group)   # written by soton_onboard.py
    d = pd.read_parquet(cache).drop(columns=['rh_onboard', 't_onboard', 'rh', 'temperature'])
    w = pd.read_csv(os.path.join(BASE, 'southampton', 'df_weather_2min.csv'), parse_dates=['datetime_cut'])
    w = w.drop(columns=['date']).rename(columns={'datetime_cut': 'date'}).drop_duplicates('date')
    d = d.merge(w[['date', 'rh', 'temperature', 'dew_point_temperature', 'air_pressure']], on='date', how='left')
    d = d.sort_values(['sensor', 'date'])
    d[['rh', 'temperature', 'dew_point_temperature', 'air_pressure']] = (
        d.groupby('sensor')[['rh', 'temperature', 'dew_point_temperature', 'air_pressure']].transform(
            lambda s: s.interpolate(limit_direction='both')))
    return [(s, g.drop(columns='sensor').reset_index(drop=True)) for s, g in d.groupby('sensor')]


B.load = frames
B.EXTRA_OVERRIDE = True
R.CITIES = tuple(GROUP)

if __name__ == '__main__':
    R.main()
