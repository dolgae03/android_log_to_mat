
import src.android import android
import src.android as android
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.io as pio
from dash import Dash, html, dcc
import pickle

from typing import Dict, Tuple, Set, Any

import scipy.io
import numpy as np, os
from tqdm import tqdm
from pathlib import Path

import argparse



raw_data = android.AndroidRawGnss(input_path='/home/user/mskim/gnss/data/raw/13349_gnss_log_2025_02_18_19_38_07.txt',filter_measurements=False,verbose=True)

df = raw_data.pandas_df()
df['tow'] = df['gps_millis'].apply(lambda x: int(round(x / 1000)) % 604800)

df_gps = df[(df['gnss_id']=='gps')]
df_G3 = df_gps[(df_gps['sv_id']==12)].copy()
any(df_G3.duplicated(subset='unix_millis', keep=False)) # 중복 데이터가 있는지 확인: False가 나와야 함

df_G3['tdpr_m'] = df_G3['raw_pr_m'].diff() / df_G3['gps_millis'].apply(lambda x: x*1.0e-3 if x != np.nan else np.nan).diff()

fig = make_subplots(subplot_titles = ['PR', 'timediff_PR vs DR', 'TDPR - DR']
                 , rows=3,cols=1, shared_xaxes='columns').update_layout(height=700)

fig.add_trace(
    go.Scatter(x=df_G3['tow'], y=df_G3['raw_pr_m'], name='PR(raw_pr_m)[m]'),
     row=1,col=1
)

print(df_G3['raw_pr_m'])

fig.update_yaxes(row=1, col=1, title_text='[meter]')

fig.add_trace(
    go.Scatter(x=df_G3['tow'], y=df_G3['PseudorangeRateMetersPerSecond'], name='DR(PseudorangeRateMetersPerSecond)[m/s]'),
    row=2,col=1
)

fig.add_trace(
  go.Scatter(x=df_G3['tow'], y=df_G3['tdpr_m'], name='TDPR[m/s]'),
    row=2,col=1
)

fig.update_yaxes(row=2, col=1, title_text='[meter/sec]')

# mean_diff = (df_G3['PseudorangeRateMetersPerSecond'] - df_G3['tdpr_m']).mean()
mean_diff = 0

fig.add_trace(
    go.Scatter(x=df_G3['tow'], y=df_G3['PseudorangeRateMetersPerSecond'] - df_G3['tdpr_m'] - mean_diff, name='TDPR - DR[m/s]'),
    row=3,col=1
)

fig.update_yaxes(row=3, col=1, title_text='[meter/sec]')
fig.update_xaxes(row=3, col=1, title_text='TOW[sec]')
fig.update_layout(title_text='GPS 3 SV Time Differenced Psedorange and Doppler'
                  , width=500, legend_x=0, legend_y=-0.5)
pio.write_image(fig, "./fig/fig.png")  # PNG 파일로 저장


df_G3['PR(raw_pr_m)'] = df_G3['raw_pr_m']
df_G3['CP(accumulated_delta_range_m)'] = df_G3['accumulated_delta_range_m']

mean_diff = (df_G3['raw_pr_m'] - df_G3['accumulated_delta_range_m']).mean()
df_G3['PR - CP'] = df_G3['PR(raw_pr_m)'] - df_G3['CP(accumulated_delta_range_m)'] - mean_diff

print((df_G3['PR(raw_pr_m)'] - df_G3['CP(accumulated_delta_range_m)'] - mean_diff).abs().mean())


fig = make_subplots(subplot_titles = ['PR', 'timediff_PR vs DR', 'TDPR - DR']
                 , rows=2,cols=1, shared_xaxes='columns').update_layout(height=900)

fig = px.scatter(df_G3, x='tow', y=['PR - CP'], range_x=[362000,363000])


fig.update_layout(title_text='GPS 3 SV Pseudorange and Carrier Phase', yaxis_title_text='Measurement Value [meter]', xaxis_title_text='TOW [sec]', width=500,
                  legend_x=0, legend_y=-0.5)

pio.write_image(fig, "./fig/fig2.png")  # PNG 파일로 저장


df_G3['adr_diff'] = df_G3['accumulated_delta_range_m'].diff().diff()

# 차분된 값을 플롯
fig = px.scatter(
    df_G3,
    x='tow',
    y='adr_diff',
    range_y=[-100, 100],
    title='ADR Difference Over Time (GPS 3 SV)',
    labels={'adr_diff': 'Δ ADR [m]', 'tow': 'TOW [sec]'}
)

fig.update_layout(
    width=600,
    height=400,
    yaxis_title='Δ ADR [m]',
    xaxis_title='TOW [sec]',
    legend=dict(x=0, y=-0.5)
)

# 이미지 저장
pio.write_image(fig, "./fig/fig3_adr_diff.png")