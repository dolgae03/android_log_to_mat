
import gnss_lib_py as glp
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


def init_matlab_data_format(base_time: int, total_length: int, total_sat_num: int) -> Dict[str, Any]:
    data_dict = {
        'time_zero' : base_time,
        'time_gps' : np.array([[idx] for idx in range(1, total_length + 1)]),
        'time' : np.full((total_length, 1), 0.0),

        'pr1' : np.full((total_length, total_sat_num), np.nan),
        'ph1' : np.full((total_length, total_sat_num), np.nan),
        'dop1' : np.full((total_length, total_sat_num), np.nan),
        'snr1' : np.full((total_length, total_sat_num), np.nan),
        'loi1' : np.full((total_length, total_sat_num), np.nan),
        'clock_drift' : np.full((total_length, total_sat_num), np.nan),
        'clock_drift_uncertainty' : np.full((total_length, total_sat_num), np.nan),

        'XS_tot1' : np.full((total_length, total_sat_num, 3), np.nan),
        'VS_tot1' : np.full((total_length, total_sat_num, 3), np.nan),

        'activate_constellation' : np.array([True, False, True, False, True, False, False]),
        'constellation_name': np.array(['GPS', 'GLO', 'GAL', 'QZS', 'BDS', 'IRN', 'SBS'], dtype=object),
        'constellation_idx' : np.array([1,33,60,96,103,183,183])
    }
    
    return data_dict


def main(target_file_name: str, calculate_sv_position = True):
    BASE_DIR = Path('./data/raw')
    target_file_dir = BASE_DIR / target_file_name

    target_data = {}

    raw_data = glp.AndroidRawGnss(input_path=target_file_dir,filter_measurements=False,verbose=False)
    # raw_data = raw_data[raw_data['signal_type'].isin(['l1', 'e1', 'b1'])].copy()
    # full_states = glp.add_sv_states_rinex(raw_data, constellations = ['gps', 'beidou', 'galileo'])
    # full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states['b_sv_m']

    df = raw_data.pandas_df()

    '''
    ['# Raw', 'TimeNanos', 'LeapSecond', 'TimeUncertaintyNanos', 'FullBiasNanos', 
    'BiasNanos', 'BiasUncertaintyNanos', 'DriftNanosPerSecond', 'DriftUncertaintyNanosPerSecond', 
    'HardwareClockDiscontinuityCount', 'TimeOffsetNanos', 'State', 'ReceivedSvTimeNanos', 'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateMetersPerSecond', 'PseudorangeRateUncertaintyMetersPerSecond', 'AccumulatedDeltaRangeState', 'CarrierFrequencyHz', 
    'CarrierCycles', 'CarrierPhase', 'CarrierPhaseUncertainty', 'MultipathIndicator', 'SnrInDb', 'AgcDb', 'BasebandCn0DbHz', 
    'FullInterSignalBiasNanos', 'FullInterSignalBiasUncertaintyNanos', 'SatelliteInterSignalBiasNanos', 'SatelliteInterSignalBiasUncertaintyNanos', 
    'CodeType', 'ChipsetElapsedRealtimeNanos', 'IsFullTracking', 'unix_millis', 'sv_id', 'cn0_dbhz', 'accumulated_delta_range_m', 
    'accumulated_delta_range_sigma_m', 'gnss_id', 'signal_type', 'gps_millis', 'raw_pr_m', 'raw_pr_sigma_m']
    '''

    IDX_MAPPING = {
        'gps' : 1,
        'galileo' : 60,
        'beidou' : 103,
        'glonass' : 33,
        'qzss' : 96
    }


    df['gps_time'] = df['gps_millis'].apply(lambda x: int(round(x / 1000)))
    df['gnss_idx'] = df.apply(lambda row: (IDX_MAPPING.get(row['gnss_id'], 0) - 1) + (row['sv_id'] - 1),
                            axis=1)
    df['loi'] = df.apply(lambda row: row['AccumulatedDeltaRangeState'] & 1, axis=1)

    START_GPS_TIME = df['gps_time'].min()
    END_GPS_TIME = df['gps_time'].max()
    INTERVAL = END_GPS_TIME - (START_GPS_TIME + 1) + 1
    TOTAL_SAT_IDX = 182

    CONSTELLATION = {'C' : 46,
                    'G' : 32,
                    'E' : 60,}

    all_sats = [f"{each_code}{sv:02d}" for each_code, max_num in CONSTELLATION.items() for sv in range(1, max_num + 1)]

    result_dict = init_matlab_data_format(START_GPS_TIME, INTERVAL, TOTAL_SAT_IDX)
    ephem = None

    full_idx = np.arange(TOTAL_SAT_IDX)

    df_filtered = df[df['signal_type'].isin(['l1', 'e1', 'b1'])].copy()
    grouped_df = df_filtered.groupby('gps_time')

    for idx, each_gps_time in enumerate(tqdm(range(START_GPS_TIME + 1, END_GPS_TIME + 1), desc="Processing GPS Times")):
        # gps_time 및 signal_type 조건으로 필터링 (query 사용)
        # gps_time에 해당하는 그룹을 빠르게 조회합니다.
        try:
            current_df = grouped_df.get_group(each_gps_time).copy()
        except KeyError:
            # 해당 gps_time에 데이터가 없는 경우 건너뜁니다.
            continue

        # 시간 계산 (예: gps_millis 첫 번째 값 사용)
        result_dict['time'][idx] = current_df['gps_millis'].iloc[0] / 1000.0 - START_GPS_TIME

        # vectorized 방식으로 gnss_idx 계산
        # IDX_MAPPING이 없는 경우를 위해 fillna로 기본값 설정 (예: 1) 후 계산
        current_df['gnss_idx'] = (current_df['gnss_id'].map(IDX_MAPPING).fillna(1).astype(int) - 1) + (current_df['sv_id'] - 1)

        current_df = current_df.drop_duplicates(subset=['gnss_idx'], keep='first')

        # 전체 위성 인덱스(full_idx)에 맞춰 정렬 (merge보다 빠름)
        current_df = current_df.set_index('gnss_idx').reindex(full_idx).reset_index()

        # 필요한 컬럼을 numpy 배열로 추출하여 result_dict에 저장 (flatten 불필요)
        result_dict['pr1'][idx, :] = current_df['raw_pr_m'].to_numpy()
        result_dict['ph1'][idx, :] = current_df['accumulated_delta_range_m'].to_numpy()
        result_dict['dop1'][idx, :] = current_df['PseudorangeRateMetersPerSecond'].to_numpy()
        result_dict['snr1'][idx, :] = current_df['cn0_dbhz'].to_numpy()
        result_dict['loi1'][idx, :] = current_df['loi'].to_numpy()
        result_dict['clock_drift'][idx, :] = current_df['clock_drift'].to_numpy()
        result_dict['clock_drift_uncertainty'][idx, :] = current_df['clock_drift_uncertainty'].to_numpy()

        if calculate_sv_position:
            # gps_millis 계산을 한 번만 수행
            time_millis = each_gps_time * 1000
            dt = glp.gps_millis_to_datetime(time_millis)

            # ephemeris 갱신 조건 체크
            if ephem is None or (dt.minute == 0 and dt.second == 0):
                ephem = glp.get_time_cropped_rinex(time_millis, all_sats, ephemeris_directory="ephemeris")

            # SV 상태 계산
            sv_states_tx = glp.find_sv_states(time_millis, ephem).pandas_df()
            sv_states_tx['gnss_idx'] = (sv_states_tx['gnss_id'].map(IDX_MAPPING).fillna(1).astype(int) - 1) + (sv_states_tx['sv_id'] - 1)
            sv_states_tx = sv_states_tx.set_index('gnss_idx').reindex(full_idx).reset_index()

            # 위성 위치와 속도 배열 추출
            xyz_pos_array = sv_states_tx[['x_sv_m', 'y_sv_m', 'z_sv_m']].to_numpy()
            xyz_vel_array = sv_states_tx[['vx_sv_mps', 'vy_sv_mps', 'vz_sv_mps']].to_numpy()

        else:
            # `calculate_sv_position`이 False인 경우 NaN 배열로 초기화
            xyz_pos_array = np.full((len(full_idx), 3), np.nan)
            xyz_vel_array = np.full((len(full_idx), 3), np.nan)

        result_dict['XS_tot1'][idx, :, :] = xyz_pos_array
        result_dict['VS_tot1'][idx, :, :] = xyz_vel_array

    # 확장자 변경
    base_name, _ = os.path.splitext(target_file_name)
    new_target_file_name = base_name + '.mat'

    # result_dict를 저장하는 부분
    scipy.io.savemat(f'./data/mat/{new_target_file_name}', result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS Satellite Selection and Processing")
    parser.add_argument("--input_file_name","-n", type=str, default='gnss_log_2024_12_12_12_43_03.txt', help="Path to GNSS log file")
    parser.add_argument("--calc_sv", '-c', action="store_false")
    
    args = parser.parse_args()
    
    print(args.input_file_name, args.calc_sv)
    main(args.input_file_name, args.calc_sv)



# print(df.columns.tolist())


# df_gps = df[(df['gnss_id']=='gps')]
# df_G3 = df_gps[(df_gps['sv_id']==3)].copy()
# any(df_G3.duplicated(subset='unix_millis', keep=False)) # 중복 데이터가 있는지 확인: False가 나와야 함

# df_G3['tdpr_m'] = df_G3['raw_pr_m'].diff() / df_G3['gps_millis'].apply(lambda x: x*1.0e-3 if x != np.nan else np.nan).diff()

# fig = make_subplots(subplot_titles = ['PR', 'timediff_PR vs DR', 'TDPR - DR']
#                  , rows=3,cols=1, shared_xaxes='columns').update_layout(height=700)

# fig.add_trace(
#     go.Scatter(x=df_G3['tow'], y=df_G3['raw_pr_m'], name='PR(raw_pr_m)[m]'),
#      row=1,col=1
# )

# fig.update_yaxes(row=1, col=1, title_text='[meter]')

# fig.add_trace(
#     go.Scatter(x=df_G3['tow'], y=df_G3['PseudorangeRateMetersPerSecond'], name='DR(PseudorangeRateMetersPerSecond)[m/s]'),
#     row=2,col=1
# )

# fig.add_trace(
#   go.Scatter(x=df_G3['tow'], y=df_G3['tdpr_m'], name='TDPR[m/s]'),
#     row=2,col=1
# )

# fig.update_yaxes(row=2, col=1, title_text='[meter/sec]')

# mean_diff = (df_G3['PseudorangeRateMetersPerSecond'] - df_G3['tdpr_m']).mean()

# fig.add_trace(
#     go.Scatter(x=df_G3['tow'], y=df_G3['PseudorangeRateMetersPerSecond'] - df_G3['tdpr_m'] - mean_diff, name='TDPR - DR[m/s]'),
#     row=3,col=1
# )

# fig.update_yaxes(row=3, col=1, title_text='[meter/sec]')
# fig.update_xaxes(row=3, col=1, title_text='TOW[sec]')
# fig.update_layout(title_text='GPS 3 SV Time Differenced Psedorange and Doppler'
#                   , width=500, legend_x=0, legend_y=-0.5)
# pio.write_image(fig, "fig.png")  # PNG 파일로 저장


# df_G3['PR(raw_pr_m)'] = df_G3['raw_pr_m']
# df_G3['CP(accumulated_delta_range_m)'] = df_G3['accumulated_delta_range_m']

# mean_diff = (df_G3['raw_pr_m'] - df_G3['accumulated_delta_range_m']).mean()
# df_G3['PR - CP'] = df_G3['PR(raw_pr_m)'] - df_G3['CP(accumulated_delta_range_m)'] - mean_diff

# print((df_G3['PR(raw_pr_m)'] - df_G3['CP(accumulated_delta_range_m)'] - mean_diff).abs().mean())


# fig = make_subplots(subplot_titles = ['PR', 'timediff_PR vs DR', 'TDPR - DR']
#                  , rows=2,cols=1, shared_xaxes='columns').update_layout(height=900)

# fig = px.scatter(df_G3, x='tow', y=['PR - CP'], range_x=[362000,363000])


# fig.update_layout(title_text='GPS 3 SV Pseudorange and Carrier Phase', yaxis_title_text='Measurement Value [meter]', xaxis_title_text='TOW [sec]', width=500,
#                   legend_x=0, legend_y=-0.5)

# pio.write_image(fig, "fig2.png")  # PNG 파일로 저장


# fig = px.scatter(df_G3, x='tow', y=['HardwareClockDiscontinuityCount'], range_x=[362000,363000])


# fig.update_layout(title_text='GPS 3 SV Pseudorange and Carrier Phase', yaxis_title_text='Measurement Value [meter]', xaxis_title_text='TOW [sec]', width=500,
#                   legend_x=0, legend_y=-0.5)

# pio.write_image(fig, "fig3.png")  # PNG 파일로 저장