
import gnss_lib_py as glp
import src.android as android
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.io as pio
from dash import Dash, html, dcc
from src.data_class import Measurement, TimeTag, SatelliteInfo, MeasurementValue
import pickle

from typing import Dict, Tuple, Set, Any

import scipy.io
import numpy as np, os
from tqdm import tqdm
from pathlib import Path

import argparse, csv
import simplekml

from datetime import datetime, timedelta, timezone

import numpy as np
from gnss_lib_py.utils.coordinates import geodetic_to_ecef, ecef_to_geodetic


# GPS epoch
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


STATE_CODE_LOCK     = 1        # 0x00000001
STATE_BIT_SYNC      = 2
STATE_SUBFRAME_SYNC = 4
STATE_TOW_DECODED   = 8
STATE_MSEC_AMBIGUOUS  = 16   # <<< 이거 추가!
STATE_TOW_KNOWN     = 16384    # 0x00004000


def gps_sec_to_iso(gps_sec: int) -> str:
    """
    gps_sec: GPS epoch(1980-01-06) 기준 초라고 가정.
    (만약 ms면 gps_sec/1000.0 으로 바꿔서 써야 함)
    """
    t_utc = GPS_EPOCH + timedelta(seconds=gps_sec)
    return t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

'''
['# Raw', 'TimeNanos', 'LeapSecond', 'TimeUncertaintyNanos', 'FullBiasNanos', 
'BiasNanos', 'BiasUncertaintyNanos', 'DriftNanosPerSecond', 'DriftUncertaintyNanosPerSecond', 
'HardwareClockDiscontinuityCount', 'TimeOffsetNanos', 'State', 'ReceivedSvTimeNanos', 'ReceivedSvTimeUncertaintyNanos',
'PseudorangeRateMetersPerSecond', 'PseudorangeRateUncertaintyMetersPerSecond', 'AccumulatedDeltaRangeState', 'CarrierFrequencyHz', 
'CarrierCycles', 'CarrierPhase', 'CarrierPhaseUncertainty', 'MultipathIndicator', 'SnrInDb', 'AgcDb', 'BasebandCn0DbHz', 
'FullInterSignalBiasNanos', 'FullInterSignalBiasUncertaintyNanos', 'SatelliteInterSignalBiasNanos', 'SatelliteInterSignalBiasUncertaintyNanos', 
'CodeType', 'ChipsetElapsedRealtimeNanos', 'IsFullTracking', 'unix_millis', 'sv_id', 'cn0_dbhz', 'accumulated_delta_range_m', 
'accumulated_delta_range_sigma_m', 'gnss_id', 'signal_type', 'gps_millis', 'raw_pr_m', 'raw_pr_sigma_m', 'State']
'''

def init_matlab_data_format(base_time: int, total_length: int, total_sat_num: int) -> Dict[str, Any]:
    measurement = { 
        f'{meas}{frequency}' : np.full((total_length, total_sat_num), np.nan) 
        for meas in ['pr', 'ph', 'dop', 'snr', 'loi'] for frequency in ['1', '2', '3','1_l'] 
    }


    data_dict = {
        **measurement,
        'time_zero' : base_time,
        'time_GPS' : np.array([[idx] for idx in range(1, total_length + 1)]),
        'time' : np.full((total_length, 1), 0.0),
        
        'clock_drift' : np.full((total_length, total_sat_num), np.nan),
        'clock_drift_uncertainty' : np.full((total_length, total_sat_num), np.nan),

        'XS_tot1' : np.full((total_length, total_sat_num, 3), np.nan),
        'VS_tot1' : np.full((total_length, total_sat_num, 3), np.nan),

        'activate_constellation' : np.array([True, False, True, False, True, False, False]),
        'constellation_name': np.array(['GPS', 'GLO', 'GAL', 'QZS', 'BDS', 'IRN', 'SBS'], dtype=object),
        'constellation_idx' : np.array([1,33,60,96,103,183,183])
    }
    
    return data_dict

import matplotlib.pyplot as plt

BASE_DIR = Path('./data/raw')

def _check_each_clock_value(
    df: pd.DataFrame,
    target_constellation: str = "beidou",
    target_freq: int = 1_575_420_030,      # GPS L1 C/A
    x_ratio_start: float = 0.0,
    x_ratio_end: float = 1.0,
    save_dir: str = "./fig/sample",
) -> None:
    """
    하나의 위성-주파수 트랙에서 시계(Clock) 값과 의사거리(raw_pr_m)를 scatter 플롯으로 확인·저장한다.
    모든 결과 PNG는 `{save_dir}/{target_constellation}_{target_svid}/`에 저장된다.
    """

    for target_svid in range(100):
        sat_df = df[
            (df["sv_id"] == target_svid)
            & (df["CarrierFrequencyHz"] == target_freq)
            & (df["gnss_id"] == target_constellation)
        ]
        if sat_df.empty:
            # raise ValueError(
            #     f"Svid {target_svid} @ {target_freq} Hz 데이터가 없습니다."
            # )
            print(f"Svid {target_svid} @ {target_freq} Hz 데이터가 없습니다.")
            continue

        # ── 2) x-구간(비율) 잘라서 사용 ─────────────────────────────────────
        gps = sat_df["gps_millis"]
        x_min, x_max = gps.min(), gps.max()
        xlim_min = x_min + (x_max - x_min) * x_ratio_start
        xlim_max = x_min + (x_max - x_min) * x_ratio_end
        plot_df = sat_df[(gps >= xlim_min) & (gps <= xlim_max)]

        # ── 3) 저장 폴더 준비  (예: ./fig/sample/gps_14/) ────────────────────
        out_dir = Path(save_dir) / f"{x_ratio_start}_{x_ratio_end}" / f"{target_constellation}" / f"{target_freq}" / f"{target_svid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 4) Clock 필드 scatter 플롯 ───────────────────────────────────────
        target_fields = {
            'clock' : [
                ("FullBiasNanos",                     "full_bias_nanos"),
                ("BiasNanos",                         "bias_nanos"),
                ("TimeNanos",                         "time_nanos"),
                ("ReceivedSvTimeNanos",               "received_sv_time_nanos"),
                ("TimeOffsetNanos",                   "timeoffset_nanos"),
                ("FB-gps_week",                       "diff"),
                ("Final_Plus(Timeoffsetnanos - biasnanos)", "new"),
            ],
            'measurement' : [
                ("raw_pr_m",                      "raw_pr_m(m)"),
                ("AccumulatedDeltaRangeState",    'Carrier State'),
                ("accumulated_delta_range_m",     'Carrier(m)'),
                ("PseudorangeRateMetersPerSecond",'Doppler(m_s)')
            ]
        }

        for data_cateogry, each_fields in target_fields.items():
            for field, fname in each_fields:
                if field not in plot_df.columns:
                    continue
                target_dir = out_dir / data_cateogry
                os.makedirs(target_dir, exist_ok=True)

                if field == 'AccumulatedDeltaRangeState':
                    y_data = plot_df[field] & 7
                else:
                    y_data = plot_df[field]
                

                plt.figure(figsize=(12, 6))
                plt.scatter(
                    plot_df["gps_millis"],
                    y_data,
                    s=10,          # dots a bit larger
                    marker="o",
                    color="darkred",
                    linewidths=0,
                    label=field,
                )
                plt.title(f"{field} vs. GPS Time  (PRN {target_svid})")
                plt.xlabel("GPS Time (ms)")
                plt.ylabel("Value")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(target_dir / f"{fname}_{target_svid}.png" , dpi=150)
                plt.close()

        
        
        if all(col in plot_df.columns for col in ["raw_pr_m", "accumulated_delta_range_m", "PseudorangeRateMetersPerSecond"]):
            os.makedirs(out_dir / 'complex', exist_ok=True)

            # 정렬
            plot_df = plot_df.sort_values("gps_millis")

            # PR Difference
            plot_df["pr_diff"] = plot_df["raw_pr_m"].diff()
            plot_df["double_pr_diff"] = plot_df["pr_diff"].diff()

            # Carrier Difference
            plot_df["carrier_diff"] = plot_df["accumulated_delta_range_m"].diff()

            # Doppler (그대로 사용: m/s 단위)
            doppler = plot_df["PseudorangeRateMetersPerSecond"]

            # import matplotlib.pyplot as plt

            # 시각화 (scatter)
            plt.figure(figsize=(14, 7))
            plt.scatter(plot_df["gps_millis"], plot_df["pr_diff"], s=6, alpha=0.3, color="blue", label="ΔPR (m)")
            plt.scatter(plot_df["gps_millis"], plot_df["carrier_diff"], s=8, alpha=0.7, color="orange", label="ΔCarrier (m)")
            plt.scatter(plot_df["gps_millis"], doppler, s=6, alpha=0.2, color="red", marker="^", label="Doppler (m/s)")
            plt.ylim([-250, 250])

            plt.title(f"ΔPR, ΔCarrier, Doppler vs GPS Time (PRN {target_svid})")
            plt.xlabel("GPS Time (ms)")
            plt.ylabel("Values (m or m/s)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'complex' / f"pr_carrier_doppler_prn{target_svid}.png", dpi=150)
            plt.close()


            # 시각화 (scatter)
            plt.figure(figsize=(12, 6))
            data = plot_df["pr_diff"]
            data = data - data.mean()
            plt.scatter(plot_df["gps_millis"], data, s=10, color="blue", label="ΔPR (m)")
            plt.ylim([-250, 250])

            plt.title(f"ΔPR vs GPS Time (PRN {target_svid})")
            plt.xlabel("GPS Time (ms)")
            plt.ylabel("ΔPR (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'complex' / f"pr_diff_prn{target_svid}.png", dpi=150)
            plt.close()

            # 시각화 (scatter)
            plt.figure(figsize=(12, 6))
            data = plot_df["double_pr_diff"]
            data = data - data.mean()
            plt.scatter(plot_df["gps_millis"], data, s=10, color="blue", label="ΔPR (m)")
            plt.ylim([-50, 50])

            plt.title(f"ΔPR vs GPS Time (PRN {target_svid})")
            plt.xlabel("GPS Time (ms)")
            plt.ylabel("ΔPR (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'complex' / f"pr_double_diff_prn{target_svid}.png", dpi=150)
            plt.close()

            data = plot_df["raw_pr_m"] - plot_df["accumulated_delta_range_m"]
            data = data - data.mean()
            # 시각화 (scatter)
            plt.figure(figsize=(12, 6))
            plt.scatter(plot_df["gps_millis"], data, s=10, color="blue", label="ΔPR (m)")
            plt.ylim([-250, 250])

            plt.title(f"Code - Carrier vs GPS Time (PRN {target_svid})")
            plt.xlabel("GPS Time (ms)")
            plt.ylabel("ΔPR (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'complex' / f"code - carrier{target_svid}.png", dpi=150)
            plt.ylim([-100,100])
            plt.close()

def main_txt(target_file_name: str, calculate_sv_position = True, is_debug = False):
    target_file_dir = BASE_DIR / target_file_name

    raw_data = android.AndroidRawGnss(input_path=target_file_dir,filter_measurements=False,verbose=False)
    df = raw_data.pandas_df()

    
    pd.set_option("display.float_format", "{:.10f}".format)  # 소수점 10자리까지 표시

    C = 299_792_458.0  # m/s
    df["doppler_hz"] = - (df["PseudorangeRateMetersPerSecond"] / C) * df["CarrierFrequencyHz"]
    df['phase_cycle'] = df['accumulated_delta_range_m'] / (C / df['CarrierFrequencyHz'])

    print(df.columns)
    cols = [
        "gnss_id", "sv_id", "CarrierFrequencyHz", "gps_millis",
        "doppler_hz", "phase_cycle", "PseudorangeRateMetersPerSecond",
        "accumulated_delta_range_m", "raw_pr_m"
    ]

    df_sorted = df.sort_values(by=["gps_millis", "gnss_id", "sv_id", "CarrierFrequencyHz"])
    print(df_sorted[df_sorted["gnss_id"] == "gps"][cols].head())
    # assert False

    if is_debug:
        target_constellation = {
            'gps' : [
                1_575_420_030,
                1_176_450_050
            ],
            'beidou' : [
                1_575_420_030,
                1_561_097_980,
                1_176_450_050
            ],
            'galileo' : [
                1_575_420_030,
                1_176_450_050
            ],
            'qzss' : [
                1_575_420_030,
                1_176_450_050
            ]
        }

        for target_constellation, frequency_list in target_constellation.items():
            for target_freq in frequency_list:
                _check_each_clock_value(df, target_constellation=target_constellation,
                                            target_freq=target_freq,
                                            save_dir=Path('./fig/sample') / target_file_name)
                    
        return

    # 특정 주파수 값(1176450050) 필터링하여 출력
    filtered_df = df
    print(filtered_df[['gnss_id', 'CodeType', 'CarrierFrequencyHz']].drop_duplicates())

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
    df['loi'] = df.apply(lambda row: row['AccumulatedDeltaRangeState'] & 4, axis=1)

    START_GPS_TIME = df['gps_time'].min()
    END_GPS_TIME = df['gps_time'].max()
    INTERVAL = END_GPS_TIME - (START_GPS_TIME + 1) + 1
    TOTAL_SAT_IDX = 182

    CONSTELLATION = {'C' : 46,
                    'G' : 32,
                    'E' : 60,}
    
    frequency_df['gnss_idx'] = (frequency_df['gnss_id'].map(IDX_MAPPING).fillna(1).astype(int) - 1) + (frequency_df['sv_id'] - 1)

    all_sats = [f"{each_code}{sv:02d}" for each_code, max_num in CONSTELLATION.items() for sv in range(1, max_num + 1)]

    result_dict = init_matlab_data_format(START_GPS_TIME, INTERVAL, TOTAL_SAT_IDX)
    ephem = None

    full_idx = np.arange(TOTAL_SAT_IDX)

    grouped_df = df.groupby('gps_time')

    for idx, each_gps_time in enumerate(tqdm(range(START_GPS_TIME + 1, END_GPS_TIME + 1), desc="Processing GPS Times")):
        try:
            current_df = grouped_df.get_group(each_gps_time).copy()
        except KeyError:
            continue

        result_dict['time'][idx] = current_df['gps_millis'].iloc[0] / 1000.0 - START_GPS_TIME

        for signal, freq_list in {'1': [1575420030,1561097980], '3': [1176450050]}.items():
            frequency_df = current_df[current_df['CarrierFrequencyHz'].isin(freq_list)].copy()

            if current_df.empty:
                continue

            frequency_df['gnss_idx'] = (frequency_df['gnss_id'].map(IDX_MAPPING).fillna(1).astype(int) - 1) + (frequency_df['sv_id'] - 1)
            frequency_df = frequency_df.drop_duplicates(subset=['gnss_idx'], keep='first')
            frequency_df = frequency_df.set_index('gnss_idx').reindex(full_idx).reset_index()

            result_dict[f'pr{signal}'][idx, :] = frequency_df['raw_pr_m'].to_numpy()
            result_dict[f'ph{signal}'][idx, :] = frequency_df['phase_cycle'].to_numpy()
            result_dict[f'dop{signal}'][idx, :] = frequency_df['doppler_hz'].to_numpy()
            result_dict[f'snr{signal}'][idx, :] = frequency_df['cn0_dbhz'].to_numpy()
            result_dict[f'loi{signal}'][idx, :] = frequency_df['loi'].to_numpy()

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


def main_csv(target_file_name: str, calculate_sv_position = True, is_debug = False):
    raw_data = android.AndroidRawGnss(input_path=BASE_DIR / target_file_name,
                                      filter_measurements=False,
                                      verbose=False)

    if calculate_sv_position:
        raw_data = glp.add_sv_states(raw_data)

    df = raw_data.pandas_df()
    df['gps_time'] = df['gps_millis'].apply(lambda x: int(round(x / 1000)))
    df['loi'] = df.apply(lambda row: row['AccumulatedDeltaRangeState'] & 4, axis=1)

    df['is_code_lock'] = (df['State'] & STATE_CODE_LOCK) != 0
    df['is_tow_good'] = ((df['State'] & STATE_TOW_DECODED) != 0) | \
                        ((df['State'] & STATE_TOW_KNOWN) != 0)
    
    df['no_msec_ambiguity'] = (df['State'] & STATE_MSEC_AMBIGUOUS) == 0
    df['valid_pr'] = df['is_code_lock'] & df['is_tow_good'] & df['no_msec_ambiguity']
    df = df[df['valid_pr']]

    CONSTELLATION = {'G' : 32}

    all_sats = [f"{each_code}{sv:02d}" for each_code, max_num in CONSTELLATION.items() for sv in range(1, max_num + 1)]

    grouped_df = df.groupby('gps_time')
    ephem = None
    iono_params = [0.0]*8

    START_GPS_TIME = df['gps_time'].min()
    END_GPS_TIME = df['gps_time'].max()
            
    def keep_first_6_and_zero_rest(x):
        x = int(x)
        s = str(x)

        if len(s) <= 6:
            return x

        prefix = s[:6]
        zeros = '0' * (len(s) - 6)
        return int(prefix + zeros)

    df['exact_doppler_hz'] = df['CarrierFrequencyHz'].apply(keep_first_6_and_zero_rest)
    c = 299792458.0

    df['doppler_hz'] = -(df['PseudorangeRateMetersPerSecond'] / c) * df['exact_doppler_hz']
    df['phase_cycle'] = df['accumulated_delta_range_m'] * df['exact_doppler_hz'] / c

    # print(df.columns)
    # cols = [
    #     "gnss_id", "sv_id", "CarrierFrequencyHz", "gps_millis",
    #     "doppler_hz", "phase_cycle", "PseudorangeRateMetersPerSecond",
    #     "accumulated_delta_range_m", "raw_pr_m", "State"
    # ]

    # df_sorted = df.sort_values(by=["gps_millis", "gnss_id", "sv_id", "CarrierFrequencyHz"])
    # print(df_sorted[(df_sorted["gnss_id"] == "beidou") & (~(df_sorted["State"] & 1))][cols].head())
    # input('here')

    data_list = []

    for idx, each_gps_time in enumerate(tqdm(range(START_GPS_TIME + 1, END_GPS_TIME + 1), desc="Processing GPS Times")):
        try:
            current_df = grouped_df.get_group(each_gps_time).copy()
        except KeyError:
            continue

        if calculate_sv_position:
            time_millis = each_gps_time * 1000
            dt = glp.gps_millis_to_datetime(time_millis)

            if ephem is None or (dt.minute == 0 and dt.second == 0):
                ephem = glp.get_time_cropped_rinex(time_millis, all_sats, ephemeris_directory="./data/ephemeris")
                print(ephem.iono_params)
                if len(ephem.iono_params) == 0 or 'gps' not in list(ephem.iono_params.values())[0]:
                    print("No iono params available!")
                else:
                    iono_params = list(ephem.iono_params.values())[0]['gps'].flatten()

        for _, row in current_df.iterrows():
            measurement = Measurement(
                time=TimeTag.from_sec(each_gps_time),
                sat=SatelliteInfo(
                    sv_pos=[row['x_sv_m'], row['y_sv_m'], row['z_sv_m']],
                    sv_vel=[row['vx_sv_mps'], row['vy_sv_mps'], row['vz_sv_mps']],
                    sv_clock_bias = row['b_sv_m'],
                    sv_clock_drift = row['b_dot_sv_mps'],
                    constellation=row['gnss_id'].upper(),
                    iono_coff=iono_params,
                    prn=row['sv_id'],
                    frequency_hz=row['exact_doppler_hz'],
                    code_type=row['CodeType']
                ),
                value=MeasurementValue(
                    pseudorange_m=row['raw_pr_m'],
                    phase_cycle=row['phase_cycle'],
                    doppler_hz=row['doppler_hz'],
                    snr_dbhz=row['cn0_dbhz'],
                    loi=row['loi']
                )
            )
            data_list.append(measurement.to_csv(sep='\t'))


    base_name, _ = os.path.splitext(target_file_name)

    os.makedirs('./data/csv', exist_ok=True)

    with open(f'./data/csv/{base_name}.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')

        header = Measurement.headers()
        writer.writerow(header)

        for row in data_list:
            f.write(row + '\n')

def main_decimeter(target_file_name: str, year: int):
    if year == 2022:
        function = glp.AndroidDerived2022
        gt_function = glp.AndroidGroundTruth2022
    else:
        function = glp.AndroidDerived2023
        gt_function = glp.AndroidGroundTruth2023

    raw_data = function(f"./data/challenge/{target_file_name}")

    file_name, ext = os.path.splitext(target_file_name)
    gt_data = gt_function(f"./data/challenge/{file_name}_ground_truth{ext}")

    USER_EXTERNAL_CORRECTION = False

    # df = state_ekf.pandas_df()
    # gt_df = gt_data.pandas_df()

    # df = pd.merge(df, gt_df[['gps_millis','x_rx_gt_m','y_rx_gt_m', 'z_rx_gt_m']], on='gps_millis', how='left')

    # return

    df = raw_data.pandas_df()
    gt_df = gt_data.pandas_df()
    state_ekf = glp.solve_gnss_ekf(raw_data).pandas_df()

    df = pd.merge(df, gt_df[['gps_millis','x_rx_gt_m','y_rx_gt_m', 'z_rx_gt_m']], on='gps_millis', how='left')
    df = pd.merge(df, state_ekf[['gps_millis','x_rx_ekf_m','y_rx_ekf_m', 'z_rx_ekf_m']], on='gps_millis', how='left')

    IDX_MAPPING = {
        'gps' : 1,
        'galileo' : 60,
        'beidou' : 103,
        'glonass' : 33,
        'qzss' : 96
    }

    # ---- (A) gnss_idx 부여 ----
    df = attach_gnss_idx(df, IDX_MAPPING)
    df['gps_millis'] = ((df['gps_millis'] / 1000).round() * 1000).astype('int64')

    if USER_EXTERNAL_CORRECTION:
        # # ---- (B) correction CSV 로드 & long 변환 ----
        corr_long_l1 = load_corrections_long(f"./data/challenge/{file_name}_corrected_L1{ext}", frequency_hz=1575420000, target_column_name='pr_correction_m_additional')
        corr_long_l5 = load_corrections_long(f"./data/challenge/{file_name}_corrected_L5{ext}", frequency_hz=1176450000, target_column_name='pr_correction_m_additional')
        corr_long = pd.concat([corr_long_l1, corr_long_l5], ignore_index=True)

        corr_long['gps_millis'] = ((corr_long['gps_millis'] / 1000).round() * 1000).astype('int64')

        df = df.merge(
            corr_long,
            on=['gps_millis', 'gnss_idx', 'CarrierFrequencyHz'],
            how='left'
        )

    df['gps_time'] = df['gps_millis'].apply(lambda x: int(round(x / 1000)))
    df['loi'] = df.apply(lambda row: row['AccumulatedDeltaRangeState'] & 4, axis=1)

    df['pr_correction_m'] = df['corr_pr_m'] - df['raw_pr_m']
    
    grouped_df = df.groupby('gps_time')
    iono_params = [0.0]*8

    START_GPS_TIME = df['gps_time'].min()
    END_GPS_TIME = df['gps_time'].max()

    data_list = []
    ekf_result_list = []

    kml = simplekml.Kml()
    track_ekf = kml.newgxtrack(name="EKF Track")
    track_gt = kml.newgxtrack(name="GT Track")

    ekf_when_list = []
    gt_when_list = []
    ekf_coord_list = []
    gt_coord_list = []

    for idx, each_gps_time in enumerate(tqdm(range(START_GPS_TIME + 1, END_GPS_TIME + 1), desc="Processing GPS Times")):
        try:
            current_df = grouped_df.get_group(each_gps_time).copy()
        except KeyError:
            print('No data for gps_time:', each_gps_time)
            continue

        first_row = True  # ★ 이 epoch에서 첫 번째 row인지 체크용

        current_df['DopplerHz'] = -current_df['PseudorangeRateMetersPerSecond'] / 299792458 * current_df['CarrierFrequencyHz']
        current_df['phase_cycle'] = current_df['accumulated_delta_range_m'] / (299792458 / current_df['CarrierFrequencyHz'])

        for _, row in current_df.iterrows():
            if USER_EXTERNAL_CORRECTION :
                current_correction = row['pr_correction_m_additional']
            else:
                current_correction = row['pr_correction_m']

            sv_location = [row['x_sv_m'], row['y_sv_m'], row['z_sv_m']]

            measurement = Measurement(
                time=TimeTag.from_sec(each_gps_time),
                sat=SatelliteInfo(
                    sv_pos=sv_location,
                    sv_vel=[row['vx_sv_mps'], row['vy_sv_mps'], row['vz_sv_mps']],
                    sv_clock_bias = row['b_sv_m'],
                    sv_clock_drift = row['b_dot_sv_mps'],
                    constellation=row['gnss_id'].upper(),
                    iono_coff=iono_params,
                    prn=row['sv_id'],
                    frequency_hz=int(row['CarrierFrequencyHz']),
                    code_type=row['CodeType'],
                    pr_correction=current_correction,
                ),
                value=MeasurementValue(
                    pseudorange_m=row['raw_pr_m'],
                    phase_cycle=row['phase_cycle'],
                    doppler_hz=row['DopplerHz'],
                    snr_dbhz=row['cn0_dbhz'],
                    loi=row['loi']
                ),
                ground_truth=[row['x_rx_gt_m'], row['y_rx_gt_m'], row['z_rx_gt_m']]
            )
            data_list.append(measurement.to_csv(sep='\t'))

            # ---------- ★ 여기부터 KML용 코드 추가 ----------
            if first_row:
                first_row = False

                ekf_result_list.append((
                    each_gps_time,
                    row['x_rx_ekf_m'],
                    row['y_rx_ekf_m'],
                    row['z_rx_ekf_m'],
                    row['x_rx_gt_m'],
                    row['y_rx_gt_m'],
                    row['z_rx_gt_m']
                ))

                time_str = gps_sec_to_iso(each_gps_time)

                ecef_ekf = np.array([[row['x_rx_ekf_m'],
                                    row['y_rx_ekf_m'],
                                    row['z_rx_ekf_m']]])   # (1,3)
                geod_ekf = ecef_to_geodetic(ecef_ekf)        # (1,3)  -> [[lat, lon, h]]
                lat_ekf, lon_ekf, h_ekf = [float(v) for v in geod_ekf[0]]

                # ----- GT: ECEF -> Geodetic -----
                ecef_gt = np.array([[row['x_rx_gt_m'],
                                    row['y_rx_gt_m'],
                                    row['z_rx_gt_m']]])     # (1,3)
                geod_gt = ecef_to_geodetic(ecef_gt)
                lat_gt, lon_gt, h_gt = [float(v) for v in geod_gt[0]]

                # KML은 (lon, lat, alt) 순서
                ekf_coord = (lon_ekf, lat_ekf, h_ekf)
                gt_coord  = (lon_gt,  lat_gt,  h_gt)

                ekf_when_list.append(time_str)
                gt_when_list.append(time_str)
                ekf_coord_list.append(ekf_coord)
                gt_coord_list.append(gt_coord)


    # EKF track
    track_ekf.newwhen(ekf_when_list)
    track_ekf.newgxcoord(ekf_coord_list)
    track_ekf.style.linestyle.width = 3
    track_ekf.style.linestyle.color = simplekml.Color.red

    track_gt.newwhen(gt_when_list)
    track_gt.newgxcoord(gt_coord_list)
    track_gt.style.linestyle.width = 3
    track_gt.style.linestyle.color = simplekml.Color.blue
    
    base_name, _ = os.path.splitext(target_file_name)

    os.makedirs('./data/csv', exist_ok=True)
    os.makedirs('./data/kml', exist_ok=True)

    kml.save(f"./data/kml/{base_name}.kml")

    with open(f'./data/csv/{base_name}.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(Measurement.headers())

        for row in data_list:
            f.write(row + '\n')

    with open(f'./data/csv/{base_name}_kf_no_correction.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow([
            "time_sec",
            "pos_x", "pos_y", "pos_z",
            "pos_x_true", "pos_y_true", "pos_z_true"
        ])

        for row in ekf_result_list:
            writer.writerow(row)

import re
import numpy as np
import pandas as pd

def load_corrections_long(corr_csv_path: str, frequency_hz: int, target_column_name: str) -> pd.DataFrame:
    """
    변환된 correction CSV(열: UnixMillis, corr_001..corr_182)를
    long 형태로 변환:
      [gps_millis, gnss_idx, pr_correction_m_additional, CarrierFrequencyHz]
    + gps_time(초) 추가
    """
    corr = pd.read_csv(corr_csv_path)
    corr['unix_millis'] = corr['UnixMillis']

    # corr_* 컬럼 찾기
    corr_cols = [c for c in corr.columns if re.match(r'^corr_\d{3}$', c)]
    if not corr_cols:
        raise ValueError("correction CSV에 'corr_001' 형태의 열이 없습니다.")

    # wide → long
    corr_long = corr.melt(
        id_vars=['unix_millis'],
        value_vars=corr_cols,
        var_name='corr_col',
        value_name=target_column_name
    )

    # corr_001 → 0-based index(= 0..181)
    corr_long['gnss_idx'] = corr_long['corr_col'].str.extract(r'(\d{3})').astype(int) - 1
    corr_long = corr_long[['unix_millis', 'gnss_idx', target_column_name]]

    # 타입 정리
    corr_long['gnss_idx'] = corr_long['gnss_idx'].astype('int32')
    corr_long[target_column_name] = pd.to_numeric(corr_long[target_column_name], errors='coerce')
    corr_long['CarrierFrequencyHz'] = frequency_hz

    # --- GPS Time 추가 ---
    # UNIX epoch(1970) → GPS epoch(1980) 보정
    UNIX_GPS_OFFSET = 315964800 - 18  # 18 leap seconds 보정 (UTC→GPS)
    corr_long['gps_time'] = corr_long['unix_millis'] / 1000 - UNIX_GPS_OFFSET  # [초 단위]

    # gps_time을 밀리초로 추가하고 싶으면 아래도 가능:
    corr_long['gps_millis'] = corr_long['gps_time'] * 1000

    return corr_long

def attach_gnss_idx(df: pd.DataFrame, IDX_MAPPING: dict) -> pd.DataFrame:
    """
    df에 gnss_idx 열을 추가:
      gnss_idx = (IDX_MAPPING[gnss_id] - 1) + (sv_id - 1)
    - gnss_id는 대문자로 통일
    - sv_id는 1-based라고 가정
    """
    tmp = df.copy()
    tmp['gnss_id'] = tmp['gnss_id'].astype(str)
    tmp['sv_id'] = tmp['sv_id'].astype(int)

    tmp['gnss_idx'] = (
        tmp['gnss_id'].map(IDX_MAPPING).astype(int) - 1
    ) + (tmp['sv_id'] - 1)

    tmp['gnss_idx'] = tmp['gnss_idx'].astype('int32')

    # print(tmp[['gnss_id', 'sv_id', 'gnss_idx']].drop_duplicates())
    # input()
    return tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS Satellite Selection and Processing")
    parser.add_argument("--input_file_name","-n", type=str, default='29740_gnss_log_2025_09_28_17_00_44.txt', help="Path to GNSS log file")
    parser.add_argument("--calc_sv", '-c', action="store_false")
    parser.add_argument("--debug", '-d', action='store_true')
    parser.add_argument("--file_format", '-f', type=str, default='csv', choices=['mat','csv'], help="Input file format")
    parser.add_argument("--year", '-y', type=int, default=2022, help="Year for processing")

    args = parser.parse_args()

    # main_txt('29740_gnss_log_2025_05_07_15_36_12.txt', calculate_sv_position=True, is_debug=True)
    # main_csv('29740_gnss_log_2025_05_07_15_36_12.txt', calculate_sv_position=False, is_debug=True)
    # main_csv('29740_gnss_log_2025_09_28_17_00_44.txt', calculate_sv_position=True, is_debug=False)
    main_csv('29740_gnss_log_2025_09_28_17_00_44_test.txt', calculate_sv_position=False, is_debug=False)
    # main_csv('test.txt', calculate_sv_position=True, is_debug=False)

    # print(args.input_file_name, args.calc_sv)
    # main_decimeter('2023-05-23-19-16-us-ca-mtv-ie2.csv', 2023)
    # main_decimeter('2023-05-19-20-10-us-ca-mtv-ie2.csv', 2023)
    # main_decimeter('2023-05-09-21-32-us-ca-mtv-pe1.csv', 2023)
    # main_decimeter('2022-08-04-20-07-us-ca-sjc-q_pixel5.csv', 2023)
    # main_decimeter('2022-01-26-20-02-us-ca-mtv-pe1_pixel5.csv', 2023)
    # main_decimeter('2023-05-26-18-51-us-ca-sjc-ge2.csv', 2023)
    # main_decimeter('2023-09-05-20-13-us-ca.csv', 2023)
    # main_decimeter('2023-09-05-23-07-us-ca-routen.csv', 2023)
    # main_decimeter('2023-09-06-18-04-us-ca.csv', 2023)
    # main_decimeter('2023-09-07-18-59-us-ca.csv', 2023)
    # main_decimeter('2023-09-06-22-49-us-ca-routebb1_samsung_s.csv', 2023)

    # main_decimeter('device_gnss2022.csv', 2022)
    # main_decimeter('device_gnss2023.csv', 2023)

    # if args.file_format == 'csv':
    #     main_csv(args.input_file_name, args.calc_sv)
    # elif args.file_format == 'mat':
    #     main_txt(args.input_file_name, args.calc_sv, args.debug)



# print(df.columns.tolist())

        # ── 5) ReceivedSvTimeNanos 앞 10개 콘솔 출력 ────────────────────────
        # if "ReceivedSvTimeNanos" in plot_df.columns:
        #     print(
        #         f"\n📌 ReceivedSvTimeNanos – 처음 10개 (PRN {target_svid}, {target_freq} Hz)"
        #     )
        #     print(
        #         plot_df[
        #             [
        #                 "gps_millis",
        #                 "TimeNanos",
        #                 "ReceivedSvTimeNanos",
        #                 "FullBiasNanos",
        #                 "TimeOffsetNanos",
        #                 "BiasNanos",
        #                 "CarrierFrequencyHz",
        #                 "raw_pr_m",
        #             ]
        #         ].head(10)
        #     )

        # neg_mask = plot_df["raw_pr_m"] > 2e10
        # if neg_mask.any():
        # # 👉 디버깅용 샘플 확인
        #     dbg_cols = [
        #         "gps_millis", "sv_id", "CarrierFrequencyHz",
        #         "TimeNanos", "ReceivedSvTimeNanos",
        #         "FullBiasNanos", "TimeOffsetNanos", "BiasNanos",
        #         "raw_pr_m",
        #     ]
        #     print(f"⚠️ raw_pr_m < 0 인 행 {neg_mask.sum()}개 / {len(plot_df)}개")
        #     print(plot_df.loc[neg_mask, dbg_cols].head(20))

        #     # (a) 분석에서 제외
        #     plot_df = plot_df.loc[~neg_mask].copy()

        # ── 6) raw_pr_m scatter 플롯 ─────────────────────────────────────────