
import gnss_lib_py as glp
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

            # 시각화 (scatter)
            plt.figure(figsize=(14, 7))
            plt.scatter(plot_df["gps_millis"], plot_df["pr_diff"], s=10, color="blue", label="ΔPR (m)")
            plt.scatter(plot_df["gps_millis"], plot_df["carrier_diff"], s=10, color="orange", label="ΔCarrier (m)")
            plt.scatter(plot_df["gps_millis"], doppler, s=30, edgecolors="red", facecolors="none", marker="^", alpha=0.5, label="Doppler (m/s)")
            # plt.ylim([-10000,10000])

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
            plt.scatter(plot_df["gps_millis"], plot_df["pr_diff"], s=10, color="blue", label="ΔPR (m)")
            plt.ylim([-4000, 4000])

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
            plt.scatter(plot_df["gps_millis"], plot_df["double_pr_diff"], s=10, color="blue", label="ΔPR (m)")
            plt.ylim([-1000, 1000])

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

            plt.title(f"Code - Carrier vs GPS Time (PRN {target_svid})")
            plt.xlabel("GPS Time (ms)")
            plt.ylabel("ΔPR (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'complex' / f"code - carrier{target_svid}.png", dpi=150)
            plt.ylim([-150000,0])
            plt.close()

def main(target_file_name: str, calculate_sv_position = True, is_debug = False):
    BASE_DIR = Path('./data/raw')
    target_file_dir = BASE_DIR / target_file_name

    target_data = {}

    raw_data = android.AndroidRawGnss(input_path=target_file_dir,filter_measurements=False,verbose=False)
    # raw_data = raw_data[raw_data['signal_type'].isin(['l1', 'e1', 'b1'])].copy()
    # full_states = glp.add_sv_states_rinex(raw_data, constellations = ['gps', 'beidou', 'galileo'])
    # full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states['b_sv_m']

    df = raw_data.pandas_df()

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
    df['loi'] = df.apply(lambda row: row['AccumulatedDeltaRangeState'] & 4, axis=1)

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

    ### Process For L1 Frequency
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
            result_dict[f'ph{signal}'][idx, :] = frequency_df['accumulated_delta_range_m'].to_numpy()
            result_dict[f'dop{signal}'][idx, :] = frequency_df['PseudorangeRateMetersPerSecond'].to_numpy()
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

    ### Process For L5 Frequency

    # 확장자 변경
    base_name, _ = os.path.splitext(target_file_name)
    new_target_file_name = base_name + '.mat'

    # result_dict를 저장하는 부분
    scipy.io.savemat(f'./data/mat/{new_target_file_name}', result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS Satellite Selection and Processing")
    parser.add_argument("--input_file_name","-n", type=str, default='29740_gnss_log_2025_05_07_15_36_12.txt', help="Path to GNSS log file")
    parser.add_argument("--calc_sv", '-c', action="store_false")
    parser.add_argument("--debug", '-d', action='store_true')

    args = parser.parse_args()
    
    print(args.input_file_name, args.calc_sv)
    main(args.input_file_name, args.calc_sv, args.debug)



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