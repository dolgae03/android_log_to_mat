"""Functions to process Android measurements.

Tested on Google Android's GNSSLogger App v3.0.6.4

Includes functionality for:
  * Fix Measurements
  * Raw Measurements
  * Accel Measurements
  * Gyro Measurements
  * Mag Measurements
  * Bearing Measurements

"""

__authors__ = "Ashwin Kanhere, Derek Knowles, Shubh Gupta, Adam Dai"
__date__ = "02 Nov 2021"


import os
import csv

import numpy as np
import pandas as pd

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.navdata.operations import loop_time
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.sv_models import add_sv_states
from gnss_lib_py.utils.time_conversions import get_leap_seconds
from gnss_lib_py.utils.time_conversions import unix_to_gps_millis

class AndroidRawGnss:
    """Handles Raw GNSS measurements from Android.

    Data types in the Android's GNSSStatus messages are documented on
    their website [1]_.

    Parameters
    ----------
    input_path : string or path-like
        Path to measurement csv or txt file.
    filter_measurements : bool
        Filter noisy measurements based on known conditions.
    measurement_filters : dict
        Conditions under which measurements should be filtered. An
        emptry dictionary passed into measurement_filters is
        equivalent to setting filter_measurements to False. See the
        docstring for ``filter_raw_measurements`` for details.
    remove_rx_b_from_pr : bool
        If true, removes the estimated initial receiver clock bias
        from the pseudorange values.
    verbose : bool
        If true, prints extra debugging statements.

    References
    ----------
    .. [1] https://developer.android.com/reference/android/location/GnssStatus


    """
    def __init__(self, input_path,
                 filter_measurements=True,
                 measurement_filters = {"bias_valid" : True,
                                        "bias_uncertainty" : 40.,
                                        "arrival_time" : True,
                                        "unknown_constellations" : True,
                                        "time_valid" : True,
                                        "state_decoded" : True,
                                        "sv_time_uncertainty" : 500.,
                                        "adr_valid" : True,
                                        "adr_uncertainty" : 15.
                                        },
                 remove_rx_b_from_pr = False,
                 verbose=False):

        print('herehkdsfjasfdk')
        
        self.verbose = verbose
        self.filter_measurements = filter_measurements
        self.measurement_filters = measurement_filters
        self.remove_rx_b_from_pr = remove_rx_b_from_pr
        self.df = self.preprocess(input_path)
        print(self.df['FullBiasNanos'][0])
        self.postprocess()

    def pandas_df(self):
        return self.df

    def preprocess(self, input_path):
        """Extract Raw measurements.

        Built on the first parts of make_gnss_dataframe and correct_log

        Parameters
        ----------
        input_path : string or path-like
            Path to measurement csv or txt file.

        Returns
        -------
        measurements : pd.DataFrame
            Subset of "Raw" measurements.

        """

        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            skip_rows = []
            header_row = None
            for row in reader:
                if len(row) == 0:
                    skip_rows.append(row_idx)
                elif len(row[0]) == 0:
                    skip_rows.append(row_idx)
                elif row[0][0] == '#':
                    if 'Raw' in row[0]:
                        header_row = row_idx
                    elif header_row is not None:
                        skip_rows.append(row_idx)
                elif row[0] != 'Raw':
                    skip_rows.append(row_idx)
                row_idx += 1

        print(header_row)
        measurements = pd.read_csv(input_path,
                                    skip_blank_lines = False,
                                    header = header_row,
                                    skiprows = skip_rows,
                                        dtype={
                                            'AccumulatedDeltaRangeUncertaintyMeters': np.float64,
                                            'TimeOffsetNanos': np.float64,
                                            'BiasNanos': np.float64,
                                            'ReceivedSvTimeUncertaintyNanos': np.float64,
                                            'DriftNanosPerSecond': np.float64,
                                            'DriftUncertaintyNanosPerSecond': np.float64,
                                            
                                            'FullBiasNanos': np.int64,
                                            'TimeNanos': np.int64,
                                            'ReceivedSvTimeNanos': np.int64,
                                        }
                                    )

        return measurements

    def postprocess(self):
        """Postprocess loaded NavData.

        Strategy for computing the arrival time was taken from an
        EUSPA white paper [2]_ and Google's source code
        Arrival time taken from [3]_.

        References
        ----------
        .. [2] https://www.euspa.europa.eu/system/files/reports/gnss_raw_measurement_web_0.pdf
        .. [3] https://github.com/google/gps-measurement-tools/blob/master/opensource/ProcessGnssMeas.m
        """

        # rename gnss_id
        gnss_id = np.array([consts.CONSTELLATION_ANDROID[i] for i in self.df["ConstellationType"]])
        self.df["gnss_id"] = gnss_id
        self.df['sv_id'] = self.df['Svid']
        self.df['cn0_dbhz'] = self.df['Cn0DbHz']
        self.df['accumulated_delta_range_m'] = self.df['AccumulatedDeltaRangeMeters']
        self.df['accumulated_delta_range_sigma_m'] = self.df['AccumulatedDeltaRangeUncertaintyMeters']

        # update svn for QZSS constellation
        qzss_mask = self.df["gnss_id"] == "qzss"

        # 해당 행들의 sv_id 값을 새로운 리스트로 치환
        self.df.loc[qzss_mask, "sv_id"] = [
            consts.QZSS_PRN_SVN[i] for i in self.df.loc[qzss_mask, "sv_id"]
        ]

        # add singal type information where available
        self.df["signal_type"] = np.array([consts.CODE_TYPE_ANDROID.get(x,{}).get(y,"") \
                                        for x,y in zip(self.df["gnss_id"],
                                                       self.df["CodeType"])])

        # add gps milliseconds
        self.df["gps_millis"] = unix_to_gps_millis(self.df["utcTimeMillis"])

        # 주(週) 나노초 단위 상수
        WEEK_NANOS = consts.WEEKSEC * int(1e9)
        # ────────────────── ① 점프 탐지 & 세그먼트 ID 생성
        jumps = self.df["FullBiasNanos"].diff().abs() > int(5e7)
        segment_id = jumps.cumsum()          # 첫 구간은 0, jump마다 +1
        print(segment_id.max())

        # ────────────────── ② 세그먼트별 기준값(first() 이용)
        base_full_bias  = self.df.groupby(segment_id)["FullBiasNanos"].transform("first")
        base_bias_nanos = self.df.groupby(segment_id)["BiasNanos"].transform("first")
        # ────────────────── ② 세그먼트별 기준값(first() 이용)
        # base_full_bias  = self.df["FullBiasNanos"]
        # base_bias_nanos = self.df["BiasNanos"]
        # ────────────────── ③ GPS 주(週) 보정
        gps_week_nanos = (-self.df["FullBiasNanos"] // WEEK_NANOS) * WEEK_NANOS
        self.df["FB-gps_week"] = base_full_bias + gps_week_nanos

        # ────────────────── ④ 수신·송신 시각 차이
        tx_rx_gnss_diff_ns = (
            self.df["TimeNanos"]
            - (base_full_bias + gps_week_nanos)
        )

        tx_rx_gnss_diff_ns = tx_rx_gnss_diff_ns - ((tx_rx_gnss_diff_ns // WEEK_NANOS) * WEEK_NANOS)

        # # calculate pseudorange
        # gps_week_nanos = (-self.df["FullBiasNanos"] // (consts.WEEKSEC * int(1e9))) * (consts.WEEKSEC * int(1e9))
        # tx_rx_gnss_diff_ns = (self.df["TimeNanos"] - self.df["ReceivedSvTimeNanos"] - (self.df["FullBiasNanos",0] + gps_week_nanos) )
        # # tx_rx_gnss_ns = self.df["TimeNanos"] - self.df["FullBiasNanos",0] + self.df["TimeOffsetNanos"] - self.df["BiasNanos",0]

        # 수신 시간 초기화
        t_rx_ns = np.zeros(len(self.df), dtype=np.int64)
        print(type(t_rx_ns[0]))

        # gps constellation
        tx_rx_gps_ns = tx_rx_gnss_diff_ns
        t_rx_ns = np.where(self.df["gnss_id"]=="gps",
                             tx_rx_gps_ns,
                             t_rx_ns)
        print('gps', type(t_rx_ns[0]))
    
        # beidou constellation
        tx_rx_beidou_secs = tx_rx_gnss_diff_ns - 14 * int(1e9)
        t_rx_ns = np.where(self.df["gnss_id"]=="beidou",
                             tx_rx_beidou_secs,
                             t_rx_ns)
        print('beidou', type(t_rx_ns[0]))

        # galileo constellation
        # nanos_per_100ms = 100*1E6
        # ms_number_nanos = np.floor(-self.df["FullBiasNanos"]/nanos_per_100ms)*nanos_per_100ms
        # tx_rx_galileo_secs = (tx_rx_gnss_ns - ms_number_nanos)*1E-9
        t_rx_ns = np.where(self.df["gnss_id"]=="galileo",
                             tx_rx_gnss_diff_ns,
                             t_rx_ns)
        print('galileo', type(t_rx_ns[0]))

        # nanos_per_day = int(1E9 * 24 * 60 * 60)
        # day_number_nanos = (-self.df["FullBiasNanos"] // nanos_per_day * nanos_per_day).astype(np.int64)

        # tx_rx_glonass_ns = (
        #     (tx_rx_gnss_diff_ns - day_number_nanos) 
        #     + int(3 * 60 * 60 * 1e9) 
        #     - int(get_leap_seconds(self.df["gps_millis", 0].item()) * 1e9)
        # )

        # print('glonass', type(t_rx_ns[0]))

        # t_rx_ns = np.where(
        #     self.df["gnss_id"] == "glonass",
        #     tx_rx_glonass_ns.astype(np.int64),  # 명시적으로 int64로 변환
        #     t_rx_ns.astype(np.int64)            # 기존 값을 int64로 변환
        # )

        t_rx_ns = np.where(self.df["gnss_id"]=="glonass",
                             tx_rx_gnss_diff_ns,
                             t_rx_ns)
        print('galileo', type(t_rx_ns[0]))

        print('glonass', type(t_rx_ns[0]))

        # qzss constellation
        tx_rx_qzss_ns = tx_rx_gnss_diff_ns
        t_rx_ns = np.where(self.df["gnss_id"]=="qzss",
                             tx_rx_qzss_ns,
                             t_rx_ns)

        # sbas constellation
        tx_rx_sbas_ns = tx_rx_gnss_diff_ns
        t_rx_ns = np.where(self.df["gnss_id"]=="sbas",
                             tx_rx_sbas_ns,
                             t_rx_ns)

        # irnss constellation
        tx_rx_irnss_secs = tx_rx_gnss_diff_ns
        t_rx_ns = np.where(self.df["gnss_id"]=="irnss",
                             tx_rx_irnss_secs,
                             t_rx_ns)
        # unknown constellation
        tx_rx_unknown_secs = tx_rx_gnss_diff_ns
        t_rx_ns = np.where(self.df["gnss_id"]=="unknown",
                             tx_rx_unknown_secs,
                             t_rx_ns)
        
        print('qzss', type(t_rx_ns[0]))
    

        self.df['Final_Plus(Timeoffsetnanos - biasnanos)'] = (self.df["TimeOffsetNanos"] - base_bias_nanos)
        # 최종 계산 과정
        final_calculation = t_rx_ns - self.df["ReceivedSvTimeNanos"] + (self.df["TimeOffsetNanos"] - base_bias_nanos)
        print('final_calc', type(t_rx_ns[0]))

        # 수신 시각과 송신 시각 차이로부터 의사거리 계산
        raw_pr_m = final_calculation * (consts.C * 1E-9)  # t_rx_secs는 이미 float64

        def print_debug_info(
            idx,
            gnss_id,
            prn,
            frequency,
            code_type,
            raw_pr_m,
            full_bias_nanos,
            time_nanos,
            received_sv_time_nanos,
            time_offset_nanos,
            bias_nanos,
            gps_week_nanos,
            tx_rx_gnss_diff_ns,
            t_rx_ns,
            final_calculation,
        ):
            print(f"\n[DEBUG] 음수 Pseudorange 발생 at index {idx}")
            print(f"\nGnss id and cod type : {gnss_id} / {code_type} / {prn} / {frequency}")
            print(f"raw_pr_m              = {raw_pr_m:.10f} m")

            print("\n--- GNSS Timing Info ---")
            print(f"FullBiasNanos         = {full_bias_nanos} / {type(full_bias_nanos)}")
            print(f"TimeNanos             = {time_nanos} / {type(full_bias_nanos)}")
            print(f"ReceivedSvTimeNanos   = {received_sv_time_nanos} / {type(full_bias_nanos)}")
            print(f"TimeOffsetNanos       = {time_offset_nanos:.10f}")
            print(f"BiasNanos             = {bias_nanos:.10f}")
            print(f"gps_week_nanos        = {gps_week_nanos:.10f}")

            print("\n--- 계산 중간 결과 ---")
            print(f"t_rx_ns               = {t_rx_ns:.10f}")
            print(f"tx_rx_gnss_diff_ns    = {tx_rx_gnss_diff_ns:.10f}")
            print(f"final_calculation     = {final_calculation:.10f}")
            print("-" * 60)

        mask = (raw_pr_m < 0) #& (self.df["sv_id"] == 12)   # ← 컬럼명이 sv_id/svid 중 실제 이름에 맞춰주세요

        # negative_indices = np.where(mask)[0]
        # if len(negative_indices) > 0:
        #     for idx in negative_indices:
        #         print_debug_info(
        #             idx=idx,
        #             gnss_id=self.df['gnss_id'][idx],
        #             prn = self.df['Svid'][idx],
        #             code_type=self.df['CodeType'][idx],
        #             frequency=self.df['CarrierFrequencyHz'][idx],
        #             raw_pr_m=raw_pr_m[idx],
        #             full_bias_nanos=self.df["FullBiasNanos"][idx],
        #             time_nanos=self.df["TimeNanos"][idx],
        #             received_sv_time_nanos=self.df["ReceivedSvTimeNanos"][idx],
        #             time_offset_nanos=self.df["TimeOffsetNanos"][idx],
        #             bias_nanos=self.df["BiasNanos"][idx],
        #             gps_week_nanos=gps_week_nanos,
        #             tx_rx_gnss_diff_ns=tx_rx_gnss_diff_ns[idx],
        #             t_rx_ns=t_rx_ns[idx],
        #             final_calculation=final_calculation[idx],
        #         )
        #     input('enter')

        self.df["raw_pr_m"] = raw_pr_m

        # 5. (선택) 불확도 등 추가
        self.df["raw_pr_sigma_m"] = consts.C * 1e-9 * self.df["ReceivedSvTimeUncertaintyNanos"]
        self.df["clock_drift"] = consts.C * 1e-9 * self.df["ReceivedSvTimeUncertaintyNanos"]
        self.df["clock_drift_uncertainty"] = consts.C * 1e-9 * self.df["DriftUncertaintyNanosPerSecond"]


    @staticmethod
    def _row_map():
        """Map of row names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}

        """

        row_map = {
                   "utcTimeMillis" : "unix_millis",
                   "Svid" : "sv_id",
                   "Cn0DbHz" : "cn0_dbhz",
                   "AccumulatedDeltaRangeMeters" : "accumulated_delta_range_m",
                   "AccumulatedDeltaRangeUncertaintyMeters" : "accumulated_delta_range_sigma_m",
                   "ConstellationType" : "gnss_id",
                  }

        return row_map

# class AndroidRawFixes(NavData):
#     """Class handling location fix measurements from an Android log.

#     Inherits from NavData().

#     Parameters
#     ----------
#     input_path : string or path-like
#         Path to measurement csv or txt file.

#     """
#     def __init__(self, input_path):
#         pd_df = self.preprocess(input_path)
#         super().__init__(pandas_df=pd_df)


#     def preprocess(self, input_path):
#         """Read Android raw file and produce location fix dataframe objects

#         Parameters
#         ----------
#         input_path : string or path-like
#             File location of data file to read.

#         Returns
#         -------
#         fix_df : pd.DataFrame
#             Dataframe that contains the location fixes from the log.

#         """

#         if not isinstance(input_path, (str, os.PathLike)):
#             raise TypeError("input_path must be string or path-like")
#         if not os.path.exists(input_path):
#             raise FileNotFoundError(input_path,"file not found")

#         with open(input_path, encoding="utf8") as csvfile:
#             reader = csv.reader(csvfile)
#             row_idx = 0
#             skip_rows = []
#             header_row = None
#             for row in reader:
#                 if len(row) == 0:
#                     skip_rows.append(row_idx)
#                 elif len(row[0]) == 0:
#                     skip_rows.append(row_idx)
#                 elif row[0][0] == '#':
#                     if 'Fix' in row[0]:
#                         header_row = row_idx
#                     elif header_row is not None:
#                         skip_rows.append(row_idx)
#                 elif row[0] != 'Fix':
#                     skip_rows.append(row_idx)
#                 row_idx += 1

#         fix_df = pd.read_csv(input_path,
#                              skip_blank_lines = False,
#                              header = header_row,
#                              skiprows = skip_rows,
#                              )

#         return fix_df

#     def postprocess(self):
#         """Postprocess loaded data.

#         """

#         # add gps milliseconds
#         self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

#         # rename provider
#         self["fix_provider"] = np.array([self._provider_map().get(i,"")\
#                                          for i in self["fix_provider"]])

#         # add heading in radians
#         self["heading_rx_rad"] = np.deg2rad(self["heading_rx_deg"])

#     @staticmethod
#     def _row_map():
#         """Map of row names from loaded to gnss_lib_py standard

#         Returns
#         -------
#         row_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         row_map = {"LatitudeDegrees" : "lat_rx_deg",
#                    "LongitudeDegrees" : "lon_rx_deg",
#                    "AltitudeMeters" : "alt_rx_m",
#                    "Provider" : "fix_provider",
#                    "BearingDegrees" : "heading_rx_deg",
#                    "UnixTimeMillis" : "unix_millis",
#                    }
#         return row_map

#     @staticmethod
#     def _provider_map():
#         """Map to more intuitive names for fix provider.

#         Returns
#         -------
#         provider_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         provider_map = {"FLP" : "fused",
#                         "GPS" : "gnss",
#                         "NLP" : "network",
#                         }
#         return provider_map

# class AndroidRawAccel(NavData):
#     """Class handling Accelerometer measurements from Android.

#     Inherits from NavData().

#     Parameters
#     ----------
#     input_path : string or path-like
#         File location of data file to read.
#     sensor_fields : tuple
#         Names for the sensors to extract from the full log file.

#     """
#     def __init__(self, input_path,
#                  sensor_fields=("UncalAccel","Accel")):

#         self.sensor_fields = sensor_fields
#         pd_df = self.preprocess(input_path)
#         super().__init__(pandas_df=pd_df)

#     def preprocess(self, input_path):
#         """Read Android raw file and produce Accel dataframe objects.

#         Parameters
#         ----------
#         input_path : string or path-like
#             File location of data file to read.

#         Returns
#         -------
#         measurements : pd.DataFrame
#             Dataframe that contains the accel measurements from the log.

#         """

#         if not isinstance(input_path, (str, os.PathLike)):
#             raise TypeError("input_path must be string or path-like")
#         if not os.path.exists(input_path):
#             raise FileNotFoundError(input_path,"file not found")

#         sensor_data = {}

#         with open(input_path, encoding="utf8") as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 if len(row) == 0 or len(row[0]) == 0:
#                     continue
#                 if row[0][0] == '#':    # header row
#                     if len(row) == 1:
#                         continue
#                     sensor_field = row[0][2:]
#                     if sensor_field in self.sensor_fields:
#                         sensor_data[sensor_field] = [row[1:]]
#                 else:
#                     if row[0] in self.sensor_fields:
#                         sensor_data[row[0]].append(row[1:])

#         sensor_dfs = [pd.DataFrame(data[1:], columns = data[0],
#                                    dtype=np.float64) for _,data in sensor_data.items()]

#         # remove empty dataframes
#         sensor_dfs = [df for df in sensor_dfs if len(df) > 0]

#         if len(sensor_dfs) == 0:
#             measurements = pd.DataFrame()
#         elif len(sensor_dfs) > 1:
#             measurements = pd.concat(sensor_dfs, axis=0)
#         else:
#             measurements = sensor_dfs[0]

#         return measurements

#     def postprocess(self):
#         """Postprocess loaded data."""

#         # add gps milliseconds
#         self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

#     def _row_map(self):
#         """Map of row names from loaded to gnss_lib_py standard

#         Returns
#         -------
#         row_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         row_map = {
#                    'utcTimeMillis' : 'unix_millis',
#                    'AccelXMps2' : 'acc_x_mps2',
#                    'AccelYMps2' : 'acc_y_mps2',
#                    'AccelZMps2' : 'acc_z_mps2',
#                    'UncalAccelXMps2' : 'acc_x_uncal_mps2',
#                    'UncalAccelYMps2' : 'acc_y_uncal_mps2',
#                    'UncalAccelZMps2' : 'acc_z_uncal_mps2',
#                    'BiasXMps2' : 'acc_bias_x_mps2',
#                    'BiasYMps2' : 'acc_bias_y_mps2',
#                    'BiasZMps2' : 'acc_bias_z_mps2',
#                    }
#         row_map = {k:v for k,v in row_map.items() if k in self.rows}
#         return row_map

# class AndroidRawGyro(AndroidRawAccel):
#     """Class handling Gyro measurements from Android.

#     Parameters
#     ----------
#     input_path : string or path-like
#         File location of data file to read.

#     """
#     def __init__(self, input_path):
#         sensor_fields = ("UncalGyro","Gyro")
#         super().__init__(input_path, sensor_fields=sensor_fields)

#     def _row_map(self):
#         """Map of row names from loaded to gnss_lib_py standard

#         Returns
#         -------
#         row_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         row_map = {
#                    'utcTimeMillis' : 'unix_millis',
#                    'GyroXRadPerSec' : 'ang_vel_x_radps',
#                    'GyroYRadPerSec' : 'ang_vel_y_radps',
#                    'GyroZRadPerSec' : 'ang_vel_z_radps',
#                    'UncalGyroXRadPerSec' : 'ang_vel_x_uncal_radps',
#                    'UncalGyroYRadPerSec' : 'ang_vel_y_uncal_radps',
#                    'UncalGyroZRadPerSec' : 'ang_vel_z_uncal_radps',
#                    'DriftXMps2' : 'ang_vel_drift_x_radps',
#                    'DriftYMps2' : 'ang_vel_drift_y_radps',
#                    'DriftZMps2' : 'ang_vel_drift_z_radps',
#                    }
#         row_map = {k:v for k,v in row_map.items() if k in self.rows}
#         return row_map

# class AndroidRawMag(AndroidRawAccel):
#     """Class handling Magnetometer measurements from Android.

#     Parameters
#     ----------
#     input_path : string or path-like
#         File location of data file to read.

#     """
#     def __init__(self, input_path):
#         sensor_fields = ("UncalMag","Mag")
#         super().__init__(input_path, sensor_fields=sensor_fields)

#     def _row_map(self):
#         """Map of row names from loaded to gnss_lib_py standard

#         Returns
#         -------
#         row_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         row_map = {
#                    'utcTimeMillis' : 'unix_millis',
#                    'MagXMicroT' : 'mag_x_microt',
#                    'MagYMicroT' : 'mag_y_microt',
#                    'MagZMicroT' : 'mag_z_microt',
#                    'UncalMagXMicroT' : 'mag_x_uncal_microt',
#                    'UncalMagYMicroT' : 'mag_y_uncal_microt',
#                    'UncalMagZMicroT' : 'mag_z_uncal_microt',
#                    'BiasXMicroT' : 'mag_bias_x_microt',
#                    'BiasYMicroT' : 'mag_bias_y_microt',
#                    'BiasZMicroT' : 'mag_bias_z_microt',
#                    }
#         row_map = {k:v for k,v in row_map.items() if k in self.rows}
#         return row_map

# class AndroidRawOrientation(AndroidRawAccel):
#     """Class handling Orientation measurements from Android.

#     Parameters
#     ----------
#     input_path : string or path-like
#         File location of data file to read.

#     """
#     def __init__(self, input_path):
#         sensor_fields = ("OrientationDeg")
#         super().__init__(input_path, sensor_fields=sensor_fields)

#     def _row_map(self):
#         """Map of row names from loaded to gnss_lib_py standard

#         Returns
#         -------
#         row_map : Dict
#             Dictionary of the form {old_name : new_name}

#         """
#         row_map = {
#                    'utcTimeMillis' : 'unix_millis',
#                    'yawDeg' : 'yaw_rx_deg',
#                    'rollDeg' : 'roll_rx_deg',
#                    'pitchDeg' : 'pitch_rx_deg',
#                    }
#         row_map = {k:v for k,v in row_map.items() if k in self.rows}
#         return row_map
