

# import warnings

# import numpy as np
# from gnss_lib_py.parsers.sp3 import Sp3
# from gnss_lib_py.parsers.clk import Clk
# from gnss_lib_py.navdata.navdata import NavData
# from gnss_lib_py.utils.ephemeris_downloader import DEFAULT_EPHEM_PATH, load_ephemeris
# from gnss_lib_py.utils.sv_models import single_gnss_from_precise_eph

# import gnss_lib_py as glp

# import pandas as pd

# TARGET_GPS_TIME = 1.431251328865765e+09  # Example GPS time in milliseconds
# TARGET_GPS_TIME_MILLIS = TARGET_GPS_TIME * 1000  # Convert to microseconds
# TARGET_CONSTELLATION = 'beidou'  # Example constellation (GPS)
# SV_ID = 6

# df = pd.DataFrame(data = {'gps_millis_tx': [TARGET_GPS_TIME_MILLIS]
#                           , 'gnss_id': [TARGET_CONSTELLATION]
#                             , 'sv_id': [SV_ID]
#                           })

# navdata = NavData(pandas_df=df)

# unique_gps_millis = TARGET_GPS_TIME * 1000  # Convert to microseconds
# navdata = glp.add_sv_states(navdata)
# print(navdata)


from scipy.io import loadmat

# mat 파일 불러오기
data = loadmat('/home/user/mskim/gnss/data/challenge/correction_20230906_1847_pixel6pro.mat')
# 변수 목록 보기
print(data.keys())

# 특정 변수 접근
x = data['corr_us']
print(x.shape)
print(data['None'].shape)
y = data['y']