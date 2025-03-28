# Construct a linear trajectory starting from the Durand building and
# moving North at 10 m/s for 10 steps
# This trajectory will not have any major differences in calculated
# satellite positions but is to demonstrate the functionality

import gnss_lib_py as glp
import plotly.express as px
import pandas as pd
import numpy as np
# import plotly.graph_objects as go

from plotly.subplots import make_subplots
from datetime import datetime, timezone
# Send time at which SV states are needed in GPS millis
start_time = datetime(year=2021,
                       month=4,
                       day=29,
                       hour=22,
                       minute=30,
                       second=0)
start_time = start_time.replace(tzinfo=timezone.utc)
start_gps_millis = glp.datetime_to_gps_millis(start_time)
print(start_gps_millis, type(start_gps_millis))

# Create sequence of times
gps_millis_traj = start_gps_millis + 1000.*np.arange(10)

print(gps_millis_traj)

# Define receiver position in ECEF
rx_LLA_durand = np.reshape([37.427112, -122.1764146, 16], [3, 1])
rx_ecef_durand = np.reshape(glp.geodetic_to_ecef(rx_LLA_durand), [3, 1])

# Create sequence of moving receiver (using approximation of long to meters)
rx_LLA_traj = rx_LLA_durand + np.vstack((np.zeros(10),
                                         0.0001*10.*np.arange(10),
                                         np.zeros(10)))

# Convert trajectory to ECEF
rx_ecef_traj = glp.geodetic_to_ecef(rx_LLA_traj)

# Create state estimate with given trajectory
state_traj = glp.NavData()
state_traj['gps_millis'] = gps_millis_traj
state_traj['x_rx_m'] = rx_ecef_traj[0,:]
state_traj['y_rx_m'] = rx_ecef_traj[1,:]
state_traj['z_rx_m'] = rx_ecef_traj[2,:]

# Define all GPS satellites, so that all broadcast ephemeris parameters
# are downloaded
Constellation = {
                 'C' : 46,
                #  'G' : 32,
                #  'R' : 60,
                #  'E' : 60,
                #  'J' : 10
                 }

all_sats = [f"{each_code}{sv:02d}" for each_code, max_num in Constellation.items() for sv in range(1, max_num + 1)]

# Download ephemeris files for given time
ephem_all_sats = glp.get_time_cropped_rinex(start_gps_millis, all_sats, ephemeris_directory="ephemeris")
print(ephem_all_sats.pandas_df().columns.tolist())
print(ephem_all_sats.pandas_df().iloc[10])

sv_states_tx = glp.find_sv_states(start_gps_millis, ephem_all_sats)

print(sv_states_tx)

# # Option 2: Estimate SV states for given reception time (factors and removes
# # approximately the time taken by the signal to reach the receiver)
# # This method requires an estimate of the receiver's position and also
# # gives difference between positions and the estimated true range
# sv_states_rx, del_pos, true_range = glp.find_sv_location(start_gps_millis, rx_ecef_durand, ephem_all_sats)

# print(sv_states_tx['gnss_id'])
# print(sv_states_rx)
# print('Difference between x positions estimated for Tx and Rx times \n',
#       sv_states_tx['x_sv_m'] - sv_states_rx['x_sv_m'])
# print('Difference between x velocities estimated for Tx and Rx times\n',
#       sv_states_tx['vx_sv_mps'] - sv_states_rx['vx_sv_mps'])
# print(sv_states_tx['b_sv_m'])


# sv_posvel_traj = glp.add_visible_svs_for_trajectory(state_traj,
#                                                     ephemeris_path="ephemeris")

# sv_posvel_traj_sv25 = sv_posvel_traj.where("sv_id", 25)

# print(sv_posvel_traj_sv25)

# print('GPS milliseconds with first time subtracted\n',
#         sv_posvel_traj_sv25['gps_millis'] - start_gps_millis)

# print('Changing x ECEF SV positions\n',
#         sv_posvel_traj_sv25['x_sv_m'] - sv_posvel_traj_sv25['x_sv_m', 0])

# print('Consecutive change in x ECEF positions\n',
#         sv_posvel_traj_sv25['x_sv_m', 1:] - sv_posvel_traj_sv25['x_sv_m', :-1])

# print('Velocity along x ECEF for reference\n',
#         sv_posvel_traj_sv25['vx_sv_mps'])