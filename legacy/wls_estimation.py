import argparse
import gnss_lib_py as glp
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path
from plotly.subplots import make_subplots
from dash import Dash, html, dcc

SIGNAL_TYPE_MAP = {
    'gps' : 'l1',
    'beidou' : 'b1',
    'galileo' : 'l1'
}

def main(input_path, gnss_id, output_pickle):
    # Load raw GNSS data
    raw_data = glp.AndroidRawGnss(input_path=input_path, filter_measurements=False, verbose=True)
    
    # Compute full states
    full_states = glp.add_sv_states(raw_data, source="precise", verbose=False)
    
    # Corrected pseudorange calculation
    full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states['b_sv_m']
    
    # Apply satellite selection filter
    full_states = full_states.where("gnss_id", (gnss_id))
    
    # Solve using WLS
    wls_estimate = glp.solve_gnss_ekf(full_states)
    
    # Generate map plot
    raw_fig = glp.plot_map(wls_estimate)
    
    output_path = Path(output_pickle) / f'{gnss_id}.pkl'

    # Save figure object as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(raw_fig, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS Satellite Selection and Processing")
    parser.add_argument("--input_path", type=str, default='./data/raw/gnss_log_2024_12_12_12_43_03.txt', help="Path to GNSS log file")
    parser.add_argument("--gnss_id", type=str, default="gps", help="GNSS ID (e.g., gps, glonass, galileo, beidou)")
    parser.add_argument("--output_pickle_path", type=str, default="./data/results/", help="Output file name for the pickle object")
    
    args = parser.parse_args()
    
    main(args.input_path, args.gnss_id, args.output_pickle_path)
