# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
from simulate_bbm92 import run_key_gen

import numpy as np
import copy
from datetime import datetime
import json

import aqnsim
from aqnsim import SECOND, TERAHERTZ, SPEED_OF_LIGHT, NANOSECOND, DEFAULT_REFRACTIVE_INDEX

"""
run: dbA, dbB, pair rate, qber, jitter, est. DCA, est. DCB
run3: 11.4, 15.1, 1.35e6, 0.04, 690, 1000, 1000
run4: 11.4, 15.1, 1.34e6, 0.04, 650, 1000, 1000
run5: 11.6, 12.2, 1.25e6, 0.035, 650, 1000, 1000
run7: 11.7, 12.3, 1.28e6, 0.03, 670, 1000, 1000
run8: 11.7, 12.3, 1.32e6, 0.03, 660, 1000, 1000
[Above, likely the source rate estimates are off due to poor estimates for dark counts]

run: dbA, dbB, pair rate, qber, jitter, DCA, DCB
mar20,run1: 12.0, 12.0, 1.49e6, 0.03, 690, 500, 1800
mar20,run2: 12.0, 12.0, 1.50e6, 0.03, 690, 500, 1800
mar20,run3: 11.8, 11.9, 1.40e6, 0.03, 700, 7750, 7000
mar20,run4: 11.8, 11.9, 1.40e6, 0.03, 690, 7750, 7000
mar20,run5: 11.8, 11.9, 1.40e6, 0.03, 700, 19200, 14950
mar20,run6: 11.8, 11.9, 1.41e6, 0.03, 690, 19200, 14950
"""

# Simulation parameters
CHANNEL_LENGTH = 1  # in meters
CHANNEL_DELAY = CHANNEL_LENGTH * DEFAULT_REFRACTIVE_INDEX / SPEED_OF_LIGHT
LINK_LOSS_IN_DB_A = 12 # Loss per link for Alice, in dB
LINK_LOSS_IN_DB_B = 12 # Loss per link for Bob, in dB

# Source parameters
SOURCE_PAIR_RATE = 1.50e6 / SECOND  # Approx pairs per second from source
SOURCE_VISIBILITY = 0.94  # Visibility of the entangled source
SOURCE_WAVELENGTH = 810 * 10**-9 # Wavelength of source photons, in meters
SOURCE_BANDWIDTH_FWHM = 3 * 10**-9 # Source bandwidth FWHM, in meters

DARK_COUNT_RATES = {
    0: 50,
    1: 100,
    2: 200,
    3: 150,
    4: 200,
    5: 300,
    6: 650,
    7: 650,
}  # Average dark counts per second for each of the 8 detectors
DETECTOR_JITTER = 0.690e-9 * SECOND # Time resolution of detectors (Jitter extracted from plots)
DETECTOR_DEAD_TIME = 45e-9 * SECOND  # Dead time of detector

DETECTOR_FREQ_WIDTH = SPEED_OF_LIGHT / 700e-9 - SPEED_OF_LIGHT / 900e-9
DETECTOR_MAXIMUM_EFFICIENCY = 0.60

if __name__ == "__main__":

    # Create folder and file name
    # Set up directory to save results
    main_folder_name = "bbm92_simulation_results/"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)
    dir_path = main_folder_path + datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Get file name based on uuid and time
    file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_coinc_window"
    file_path = dir_path + "/" + file_name_string

    # In picoseconds, the coincidence windows
    coincidence_windows = np.logspace(-11, -5, 150)

    secure_key_rates = []
    raw_key_rates = []
    qbers = []
    secure_key_rate_errors = []
    raw_key_rate_errors = []
    qber_errors = []

    for coincidence_window in coincidence_windows:
        # Make dict with parameters
        simulation_parameters = {
            "channel_length": CHANNEL_LENGTH,
            "channel_delay": CHANNEL_DELAY,
            "link_loss_in_db_a": LINK_LOSS_IN_DB_A,
            "link_loss_in_db_b": LINK_LOSS_IN_DB_B,
            "source_pair_rate": SOURCE_PAIR_RATE,
            "source_visibility": SOURCE_VISIBILITY,
            "source_wavelength": SOURCE_WAVELENGTH,
            "source_bandwidth_fwhm_wavelength": SOURCE_BANDWIDTH_FWHM,
            "dark_count_rates": DARK_COUNT_RATES,
            "detector_jitter": DETECTOR_JITTER,
            "detector_dead_time": DETECTOR_DEAD_TIME,
            "detector_freq_width": DETECTOR_FREQ_WIDTH,
            "detector_maximum_efficiency": DETECTOR_MAXIMUM_EFFICIENCY,
            "minimum_time_resolution": coincidence_window,
            "num_shots": 4000000,
            "random_seed": 0,
        }
        secure_key_rate, secure_key_rate_error, raw_key_rate, raw_key_rate_error, qber, qber_error = run_key_gen(**simulation_parameters)
        print(f"coincidence window: {coincidence_window}")
        print(f"raw_key_rate: {raw_key_rate} bps")
        print(f"secure_key_rate: {secure_key_rate} bps")

        secure_key_rates.append(secure_key_rate)
        secure_key_rate_errors.append(secure_key_rate_error)
        raw_key_rates.append(raw_key_rate)
        raw_key_rate_errors.append(raw_key_rate_error)
        qbers.append(qber)
        qber_errors.append(qber_error)

    # Make copy of dict and add key_rate and qber to it
    simulation_results = copy.deepcopy(simulation_parameters)
    del simulation_results["minimum_time_resolution"]
    simulation_results["data_type"] = "aqnsim"
    simulation_results["coincidence_window"] = [cw for cw in coincidence_windows]
    simulation_results["secure_key_rates"] = secure_key_rates
    simulation_results["secure_key_rate_errors"] = secure_key_rate_errors
    simulation_results["raw_key_rates"] = raw_key_rates
    simulation_results["raw_key_rate_errors"] = raw_key_rate_errors
    simulation_results["qbers"] = qbers
    simulation_results["qber_errors"] = qber_errors

    simulation_results["qbers_label"] = "QBER"
    simulation_results["raw_key_rates_label"] = "Raw key rate (bps)"
    simulation_results["secure_key_rates_label"] = "Secure key rate (bps)"
    simulation_results["x_parameter_name"] = "coincidence_window"
    simulation_results["x_parameter_label"] = "Coincidence window (s)"

    # Save dict with parameters and sim results to JSON file
    with open(file_path + "_aqnsim_results" + ".json", "w") as file:
        file.write(json.dumps(simulation_results, indent=2))
