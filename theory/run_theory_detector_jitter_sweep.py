import os
from datetime import datetime
import copy
import json
import numpy as np
from calc_rate_and_qber import calc_rate_and_qber

FRACTIONAL_LOSS_A = 10 ** (-12 / 10)
FRACTIONAL_LOSS_B = 10 ** (-12 / 10)
DETECTOR_EFFICIENCY = 0.6
DETECTOR_DEAD_TIME = 45e-9
RATE_SOURCE = 1.5e6
DARK_COUNT_RATE_A = 500  # Total per communication partner
DARK_COUNT_RATE_B = 1800  # Total per communication partner
NUM_DETECTORS = 4
DETECTION_RES_ST_DEV = 0.690e-9
DETECTION_RESOLUTION_FWHM = DETECTION_RES_ST_DEV * 2 * np.sqrt(2 * np.log(2))
COINCIDENCE_WINDOW = 2e-9
P_OPTICAL_ERROR = 0.03


if __name__ == "__main__":
    # Create folder and file name
    # Set up directory to save results
    main_folder_name = "theory_results/"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)

    dir_path = main_folder_name + datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Get file name based on time
    file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_jitter"
    file_path = dir_path + "/" + file_name_string

    detector_jitters = np.linspace(0.1e-9, 10e-9, 1000)
    detector_jitter_fwhms = [dj * 2 * np.sqrt(2 * np.log(2)) for dj in detector_jitters]

    secure_key_rates = []
    raw_key_rates = []
    qbers = []

    for jitter_fwhm in detector_jitter_fwhms:
        # Make dict with parameters
        simulation_parameters = {
            "fractional_loss_A": FRACTIONAL_LOSS_A,
            "fractional_loss_B": FRACTIONAL_LOSS_B,
            "detector_efficiency": DETECTOR_EFFICIENCY,
            "detector_dead_time": DETECTOR_DEAD_TIME,
            "coincidence_window": COINCIDENCE_WINDOW,
            "rate_source": RATE_SOURCE,
            "dark_count_rate_A": DARK_COUNT_RATE_A,
            "dark_count_rate_B": DARK_COUNT_RATE_B,
            "probability_optical_error": P_OPTICAL_ERROR,
            "num_detectors": NUM_DETECTORS,
            "detection_resolution": jitter_fwhm,
        }

        raw_key_rate, secure_key_rate, qber = calc_rate_and_qber(**simulation_parameters)
        secure_key_rates.append(secure_key_rate)
        raw_key_rates.append(raw_key_rate)
        qbers.append(qber)

    # Make copy of dict and add key_rate and qber to it
    simulation_results = copy.deepcopy(simulation_parameters)
    del simulation_results["detection_resolution"]
    simulation_results["data_type"] = "numerics"
    simulation_results["detection_resolution"] = list(detector_jitters)
    simulation_results["secure_key_rates"] = secure_key_rates
    simulation_results["raw_key_rates"] = raw_key_rates
    simulation_results["qbers"] = qbers

    simulation_results["qbers_label"] = "QBER"
    simulation_results["raw_key_rates_label"] = "Raw key rate (bps)"
    simulation_results["secure_key_rates_label"] = "Secure key rate (bps)"
    simulation_results["x_parameter_name"] = "detection_resolution"
    simulation_results["x_parameter_label"] = "Detection resolution (s)"

    # Save dict with parameters and sim results to JSON file
    with open(file_path + "_sim_results" + ".json", "w") as file:
        file.write(json.dumps(simulation_results, indent=2))
