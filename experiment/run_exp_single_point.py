import argparse
import os
import json
import datetime
from analyze_data import analyze_data

"""
To run:
`python experiment/run_exp_single_point.py -f [FILE_NAME] -pn [PARAMETER_NAME] -pv [PARAMETER_VALUE]`
where FILE_NAME is the name of the csv file (without the .csv extension)
containing Swabian data.

Example:
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn dark_count_rate -pv 1150

"""

APPROX_DELAY = 15000  # Set the approximate delay, in picoseconds


def run_analyze_data(
    coincidence_window, simulation_results_folder_path, plot_folder_name, file_name
):
    """
    Process experimental data and return calculated QBER, raw key rate, secure key rate,
    and singles rates.

    :param coincidence_window: The coincidence window to find correlated counts, in picoseconds.
    :param simulation_results_folder_path: Simulation results folder path.
    :param plot_folder_name: Plot folder path.
    :param file_name: data file path.
    """
    qbers = []
    qber_errors = []
    raw_key_rates = []
    raw_key_rate_errors = []
    secure_key_rates = []
    secure_key_rate_errors = []
    singles_rates_A = []
    singles_rates_B = []

    (
        qber,
        qber_error,
        raw_key_rate,
        raw_key_rate_error,
        secure_key_rate,
        secure_key_rate_error,
        singles_rate_A,
        singles_rate_B,
    ) = analyze_data(
        simulation_results_folder_path,
        plot_folder_name,
        file_name,
        coincidence_window,
        approx_delay=APPROX_DELAY,
    )
    qbers.append(qber)
    qber_errors.append(qber_error)
    raw_key_rates.append(raw_key_rate)
    raw_key_rate_errors.append(raw_key_rate_error)
    secure_key_rates.append(secure_key_rate)
    secure_key_rate_errors.append(secure_key_rate_error)
    singles_rates_A.append(singles_rate_A)
    singles_rates_B.append(singles_rate_B)

    return (
        qbers,
        qber_errors,
        raw_key_rates,
        raw_key_rate_errors,
        secure_key_rates,
        secure_key_rate_errors,
        singles_rates_A,
        singles_rates_B,
    )


if __name__ == "__main__":
    """
    Run the parameter sweep and plot.
    """
    # Parse in file name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name with simulation results")
    parser.add_argument(
        "-pn", "--param-name", help="Name of parameter to manually input"
    )
    parser.add_argument(
        "-pv", "--param-value", help="Value(s) of parameter manually input"
    )

    args = parser.parse_args()
    param_value = float(args.param_value)
    param_name = args.param_name

    data_folder_name = "timetagged_data/"
    plot_folder_name = "experiment_analysis_results/"
    data_file_name = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%fZ")

    # Create folders
    if not os.path.isdir(data_folder_name):
        os.mkdir(data_folder_name)

    if not os.path.isdir(plot_folder_name):
        os.mkdir(plot_folder_name)

    # Two nanoseconds (the coincidence window), in picoseconds
    coincidence_window = 2e-9 * 1e12

    (
        qbers,
        qber_errors,
        raw_key_rates,
        raw_key_rate_errors,
        secure_key_rates,
        secure_key_rate_errors,
        singles_rates_A,
        singles_rates_B,
    ) = run_analyze_data(
        coincidence_window, data_folder_name, plot_folder_name, args.file_name
    )
    # Create a dict with results
    experiment_analysis = {}
    experiment_analysis["data_type"] = "experiment"
    experiment_analysis["data_file_name"] = args.file_name
    experiment_analysis["coincidence_window"] = coincidence_window
    experiment_analysis["secure_key_rates"] = secure_key_rates
    experiment_analysis["secure_key_rate_errors"] = secure_key_rate_errors
    experiment_analysis["raw_key_rates"] = raw_key_rates
    experiment_analysis["raw_key_rate_errors"] = raw_key_rate_errors
    experiment_analysis["qbers"] = qbers
    experiment_analysis["qber_errors"] = qber_errors

    experiment_analysis["qbers_label"] = "QBER"
    experiment_analysis["raw_key_rates_label"] = "Raw key rate (bps)"
    experiment_analysis["secure_key_rates_label"] = "Secure key rate (bps)"
    experiment_analysis["x_parameter_name"] = param_name
    experiment_analysis[param_name] = [param_value]

    # Save dict with parameters and sim results to JSON file
    with open(
        plot_folder_name + data_file_name + "_manual_input_analysis" + ".json", "w"
    ) as file:
        file.write(json.dumps(experiment_analysis, indent=2))
