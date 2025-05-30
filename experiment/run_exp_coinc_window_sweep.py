# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import argparse
import os
from datetime import datetime
import json

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

from analyze_data import analyze_data

"""
Usage:
`python experiment/run_exp_coinc_window_sweep.py -f [FILE_NAME] -dca [DARK_COUNTS_A] -dcb [DARK_COUNTS_B]`
where FILE_NAME is the name of the csv file (without the .csv extension)
containing Swabian data. Average dark counts per second are needed to get an accurate estimate of the source brightness.

If no dark count rates are supplied by user, the default value `DEFAULT_DARK_COUNTS` is taken.

Example:
python experiment/run_exp_coinc_window_sweep.py -f Mar_20_RUN_1_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_35pm -dca 500 -dcb 1800
"""

DEFAULT_DARK_COUNTS = (
    1000  # Set default dark count estimate in case dark counts are not supplied
)
APPROX_DELAY = 15000  # Set the approximate delay, in picoseconds
matplotlib.rcParams.update({"font.size": 12})


def run_sweep(
    coincidence_windows, simulation_results_folder_path, plot_folder_name, file_name
):
    """
    Analyze saved experimental data and find QBERs, secure key rates, raw key rates, and singles rates.
    Sweep the analysis over different coincidence windows.

    :param coincidence_windows: List of coincidence windows to sweep over (in picoseconds).
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

    for coincidence_window in coincidence_windows:
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
            simulation_results_folder_path=simulation_results_folder_path,
            plot_folder_name=plot_folder_name,
            file_name=file_name,
            coincidence_window=coincidence_window,
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


def plot_and_fit_source_brightness(
    coincidence_windows,
    raw_key_rates,
    singles_rates_A,
    singles_rates_B,
    dark_count_rate_A,
    dark_count_rate_B,
    plot_folder_name,
    file_name,
):
    """
    Plot and fit the source internal brightness from the singles rates and coincidence rates, for different
    coincidence windows. Coincidence to accidental ratio, CAR, = B * total_loss_A * total_loss_B * t_cc /
    (B * total_loss_A * t_cc * B * total_loss_B * t_cc) = C_r(t_cc) / (S_A * S_B * t_cc) = 1 / (B * t_cc).
    where B is the internal source brightness, total_loss_A is the fractional total loss on the link from the
    source to Alice, total_loss_B is the fractional total loss on the link from the source to Bob, t_cc is
    the coincidence window (in seconds), C_r(t_cc) is the coincidence rate, which is a function of t_cc,
    S_A is the singles rate on Alice, and S_B is the singles rate on Bob.

    This is also equation A1 in Neumann PRA 104, 022406 (2021).

    :param coincidence_windows: A list of coincidence windows (in seconds).
    :param raw_key_rates: A list of raw key rates for each coincidence window (in cps).
    :param singles_rates_A: A list of singles rate on Alice (in cps).
    :param singles_rates_B: A list of singles rate on Bob (in cps).
    :param dark_count_rate_A: The dark count rate on Alice (in cps).
    :param dark_count_rate_B: The dark count rate on Bob (in cps).
    :param plot_folder_name: The folder name for plots.
    :param file_name: The file name to use for the saved plot.
    """
    coincidence_windows = np.array(coincidence_windows)

    # Find C_R as 2 * raw_key_rates
    coincidence_rates = np.array(raw_key_rates) * 2

    # Find source internal brightness as S_A * S_B / C_R
    true_singles_rates_A = np.array(singles_rates_A) - (
        np.ones(len(singles_rates_A)) * dark_count_rate_A
    )
    true_singles_rates_B = np.array(singles_rates_B) - (
        np.ones(len(singles_rates_B)) * dark_count_rate_B
    )
    brightnesses = (
        true_singles_rates_A * true_singles_rates_B / np.array(coincidence_rates)
    )

    ### MANUALLY ADJUST LIMITS TO FIT LINE TO IF FIT FAILS
    lower_limit_coinc_window = 3e-9
    upper_limit_coinc_window = 20e-9

    lower_limit_index = (
        np.abs(coincidence_windows - lower_limit_coinc_window)
    ).argmin()
    upper_limit_index = (
        np.abs(coincidence_windows - upper_limit_coinc_window)
    ).argmin()

    coincidence_windows_trunc = coincidence_windows[lower_limit_index:upper_limit_index]
    brightnesses_trunc = brightnesses[lower_limit_index:upper_limit_index]

    # Define a horizontal line function
    def horizontal_line(x, A):
        return A

    # Fit the data to a horizontal line
    A_estimate = 1.56e6
    popt, pcov = curve_fit(
        horizontal_line, coincidence_windows_trunc, brightnesses_trunc, p0=[A_estimate]
    )
    A_fit = popt
    errs = np.sqrt(np.diag(pcov))
    A_err = errs

    # Plot
    fig, axs = plt.subplots(1)
    colors = plt.cm.get_cmap("Pastel1")

    p1 = axs.plot(
        coincidence_windows,
        brightnesses,
        color=colors(0),
        label="Estimated source brightness",
        linewidth=2,
    )

    fit_coinc_windows = np.linspace(
        coincidence_windows_trunc[0], coincidence_windows_trunc[-1], 500
    )
    p2 = axs.errorbar(
        fit_coinc_windows,
        np.ones(len(fit_coinc_windows)) * A_fit,
        np.ones(len(fit_coinc_windows)) * A_err,
        color=colors(1),
        linewidth=2,
        label=f"Fit line: $B = {A_fit[0]:.2e} \\pm {A_err[0]:.2e}$ cps",
    )
    plt.legend()

    plt.xscale("log")
    plt.yscale("log")

    axs.set_xlabel("Time stamp bin size (seconds)")
    axs.set_ylabel("Source brightness (counts per second)")
    plt.tight_layout()

    plt.savefig(
        f"{plot_folder_name}{file_name}_internal_source_brightness_final.png",
        dpi=300,
    )
    print(
        f"Estimated source brightness\n$S_A = {singles_rates_A[0]:.2e}$ cps \n$S_B = {singles_rates_B[0]:.2e}$ cps"
    )


def plot_qbers_and_raw_key_rates(
    x,
    raw_key_rates,
    raw_key_rate_errors,
    qbers,
    qber_errors,
    plot_folder_name,
    file_name,
):
    """
    Plot qbers and raw key rates.

    :param x: parameter to plot on the x axis.
    :param raw_key_rates: The raw key rate.
    :param raw_key_rate_errors: the raw key rate errors.
    :param qbers: The qbers.
    :param qber_errors: The qber errors.
    :param plot_folder_name: The folder name for plots.
    :param file_name: The file name to use for the saved plot.
    """
    matplotlib.rcParams.update({"font.size": 16})

    # Plot results
    fig, axs = plt.subplots(1)
    colors = plt.cm.get_cmap("Set1")

    p1 = axs.errorbar(
        np.array(x),
        np.array(raw_key_rates),
        np.array(raw_key_rate_errors),
        capsize=5,
        capthick=1,
        color=colors(0),
        label="Raw key rate",
        linestyle="",
    )

    ax2 = axs.twinx()
    p2 = ax2.errorbar(
        np.array(x),
        qbers,
        qber_errors,
        capsize=5,
        capthick=1,
        color=colors(1),
        label="QBER",
        linestyle="",
    )

    axs.yaxis.label.set_color(colors(0))
    ax2.yaxis.label.set_color(colors(1))
    axs.tick_params(axis="y", colors=colors(0))
    ax2.tick_params(axis="y", colors=colors(1))

    axs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axs.set_xlabel("Time stamp bin size (seconds)")
    axs.set_ylabel("Raw key rate (bits per second)")
    axs.set_ylim(bottom=0)
    ax2.set_ylabel("QBER")
    ax2.set_ylim(bottom=0, top=0.5)
    plt.xscale("log")
    plt.tight_layout()

    plt.savefig(
        f"{plot_folder_name}{file_name}_qber_vs_time_stamp_final.png",
        dpi=300,
    )


if __name__ == "__main__":
    """
    Run the parameter sweep and plot.
    """
    # Parse in file name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name with simulation results")
    parser.add_argument(
        "-dca",
        "--dark-count-a",
        help="Dark count rate on Alice (all detectors), in cps",
    )
    parser.add_argument(
        "-dcb", "--dark-count-b", help="Dark count rate on Bob (all detectors), in cps"
    )

    args = parser.parse_args()
    if args.dark_count_a:
        dark_count_rate_A = float(args.dark_count_a)
    else:
        dark_count_rate_A = DEFAULT_DARK_COUNTS

    if args.dark_count_b:
        dark_count_rate_B = float(args.dark_count_b)
    else:
        dark_count_rate_B = DEFAULT_DARK_COUNTS

    data_folder_name = "timetagged_data/"
    plot_folder_name = "experiment_analysis_results/"
    data_file_name = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%fZ")

    # Create folders
    if not os.path.isdir(data_folder_name):
        os.mkdir(data_folder_name)

    if not os.path.isdir(plot_folder_name):
        os.mkdir(plot_folder_name)

    # Make list of coincidence windows, in picoseconds
    coincidence_windows = np.logspace(1, 7, 150)

    # Convert to seconds
    coincidence_windows_in_seconds = [
        coincidence_window * 1e-12 for coincidence_window in coincidence_windows
    ]

    (
        qbers,
        qber_errors,
        raw_key_rates,
        raw_key_rate_errors,
        secure_key_rates,
        secure_key_rate_errors,
        singles_rates_A,
        singles_rates_B,
    ) = run_sweep(
        coincidence_windows, data_folder_name, plot_folder_name, args.file_name
    )
    # Create a dict with results
    experiment_analysis = {}
    experiment_analysis["data_type"] = "experiment"
    experiment_analysis["data_file_name"] = args.file_name
    experiment_analysis["coincidence_window"] = [
        cw * 1e-12 for cw in coincidence_windows
    ]
    experiment_analysis["secure_key_rates"] = secure_key_rates
    experiment_analysis["secure_key_rate_errors"] = secure_key_rate_errors
    experiment_analysis["raw_key_rates"] = raw_key_rates
    experiment_analysis["raw_key_rate_errors"] = raw_key_rate_errors
    experiment_analysis["qbers"] = qbers
    experiment_analysis["qber_errors"] = qber_errors

    experiment_analysis["dark_count_rate_A"] = dark_count_rate_A
    experiment_analysis["dark_count_rate_B"] = dark_count_rate_B

    experiment_analysis["qbers_label"] = "QBER"
    experiment_analysis["raw_key_rates_label"] = "Raw key rate (bps)"
    experiment_analysis["secure_key_rates_label"] = "Secure key rate (bps)"
    experiment_analysis["x_parameter_name"] = "coincidence_window"
    experiment_analysis["x_parameter_label"] = "Coincidence window (s)"

    # Save dict with parameters and sim results to JSON file
    with open(
        plot_folder_name + data_file_name + "_experimental_analysis" + ".json", "w"
    ) as file:
        file.write(json.dumps(experiment_analysis, indent=2))

    # Plot and fit the source brightness.
    plot_and_fit_source_brightness(
        coincidence_windows=coincidence_windows_in_seconds,
        raw_key_rates=raw_key_rates,
        singles_rates_A=singles_rates_A,
        singles_rates_B=singles_rates_B,
        dark_count_rate_A=dark_count_rate_A,
        dark_count_rate_B=dark_count_rate_B,
        plot_folder_name=plot_folder_name,
        file_name=args.file_name,
    )
    # Make plot and save
    plot_qbers_and_raw_key_rates(
        x=coincidence_windows_in_seconds,
        raw_key_rates=raw_key_rates,
        raw_key_rate_errors=raw_key_rate_errors,
        qbers=qbers,
        qber_errors=qber_errors,
        plot_folder_name=plot_folder_name,
        file_name=args.file_name,
    )
