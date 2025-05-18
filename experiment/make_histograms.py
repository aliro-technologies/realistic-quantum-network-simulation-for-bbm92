# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import csv
import json
import argparse
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
Usage:
`python experiment/make_histograms.py -f [FILE_NAME]`
where FILE_NAME is the name of the csv file (without the .csv extension)
containing Swabian data.

Example usage:
python experiment/make_histograms.py -f RUN_5_BBM92_TIMETAG_STREAM__16HIST_UBENCHPC_12345678_DAHVDAHV_Jan_30
"""


def plot_histograms(
    simulation_results_folder_path,
    plot_folder_name,
    file_name,
    approx_delay=0,
    save_plot=False,
    same_basis_detectors=True,
):
    """
    This function analyzes data by fitting Gaussians to histograms of coincidences in the streamed Swabian data.
    This requires the user to manually input the approximate delay between detectors `approx_delay` in picoseconds,
    the (half) array size in picoseconds to look for delays over, `half_array_size_in_picoseconds`, and the
    bin size (in picoseconds) `bin_size_in_picoseconds`. From this information, accurate delays and detection resolution
    (detector jitters, etc) can be computed. Finally, a histogram is computed between all of the coincidences to find the detection
    resolution AFTER the calculated delays have been subtracted.

    :param simulation_results_folder_path: Folder path name with data.
    :param plot_folder_name: Folder path name to save plots.
    :param file_name: Data file name.
    :param approx_delay: The delay to add to Bob's detectors, in picoseconds.
        Should be 10000 for runs 3, 4 and 15000 for runs 5, 7, 8.
    :param save_plot: If true, save plots.
    :param same_basis_detectors: If True, make histograms for same basis detector pairs.
        If False, make histograms for opposite basis detector pairs.
    """

    # Load in the data; increase csv file load size to avoid errors
    max_int = 131072000
    csv.field_size_limit(max_int)

    with open(
        simulation_results_folder_path + file_name + ".csv", newline=""
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_dict = row

    # Get acquisition time and convert to seconds
    acq_time_in_seconds = float(data_dict["AcqTime"]) * 1e-12

    # In the case where the data time length does not match the recorded Swabian acq time due to
    # file size issues, find the real aquisition time.
    for j in range(1, 9):
        channel_json = data_dict[f"Ch{j}_stream"]
        timestamps = json.loads(channel_json)
        if j == 1:
            min_timestamp = timestamps[0]
            max_timestamp = timestamps[-1]
        else:
            min_timestamp = min(min_timestamp, timestamps[0])
            max_timestamp = max(max_timestamp, timestamps[-1])
    total_time_in_seconds = (max_timestamp - min_timestamp) * 1e-12
    acq_time_in_seconds = total_time_in_seconds

    if save_plot:
        # Start by plotting histograms between each set of detectors and finding delays
        num_rows = 2
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()

    # Histogram pairs
    if same_basis_detectors:
        hist_pairs = [(4, 8), (4, 7), (3, 7), (3, 8), (2, 5), (2, 6), (1, 6), (1, 5)]
    else:
        hist_pairs = [(1, 7), (2, 8), (3, 6), (4, 5), (1, 8), (2, 7), (3, 5), (4, 6)]
    fitted_delay = {}

    # Define a Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    detector_st_devs = []

    for i, hist_pair in enumerate(hist_pairs):
        half_array_size_in_picoseconds = 6000
        bin_size_in_picoseconds = 30

        channel1_json = data_dict[f"Ch{hist_pair[0]}_stream"]
        channel2_json = data_dict[f"Ch{hist_pair[1]}_stream"]
        d1_timestamps = json.loads(channel1_json)
        d2_timestamps = json.loads(channel2_json)

        p1, p2 = 0, 0
        d1_common_basis_choice_times = []
        d2_common_basis_choice_times = []

        while p1 < len(d1_timestamps) and p2 < len(d2_timestamps):
            # If the basis choices match and the timestamps are close enough, add to the common_basis_choice_times list
            adjusted_d2_timestamp = d2_timestamps[p2] + approx_delay
            timestamp_difference = d1_timestamps[p1] - adjusted_d2_timestamp

            # Compute histogram; cutoff at approx_delay plus coincidence_window
            if abs(timestamp_difference) < half_array_size_in_picoseconds:
                d1_common_basis_choice_times.append(d1_timestamps[p1])
                d2_common_basis_choice_times.append(adjusted_d2_timestamp)
                p1 += 1
                p2 += 1
            elif timestamp_difference < 0:
                p1 += 1
            else:
                p2 += 1

        # Find differences between coincidence timesteps
        timestamp_differences = np.array(d1_common_basis_choice_times) - np.array(
            d2_common_basis_choice_times
        )

        delay_data = np.arange(
            -int(half_array_size_in_picoseconds),
            int(half_array_size_in_picoseconds) + bin_size_in_picoseconds,
            bin_size_in_picoseconds,
        )
        half_array_length = int(
            half_array_size_in_picoseconds / bin_size_in_picoseconds
        )
        hist_data = np.zeros(2 * half_array_length + 1)

        # Turn list of time differences into an ordered list of delays (x) and a list of frequency of each delay (y)
        for timestamp_diff in timestamp_differences:
            # Round to 10 ps
            rounded_timestamp_diff = (
                round(timestamp_diff / bin_size_in_picoseconds)
                * bin_size_in_picoseconds
            )
            rounded_timestamp_diff_index = int(
                rounded_timestamp_diff
                * half_array_length
                / half_array_size_in_picoseconds
            )
            rounded_timestamp_diff_index += int(
                half_array_size_in_picoseconds / bin_size_in_picoseconds
            )
            hist_data[rounded_timestamp_diff_index] += 1

        # Extract the delays between pairs of detectors
        # Fit the data to a Gaussian curve
        A_estimate = 10
        mu_estimate = 1000
        sigma_estimate = 1000
        popt, pcov = curve_fit(
            gaussian,
            delay_data,
            hist_data,
            p0=[A_estimate, mu_estimate, sigma_estimate],
        )
        A_fit, mu_fit, sigma_fit = popt
        detector_st_devs.append(sigma_fit)

        fitted_delay[hist_pair] = mu_fit + approx_delay

        errs = np.sqrt(np.diag(pcov))
        A_err, mu_err, sigma_err = errs

        if A_err > np.sqrt(A_fit):
            raise ValueError("Fit is bad; modify p0 parameters")

        fine_grained_delay_data = np.linspace(delay_data[0], delay_data[-1], 1000)

        if save_plot:
            fig.suptitle(
                f"Acquisition time: {acq_time_in_seconds} seconds; Added delay: {approx_delay} ps"
            )
            ax = axes[i]
            ax.hist(delay_data[:-1], delay_data, weights=hist_data[:-1])
            ax.set_title(f"Detector pair {hist_pair}")
            ax.set_ylabel("Coinc. counts")
            ax.set_xlabel("Picoseconds")

            ax.grid(True)
            ax.plot(
                fine_grained_delay_data,
                gaussian(fine_grained_delay_data, A_fit, mu_fit, sigma_fit),
                label=f"$\\mu={mu_fit:.0f} \\pm {mu_err:.0f}$\n"
                f"$\\sigma={sigma_fit:.0f} \\pm {sigma_err:.0f}$",
            )
            ax.legend(prop={"size": 8})

    print(
        f"Avg standard deviation (detector RMS jitter): {np.mean(detector_st_devs)} ps"
    )

    if not same_basis_detectors:
        file_name += "_not_same_basis"
    if save_plot:
        plt.tight_layout()
        plt.savefig(plot_folder_name + file_name + "_histograms.png", dpi=300)


if __name__ == "__main__":
    """
    Run the analysis.
    """
    # Parse in file name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name with simulation results")
    args = parser.parse_args()

    data_folder_name = "timetagged_data/"
    plot_folder_name = "experiment_analysis_results/"

    # Create folders
    if not os.path.isdir(data_folder_name):
        os.mkdir(data_folder_name)

    if not os.path.isdir(plot_folder_name):
        os.mkdir(plot_folder_name)

    plot_histograms(
        simulation_results_folder_path=data_folder_name,
        plot_folder_name=plot_folder_name,
        file_name=args.file_name,
        approx_delay=15000,
        save_plot=True,
        same_basis_detectors=True,
    )
