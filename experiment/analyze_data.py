# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import json
import argparse
import os
import re
import csv

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from get_secure_key_rate_error import get_secure_key_rate_error

"""
Usage:
`python experiment/analyze_data.py -f [FILE_NAME]`
where FILE_NAME is the name of the csv file (without the .csv extension)
containing Swabian data.

Example usage:
python experiment/analyze_data.py -f RUN_5_BBM92_TIMETAG_STREAM__16HIST_UBENCHPC_12345678_DAHVDAHV_Jan_30
"""


def analyze_data(
    simulation_results_folder_path,
    plot_folder_name,
    file_name,
    coincidence_window,
    approx_delay=0,
    save_plot=False,
):
    """
    This function analyzes data by fitting Gaussians to histograms of coincidences in the streamed Swabian data.
    This requires the user to manually input the approximate delay between detectors `approx_delay` in picoseconds,
    the (half) array size in picoseconds to look for delays over, `half_array_size_in_picoseconds`, and the
    bin size (in picoseconds) `bin_size_in_picoseconds`. From this information, accurate delays and detection resolution
    (detector jitters, etc) can be computed. Next, coincidences between detector pairs are found from the streamed Swabian files
    using a two-pointer method to compare the two arrays (a more optimal method than rounding) and qbers,
    raw_key_rates, and secure_key_rates are calculated.

    :param simulation_results_folder_path: Folder path name which contains saved data.
    :param plot_folder_name: Folder path name to save plots.
    :param file_name: Data file name.
    :param coincidence_window: The time bin size, in picoseconds.
    :param approx_delay: The delay to add to Bob's detectors, in picoseconds.
    :param save_plot: If true, save the plot.
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

    # Find the detector ordering in the saved title
    match = re.search(r"12345678_([^_]{8})_", file_name)
    if match:
        detector_map_string = match.group(1)
        if detector_map_string == "HVDAHVDA":
            alice_detector_to_bit = {1: 1, 2: 0, 3: 0, 4: 1}
            bob_detector_to_bit = {5: 0, 6: 1, 7: 0, 8: 1}
            basis0 = "HV"
            basis1 = "AD"
        elif detector_map_string == "DAHVDAHV":
            alice_detector_to_bit = {1: 0, 2: 1, 3: 1, 4: 0}
            bob_detector_to_bit = {5: 0, 6: 1, 7: 0, 8: 1}
            basis0 = "AD"
            basis1 = "HV"
        else:
            raise ValueError(
                "Invalid title string does not contain basis information in correct format."
            )
    else:
        raise ValueError("Invalid title string does not contain basis information.")

    if save_plot:
        # Start by plotting histograms between each set of detectors and finding delays
        num_rows = 2
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()

    # Histogram pairs
    # Match "wrong" basis pairs to get histograms with more counts, plus (4, 7) and (4, 8)
    hist_pairs = [
        (1, 7),
        (2, 8),
        (3, 6),
        (4, 5),
        (4, 7),
        (2, 7),
        (3, 5),
        (4, 6),
        (4, 8),
    ]

    fitted_delay = {}

    # Define a Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    detector_st_devs = []

    # Find the acquisition time
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
            if rounded_timestamp_diff_index > 0:
                rounded_timestamp_diff_index += int(
                    half_array_size_in_picoseconds / bin_size_in_picoseconds
                )
            else:
                rounded_timestamp_diff_index = (
                    int(half_array_size_in_picoseconds / bin_size_in_picoseconds)
                    + rounded_timestamp_diff_index
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

        if save_plot and i < 8:
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

    if save_plot:
        plt.tight_layout()
        plt.savefig(plot_folder_name + file_name + "_histograms_for_sync.png", dpi=300)

    # Map detector number to relevant detector pair for finding delay
    delay1 = 0
    delay7 = fitted_delay[(1, 7)]
    delay2 = delay7 - fitted_delay[(2, 7)]
    delay8 = fitted_delay[(2, 8)] + delay2
    if basis0 == "AD":
        delay4 = delay7 - fitted_delay[(4, 7)]
    else:
        delay4 = delay8 - fitted_delay[(4, 8)]
    delay5 = fitted_delay[(4, 5)] + delay4
    delay6 = fitted_delay[(4, 6)] + delay4
    delay3 = delay6 - fitted_delay[(3, 6)]

    detector_to_delay = {
        1: delay1,
        2: delay2,
        3: delay3,
        4: delay4,
        5: delay5,
        6: delay6,
        7: delay7,
        8: delay8,
    }

    # Alice's streamed data
    alice_basis_data = {}
    alice_channel_bit_data = {}
    for detector in range(1, 5):
        channel_json = data_dict[f"Ch{detector}_stream"]

        # Round time stamps
        channel_data = json.loads(channel_json)
        delay = detector_to_delay[detector]
        channel_data = [timestamp + delay for timestamp in channel_data]

        if detector in [1, 2]:
            alice_basis_data = alice_basis_data | {
                timestamp: basis0 for timestamp in channel_data
            }
        else:
            alice_basis_data = alice_basis_data | {
                timestamp: basis1 for timestamp in channel_data
            }

        alice_channel_bit_data = alice_channel_bit_data | {
            timestamp: alice_detector_to_bit[detector] for timestamp in channel_data
        }

    bob_basis_data = {}
    bob_channel_bit_data = {}
    for detector in range(5, 9):
        channel_json = data_dict[f"Ch{detector}_stream"]

        # Add delay
        channel_data = json.loads(channel_json)
        delay = detector_to_delay[detector]
        channel_data = [timestamp + delay for timestamp in channel_data]
        if detector in [5, 6]:
            bob_basis_data = bob_basis_data | {
                timestamp: basis0 for timestamp in channel_data
            }
        else:
            bob_basis_data = bob_basis_data | {
                timestamp: basis1 for timestamp in channel_data
            }

        bob_channel_bit_data = bob_channel_bit_data | {
            timestamp: bob_detector_to_bit[detector] for timestamp in channel_data
        }

    # Alice and Bob exchange time stamp data by basis (without sharing actual measurement results in that basis);
    # they only keep timestamps which both share between each of them (get rid of singles and only keep correlated pairs)
    p1, p2 = 0, 0
    alice_common_basis_choice_times = []
    bob_common_basis_choice_times = []
    alice_timestamps = sorted(list(alice_basis_data.keys()))
    bob_timestamps = sorted(list(bob_basis_data.keys()))
    last_alice_coinc_time = -coincidence_window
    last_bob_coinc_time = -coincidence_window

    while p1 < len(alice_timestamps) and p2 < len(bob_timestamps):
        # If the basis choices match and the timestamps are close enough, add to the common_basis_choice_times list
        timestamp_difference = alice_timestamps[p1] - bob_timestamps[p2]

        # Line below allows + or - t_cc/2 (because of the abs), which means the total allowed t_cc will be `coincidence_window`.
        if abs(timestamp_difference) < coincidence_window / 2:
            if (
                alice_basis_data[alice_timestamps[p1]]
                == bob_basis_data[bob_timestamps[p2]]
            ):
                last_coinc_diff_alice = alice_timestamps[p1] - last_alice_coinc_time
                last_coinc_diff_bob = bob_timestamps[p2] - last_bob_coinc_time

                # Don't collect multiple coincidences within the same coincidence window
                if (
                    abs(last_coinc_diff_alice) > coincidence_window
                    and abs(last_coinc_diff_bob) > coincidence_window
                ):
                    alice_common_basis_choice_times.append(alice_timestamps[p1])
                    bob_common_basis_choice_times.append(bob_timestamps[p2])
                    last_alice_coinc_time = alice_timestamps[p1]
                    last_bob_coinc_time = bob_timestamps[p2]
            p1 += 1
            p2 += 1
        elif timestamp_difference < 0:
            p1 += 1
        else:
            p2 += 1

    # Total singles counts
    alice_singles_counts_rate = len(alice_timestamps) / acq_time_in_seconds
    print(f"Alice singles count rate: {alice_singles_counts_rate}")
    bob_singles_counts_rate = len(bob_timestamps) / acq_time_in_seconds
    print(f"Bob singles count rate: {bob_singles_counts_rate}")
    # Alice and Bob each distill their keys
    alice_secret_key = [
        alice_channel_bit_data[timestamp]
        for timestamp in alice_common_basis_choice_times
    ]
    bob_secret_key = [
        bob_channel_bit_data[timestamp] for timestamp in bob_common_basis_choice_times
    ]
    alice_basis_info = [
        alice_basis_data[timestamp] for timestamp in alice_common_basis_choice_times
    ]
    bob_basis_info = [
        bob_basis_data[timestamp] for timestamp in bob_common_basis_choice_times
    ]

    # Compare the keys to get QBER
    assert len(alice_common_basis_choice_times) == len(bob_common_basis_choice_times)
    key_len = len(alice_common_basis_choice_times)

    raw_key_rate = key_len / acq_time_in_seconds
    raw_key_rate_error = np.sqrt(key_len) / acq_time_in_seconds

    if key_len > 0:
        qber = 1 - (
            sum(
                1 if alice_secret_key[i] == bob_secret_key[i] else 0
                for i in range(key_len)
            )
            / key_len
        )
        HV_err_count = sum(
            1
            if alice_secret_key[i] != bob_secret_key[i] and alice_basis_info[i] == "HV"
            else 0
            for i in range(key_len)
        )
        AD_err_count = sum(
            1
            if alice_secret_key[i] != bob_secret_key[i] and alice_basis_info[i] == "AD"
            else 0
            for i in range(key_len)
        )
        # Find error as standard deviation of a binomial distribution divided by
        # the square root of the number of measurements
        qber_error = np.sqrt(qber * (1 - qber)) / np.sqrt(key_len)

        # Some of the key is error and some is used to correct the error
        binary_entropy = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

        secure_key_rate = max(raw_key_rate * (1 - 2.1 * binary_entropy), 0)
        secure_key_rate_error = get_secure_key_rate_error(
            raw_key_rate, raw_key_rate_error, qber, qber_error
        )

    else:
        qber = 0
        HV_err_count = 0
        AD_err_count = 0
        qber_error = 0
        secure_key_rate = 0
        secure_key_rate_error = 0

    print("Errors from HV basis: " + str(HV_err_count))
    print("Errors from AD basis: " + str(AD_err_count))
    print(f"QBER: {qber} +/- {qber_error}")

    print(f"Raw key rate (bits per second): {raw_key_rate} +/- {raw_key_rate_error}")
    print(
        f"Secure key rate (bits per second): {secure_key_rate} +/- {secure_key_rate_error}"
    )

    # Return QBER, raw key rate
    return (
        qber,
        qber_error,
        raw_key_rate,
        raw_key_rate_error,
        secure_key_rate,
        secure_key_rate_error,
        alice_singles_counts_rate,
        bob_singles_counts_rate,
    )


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

    # Approx delay of 10000 ps for run3, run4 and 15000 ps for run5, run7, run8
    analyze_data(
        simulation_results_folder_path=data_folder_name,
        plot_folder_name=plot_folder_name,
        file_name=args.file_name,
        coincidence_window=1000,
        approx_delay=15000,
        save_plot=True,
    )
