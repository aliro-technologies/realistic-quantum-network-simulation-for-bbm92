import numpy as np
import csv
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
import argparse
import os
import re

from get_secure_key_rate_error import get_secure_key_rate_error

"""
To run: 
`python experiment/make_correlation_matrix.py -f [FILE_NAME]`
where FILE_NAME is the name of the csv file (without the .csv extension)
containing Swabian data.

Example usage:
python experiment/make_correlation_matrix.py -f RUN_5_BBM92_TIMETAG_STREAM__16HIST_UBENCHPC_12345678_DAHVDAHV_Jan_30
"""
FINAL_PLOT = True

def make_correlation_matrix(
    simulation_results_folder_path,
    plot_folder_name,
    file_name,
    coincidence_window,
    approx_delay=0,
    save_plot=False,
):
    """
    This function plots a correlation matrix for the probabilities of measurement outcomes
    for each basis (X, Z). 

    :param simulation_results_folder_path: Folder path name with data.
    :param plot_folder_name: Folder path name to save plots.
    :param file_name: Data file name.
    :param coincidence_window: The time bin size, in picoseconds. Warning: do not exceed 1000 ps.
    :param approx_delay: The delay to add to Bob's detectors, in picoseconds.
        Should be 10000 for runs 3, 4 and 15000 for runs 5, 7, 8.
    :param save_plot: If true, save plots.
    """

    # Load in the data; increase csv file load size to avoid errors
    max_int = 131072000
    csv.field_size_limit(max_int)

    with open(simulation_results_folder_path + file_name + ".csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_dict = row

    # Get acquisition time and convert to seconds
    acq_time_in_seconds = float(data_dict["AcqTime"]) * 1e-12

    # Find the detector ordering in the saved title
    match = re.search(r"12345678_([^_]{8})_", file_name)
    if match:
        detector_map_string = match.group(1)
        if detector_map_string == "HVDAHVDA":
            alice_detector_to_bit = {1:0, 2:1, 3:0, 4:1}
            bob_detector_to_bit = {5:0, 6:1, 7:0, 8:1}
            basis0 = "HV"
            basis1 = "AD"
        elif detector_map_string == "DAHVDAHV":
            alice_detector_to_bit = {1:0, 2:1, 3:0, 4:1}
            bob_detector_to_bit = {5:0, 6:1, 7:0, 8:1}
            basis0 = "AD"
            basis1 = "HV"
        else:
            raise ValueError("Invalid title string does not contain basis information in correct format.")
    else:
        raise ValueError("Invalid title string does not contain basis information.")

    if save_plot:
        # Start by plotting histograms between each set of detectors and finding delays
        num_rows = 2
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = axes.flatten()

    # Histogram pairs
    # Match wrong basis pairs to get histograms with more counts, plus (4, 7) and (4, 8)
    hist_pairs = [(1, 7), (2, 8), (3, 6), (4, 5), (4, 7), (2, 7), (3, 5), (4, 6), (4, 8)]

    fitted_delay = {}

    # Define a Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    detector_st_devs = []

    # In the case where the data time length does not match the recorded Swabian acq time due to
    # file size issues, find the real aquisition time.
    for j in range(1, 9):
        channel_json = data_dict[f'Ch{j}_stream']
        timestamps = json.loads(channel_json)
        if j == 1:
            min_timestamp = timestamps[0]
            max_timestamp = timestamps[-1]
        else:
            min_timestamp = min(min_timestamp, timestamps[0])
            max_timestamp = max(max_timestamp, timestamps[-1])
    total_time_in_seconds = (max_timestamp - min_timestamp)*1e-12
    acq_time_in_seconds = total_time_in_seconds


    for i, hist_pair in enumerate(hist_pairs):
        half_array_size_in_picoseconds = 6000
        bin_size_in_picoseconds = 30

        channel1_json = data_dict[f'Ch{hist_pair[0]}_stream']
        channel2_json = data_dict[f'Ch{hist_pair[1]}_stream']
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
        timestamp_differences = np.array(d1_common_basis_choice_times) - np.array(d2_common_basis_choice_times)

        delay_data = np.arange(-int(half_array_size_in_picoseconds), int(half_array_size_in_picoseconds)+bin_size_in_picoseconds, bin_size_in_picoseconds)
        half_array_length = int(half_array_size_in_picoseconds/bin_size_in_picoseconds)
        hist_data = np.zeros(2*half_array_length+1)

        # Turn list of time differences into an ordered list of delays (x) and a list of frequency of each delay (y)
        for timestamp_diff in timestamp_differences:
            # Round to 10 ps
            rounded_timestamp_diff = round(timestamp_diff / bin_size_in_picoseconds) * bin_size_in_picoseconds
            rounded_timestamp_diff_index = int(rounded_timestamp_diff * half_array_length / half_array_size_in_picoseconds)
            if rounded_timestamp_diff_index > 0:
                rounded_timestamp_diff_index += int(half_array_size_in_picoseconds/bin_size_in_picoseconds)
            else:
                rounded_timestamp_diff_index = int(half_array_size_in_picoseconds/bin_size_in_picoseconds) + rounded_timestamp_diff_index
            hist_data[rounded_timestamp_diff_index] += 1

        # Extract the delays between pairs of detectors    
        # Fit the data to a Gaussian curve
        A_estimate = 10
        mu_estimate = 1000
        sigma_estimate = 1000
        popt, pcov = curve_fit(gaussian, delay_data, hist_data, p0=[A_estimate, mu_estimate, sigma_estimate])
        A_fit, mu_fit, sigma_fit = popt
        detector_st_devs.append(sigma_fit)

        fitted_delay[hist_pair] = mu_fit + approx_delay

        errs = np.sqrt(np.diag(pcov))
        A_err, mu_err, sigma_err = errs

        if A_err > np.sqrt(A_fit):
            raise ValueError("Fit is bad; modify p0 parameters")

        fine_grained_delay_data = np.linspace(delay_data[0], delay_data[-1], 1000)

        if save_plot and i < 8:
            fig.suptitle(f"Acquisition time: {acq_time_in_seconds} seconds; Added delay: {approx_delay} ps")
            ax = axes[i]
            ax.hist(delay_data[:-1], delay_data, weights=hist_data[:-1])
            ax.set_title(f'Detector pair {hist_pair}')
            ax.set_ylabel("Coinc. counts")
            ax.set_xlabel("Picoseconds")

            ax.grid(True)
            ax.plot(fine_grained_delay_data, gaussian(fine_grained_delay_data, A_fit, mu_fit, sigma_fit), label=f"$\mu={mu_fit:.0f} \pm {mu_err:.0f}$\n"
                                                                                             f"$\sigma={sigma_fit:.0f} \pm {sigma_err:.0f}$")
            ax.legend(prop={'size': 8})

    print(f"Avg standard deviation (detector RMS jitter): {np.mean(detector_st_devs)} ps")

    if save_plot:
        plt.tight_layout()
        plt.savefig(plot_folder_name + file_name + "_histograms_for_sync_matrix.png", dpi=300)

    # Map detector number to relevant detector pair for finding delay and solve for delays
    delay1 = 0
    delay7 = fitted_delay[(1,7)]
    delay2 = delay7 - fitted_delay[(2,7)]
    delay8 = fitted_delay[(2,8)] + delay2
    if basis0 == "AD":
        delay4 = delay7 - fitted_delay[(4,7)]
    else:
        delay4 = delay8 - fitted_delay[(4,8)]
    delay5 = fitted_delay[(4,5)] + delay4
    delay6 = fitted_delay[(4,6)] + delay4
    delay3 = delay6 - fitted_delay[(3,6)]

    detector_to_delay = {1: delay1, 2: delay2, 3: delay3, 4: delay4, 5: delay5, 6: delay6, 7: delay7, 8:delay8}

    # Alice's streamed data
    alice_basis_data = {}
    alice_channel_bit_data = {}
    for detector in range(1, 5):
        channel_json = data_dict[f'Ch{detector}_stream']

        # Round time stamps
        channel_data = json.loads(channel_json)
        delay = detector_to_delay[detector]
        channel_data = [timestamp + delay for timestamp in channel_data]

        if detector in [1, 2]:
            alice_basis_data = alice_basis_data | {timestamp: basis0 for timestamp in channel_data}
        else:
            alice_basis_data = alice_basis_data | {timestamp: basis1 for timestamp in channel_data}

        alice_channel_bit_data = alice_channel_bit_data | {timestamp: alice_detector_to_bit[detector] for timestamp in channel_data}

    bob_basis_data = {}
    bob_channel_bit_data = {}
    for detector in range(5, 9):
        channel_json = data_dict[f'Ch{detector}_stream']

        # Add delay
        channel_data = json.loads(channel_json)
        delay = detector_to_delay[detector]
        channel_data = [timestamp + delay for timestamp in channel_data]
        if detector in [5, 6]:
            bob_basis_data = bob_basis_data | {timestamp: basis0 for timestamp in channel_data}
        else:
            bob_basis_data = bob_basis_data | {timestamp: basis1 for timestamp in channel_data}

        bob_channel_bit_data = bob_channel_bit_data | {timestamp: bob_detector_to_bit[detector] for timestamp in channel_data}

    # Alice and Bob exchange time stamp data by basis (without sharing actual measurement results in that basis);
    # they only keep timestamps which both share between each of them (get rid of singles and only keep correlated pairs)
    p1, p2 = 0, 0
    ZZ_subspace = {"00": 0, "01": 0, "10": 0, "11":0}
    XX_subspace = {"00": 0, "01": 0, "10": 0, "11":0}
    XZ_subspace = {"00": 0, "01": 0, "10": 0, "11":0}
    ZX_subspace = {"00": 0, "01": 0, "10": 0, "11":0}

    alice_timestamps = sorted(list(alice_basis_data.keys()))
    bob_timestamps = sorted(list(bob_basis_data.keys()))

    while p1 < len(alice_timestamps) and p2 < len(bob_timestamps):
        # If the timestamps are close enough, add to appropriate subspace
        timestamp_difference = alice_timestamps[p1] - bob_timestamps[p2]

        # Line below allows + or - t_cc/2 (because of the abs), which means the total allowed t_cc will be `coincidence_window`.
        if abs(timestamp_difference) < coincidence_window/2:
            measurement = str(alice_channel_bit_data[alice_timestamps[p1]]) + str(bob_channel_bit_data[bob_timestamps[p2]])

            # Check for ZZ subspace
            if alice_basis_data[alice_timestamps[p1]] == "HV" and bob_basis_data[bob_timestamps[p2]] == "HV":
                ZZ_subspace[measurement] += 1
            # Check for ZX subspace
            elif alice_basis_data[alice_timestamps[p1]] == "HV" and bob_basis_data[bob_timestamps[p2]] == "AD":
                ZX_subspace[measurement] += 1
            # Check for XZ subspace
            elif alice_basis_data[alice_timestamps[p1]] == "AD" and bob_basis_data[bob_timestamps[p2]] == "HV":
                XZ_subspace[measurement] += 1
            # Check for XX subspace
            elif alice_basis_data[alice_timestamps[p1]] == "AD" and bob_basis_data[bob_timestamps[p2]] == "AD":
                XX_subspace[measurement] += 1
            else:
                raise ValueError("Unexpected basis information encountered")

            p1 += 1
            p2 += 1
        elif timestamp_difference < 0:
            p1 += 1
        else:
            p2 += 1

    # Turn counts into probabilities
    normalized_ZZ_subspace = {key: val /sum(ZZ_subspace.values()) for key, val in ZZ_subspace.items()}
    normalized_XZ_subspace = {key: val /sum(XZ_subspace.values()) for key, val in XZ_subspace.items()}
    normalized_ZX_subspace = {key: val /sum(ZX_subspace.values()) for key, val in ZX_subspace.items()}
    normalized_XX_subspace = {key: val /sum(XX_subspace.values()) for key, val in XX_subspace.items()}

    # Find error
    normalized_ZZ_subspace_err = {key: val /sum(ZZ_subspace.values()) * np.sqrt(np.sqrt(val)**2/val**2 + np.sqrt(sum(ZZ_subspace.values()))**2/sum(ZZ_subspace.values())**2) for key, val in ZZ_subspace.items()}
    normalized_XZ_subspace_err = {key: val /sum(XZ_subspace.values()) * np.sqrt(np.sqrt(val)**2/val**2 + np.sqrt(sum(XZ_subspace.values()))**2/sum(XZ_subspace.values())**2) for key, val in XZ_subspace.items()}
    normalized_ZX_subspace_err = {key: val /sum(ZX_subspace.values()) * np.sqrt(np.sqrt(val)**2/val**2 + np.sqrt(sum(ZX_subspace.values()))**2/sum(ZX_subspace.values())**2) for key, val in ZX_subspace.items()}
    normalized_XX_subspace_err = {key: val /sum(XX_subspace.values()) * np.sqrt(np.sqrt(val)**2/val**2 + np.sqrt(sum(XX_subspace.values()))**2/sum(XX_subspace.values())**2) for key, val in XX_subspace.items()}

    exp_corr_matrix = np.array([
        [normalized_XX_subspace["00"], normalized_XX_subspace["01"], normalized_XZ_subspace["00"], normalized_XZ_subspace["01"]],
        [normalized_XX_subspace["10"], normalized_XX_subspace["11"], normalized_XZ_subspace["10"], normalized_XZ_subspace["11"]],
        [normalized_ZX_subspace["00"], normalized_ZX_subspace["01"], normalized_ZZ_subspace["00"], normalized_ZZ_subspace["01"]],
        [normalized_ZX_subspace["10"], normalized_ZX_subspace["11"], normalized_ZZ_subspace["10"], normalized_ZZ_subspace["11"]]
    ])

    def subspace_probability_to_text(val, err):
        # Keep one sig digit
        err_rounding_space = -(int(np.floor(np.log10(err))))
        val_rounding_space = err_rounding_space - 1

        # If err begins with 1, keep two sig digits
        if '1' in str(err).strip('0').strip('.').strip('0')[0]:
            err_rounding_space += 1

        err_sig_digit = round(err, err_rounding_space)
        val_sig_digit = round(val, val_rounding_space)
 
        err_sig_digit_str = str(err_sig_digit)
        err_str = err_sig_digit_str.strip('0')
        err_str = err_str.strip('.')
        err_str = err_str.strip('0')

        return str(val_sig_digit) + '(' + err_str + ')'

    exp_corr_matrix_text = np.array([
        [subspace_probability_to_text(normalized_XX_subspace["00"], normalized_XX_subspace_err["00"]),
        subspace_probability_to_text(normalized_XX_subspace["01"], normalized_XX_subspace_err["01"]),
        subspace_probability_to_text(normalized_XZ_subspace["00"], normalized_XZ_subspace_err["00"]),
        subspace_probability_to_text(normalized_XZ_subspace["01"], normalized_XZ_subspace_err["01"])],
        [subspace_probability_to_text(normalized_XX_subspace["10"], normalized_XX_subspace_err["10"]),
        subspace_probability_to_text(normalized_XX_subspace["11"], normalized_XX_subspace_err["11"]),
        subspace_probability_to_text(normalized_XZ_subspace["10"], normalized_XZ_subspace_err["10"]),
        subspace_probability_to_text(normalized_XZ_subspace["11"], normalized_XZ_subspace_err["11"])],
        [subspace_probability_to_text(normalized_ZX_subspace["00"], normalized_ZX_subspace_err["00"]),
        subspace_probability_to_text(normalized_ZX_subspace["01"], normalized_ZX_subspace_err["01"]),
        subspace_probability_to_text(normalized_ZZ_subspace["00"], normalized_ZZ_subspace_err["00"]),
        subspace_probability_to_text(normalized_ZZ_subspace["01"], normalized_ZZ_subspace_err["01"])],
        [subspace_probability_to_text(normalized_ZX_subspace["10"], normalized_ZX_subspace_err["10"]),
        subspace_probability_to_text(normalized_ZX_subspace["11"], normalized_ZX_subspace_err["11"]),
        subspace_probability_to_text(normalized_ZZ_subspace["10"], normalized_ZZ_subspace_err["10"]),
        subspace_probability_to_text(normalized_ZZ_subspace["11"], normalized_ZZ_subspace_err["11"])]
    ])


    # Calculate correlation matrix fidelity according to "Frequency-bin entanglement-based Quantum Key Distribution", Tagliavacche 2024
    ideal_corr_matrix = np.array([
        [0.5, 0, 0.25, 0.25],
        [0, 0.5, 0.25, 0.25],
        [0.25, 0.25, 0, 0.5],
        [0.25, 0.25, 0.5, 0],
    ])

    fidelity_numerator = np.trace(np.matmul(exp_corr_matrix.conj().T, ideal_corr_matrix)) * np.trace(np.matmul(ideal_corr_matrix.conj().T, exp_corr_matrix))
    fidelity_denom = np.trace(np.matmul(exp_corr_matrix.conj().T, exp_corr_matrix)) * np.trace(np.matmul(ideal_corr_matrix.conj().T, ideal_corr_matrix))

    fidelity = fidelity_numerator / fidelity_denom
    print(f"Correlation matrix fidelity: {fidelity}")

    fig, ax = plt.subplots()

    colormap = 'Purples'
    cax = ax.imshow(exp_corr_matrix, cmap=colormap, vmin=0, vmax=0.50)

    # Add colorbar
    cbar = fig.colorbar(cax)

    # Custom tick labels
    x_labels = [r"$|+\rangle_A$", r"$|-\rangle_A$", r"$|0\rangle_A$", r"$|1\rangle_A$"]
    y_labels = [r"$|+\rangle_B$", r"$|-\rangle_B$", r"$|0\rangle_B$", r"$|1\rangle_B$"]

    for (i, j), txt in np.ndenumerate(exp_corr_matrix_text):
        ax.text(j, i, txt, ha='center', va='center', color='white' if ('0.5' in txt) or ('0.4' in txt) else 'black')


    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    if not FINAL_PLOT:
        fig.suptitle(f"Correlation matrix of probability outcomes\nCoinc. window: {coincidence_window} ps, Fidelity: {fidelity:.3f}")
    plt.savefig(plot_folder_name + file_name + "_correlation_matrix.png", dpi=300)



if __name__ == "__main__":
    """
    Run the analysis.
    """
    # Parse in file name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name with simulation results")
    args = parser.parse_args()

    data_folder_name = "timetagged_data/"
    plot_folder_name = "experiment_bbm92_plots/"

    # Create folders
    if not os.path.isdir(data_folder_name):
        os.mkdir(data_folder_name)

    if not os.path.isdir(plot_folder_name):
        os.mkdir(plot_folder_name)

    make_correlation_matrix(
        simulation_results_folder_path=data_folder_name,
        plot_folder_name=plot_folder_name,
        file_name=args.file_name,
        coincidence_window=2000,
        approx_delay=15000,
        save_plot=True,
    )
