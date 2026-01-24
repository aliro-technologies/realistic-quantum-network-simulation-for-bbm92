# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
Usage:
`python plot_rate_vs_loss_sequence.py -f [FILE_PATH_NAME]`
where FILE_PATH_NAME is the name of the file path (without the extension) generated from running `run_two_node_sequence_ent_gen.py`.
"""


def plot_rate_vs_loss_sequence(sequence_simulation_results_file_path):
    """
    Plot rate vs loss for SeQUeNCe simulation results for establishing entanglement given 0 repeater nodes.

    :param sequence_simulation_results_file_path: File path for the SeQUeNCe simulation results.
    """
    colors = matplotlib.colormaps["Set1"]

    fig, axs = plt.subplots(1)
    fig.set_size_inches(6, 3)

    sim_type = "SeQUeNCe"
    file_color = colors(2)

    with open(sequence_simulation_results_file_path + ".json") as file:
        simulation_results = json.load(file)

    num_repeaters = simulation_results["num_repeaters"]
    elementary_link_quantum_delay = simulation_results["elementary_link_quantum_delay"]

    if elementary_link_quantum_delay != 1:
        raise ValueError(
            "Elementary quantum link delay must be 1: enforce normalization to simplify axes labels."
        )
    if num_repeaters != 0:
        raise ValueError("Number of repeaters must be 0 for this script.")

    num_shots = simulation_results["num_shots"]
    BSM_efficiency = simulation_results["BSM_efficiency"]

    link_losses_in_db = np.array(simulation_results["link_losses_in_db"])
    ent_gen_times = np.array(simulation_results["ent_gen_times"])

    sim_time_means = np.mean(ent_gen_times, axis=1)
    sim_time_errs = np.std(ent_gen_times, axis=1) / np.sqrt(
        np.size(ent_gen_times, axis=1)
    )
    axs.errorbar(
        link_losses_in_db,
        1 / sim_time_means,
        (1 / sim_time_means) ** 2 * sim_time_errs,
        fmt="none",
        capsize=8,
        color=file_color,
        label=sim_type,
    )

    # Calculate analytical rates
    nesting_level = np.log2(num_repeaters + 1)
    elementary_link_transmittivities = pow(10, -link_losses_in_db / 10)
    P0 = (
        elementary_link_transmittivities * BSM_efficiency
    )  # 1/2 for linear optical BSM efficiency

    # Get times to entanglement in "attempts" since `elementary_link_quantum_delay` is enforced to be 1
    theory_curve_in_attempts = elementary_link_quantum_delay / P0

    # Plot rates as c/L_0 rather than c/(2 * L_0) to incorporate minimum classical communication times
    axs.set_ylabel("Avg. entanglement\ngeneration rate ($c/L_0$)")

    axs.plot(
        link_losses_in_db,
        1 / theory_curve_in_attempts,
        color=colors(1),
        label="Theory",
    )

    axs.legend(frameon=False)
    axs.set_xlabel("Elementary link loss (dB)")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(
        f"{sequence_simulation_results_file_path}_rate_vs_loss_and_sequence.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    """
    Plot average entanglement generation rate vs loss.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file-name-sequence",
        help="File name with results for SeQUeNCe simulation",
    )

    args = parser.parse_args()

    plot_rate_vs_loss_sequence(
        sequence_simulation_results_file_path=args.file_name_sequence
    )
