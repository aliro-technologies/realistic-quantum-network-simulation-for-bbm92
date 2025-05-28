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
`python plot_rate_vs_loss.py -f [FILE_PATH_NAME]`
where FILE_PATH_NAME is the name of the file path (without the extension) generated from running `run_repeater_chain_loss_sweep.py`.
"""
SMALL_FIG = True


def plot_rate_vs_loss_one_repeater(simulation_results_file_path):
    """
    :param simulation_results_file_path: File path for the simulation results contained in a dict.
    """
    with open(simulation_results_file_path + ".json") as file:
        simulation_results = json.load(file)

    num_repeaters = simulation_results["num_repeaters"]
    elementary_link_quantum_delay = simulation_results["elementary_link_quantum_delay"]
    num_shots = simulation_results["num_shots"]
    BSM_efficiency = simulation_results["BSM_efficiency"]

    link_losses_in_db = np.array(simulation_results["link_losses_in_db"])
    ent_gen_times = np.array(simulation_results["ent_gen_times"])

    # Calculate analytical rates
    nesting_level = np.log2(num_repeaters + 1)
    elementary_link_transmittivities = pow(10, -link_losses_in_db / 10)
    P0 = (
        elementary_link_transmittivities * BSM_efficiency
    )  # 1/2 for linear optical BSM efficiency

    upper_bound_theory_curve = 1 / P0 * 2**nesting_level * elementary_link_quantum_delay
    lower_bound_theory_curve = 1 / P0 * elementary_link_quantum_delay

    if num_repeaters == 1:
        # Analytical expression for special case of ONE repeater
        theory_curve = (3 - 2 * P0) * elementary_link_quantum_delay / ((2 - P0) * P0)
    else:
        # Approximation for average time to entanglement for n repeaters
        theory_curve = (3 / 2) ** nesting_level * elementary_link_quantum_delay * 1 / P0

    # Plot results
    if SMALL_FIG:
        font_size = 14
        plt.rcParams.update({"font.size": font_size})

    fig, axs = plt.subplots(1)

    if SMALL_FIG:
        fig.set_size_inches(6, 3)

    if not SMALL_FIG:
        fig.suptitle(
            f"Entanglement generation in a {num_repeaters}-node repeater chain, {num_shots} shots"
        )
    colors = matplotlib.colormaps["Set1"]

    sim_time_means = np.mean(ent_gen_times, axis=1)
    sim_time_errs = np.std(ent_gen_times, axis=1) / np.sqrt(
        np.size(ent_gen_times, axis=1)
    )
    axs.errorbar(
        link_losses_in_db,
        1 / sim_time_means,
        (1 / sim_time_means) ** 2 * sim_time_errs,
        fmt="none",
        color=colors(0),
        capsize=8,
        label="AQNSim",
    )

    axs.set_ylabel("Avg. entanglement\ngeneration rate ($c/L_0$)")

    axs.plot(
        link_losses_in_db,
        1 / theory_curve,
        color=colors(1),
        label="Theory",
    )

    axs.legend(frameon=False)
    axs.set_xlabel("Elementary link loss (dB)")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig(
        f"{simulation_results_file_path}_rate_vs_loss.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    """
    Plot average entanglement generation rate vs loss.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", help="File name with simulation results")
    args = parser.parse_args()

    plot_rate_vs_loss_one_repeater(simulation_results_file_path=args.file_name)
