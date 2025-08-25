# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import argparse
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
Make a plot which shows the relative error between key rate, secure key rate, or qber from the numerical
simulations, the aqnsim simulations, and the experimental results.

Usage: 
`python make_error_plot.py -f [FOLDER_NAME] -y [Y_PARAMETER_NAME]`
where FOLDER_NAME is the name of the folder containing
json files with numerical results, aqnsim results, and experimental results,
Y_PARAMETER is the name of the parameter to plot on the y axis (should be either "qbers",
"raw_key_rates" or "secure_key_rates"), e.g. "link_loss_in_db", "minimum_time_resolution", etc.
"""


def plot_comparison(simulation_results_folder_path: str, y_parameter_name: str):
    """
    Plot comparison of simulations and experiment.

    :param simulation_results_folder_path: Folder path for the simulation results contained in a dict.
    :param y_parameter_name: The name of the parameter to plot on the y-axis, e.g. either "qbers",
        "raw_key_rates" or "secure_key_rates".
    """
    matplotlib.rcParams.update({"font.size": 20})

    # Get list of json files
    file_name_list = [
        f
        for f in os.listdir(simulation_results_folder_path)
        if os.path.isfile(os.path.join(simulation_results_folder_path, f))
        and f.endswith(".json")
    ]
    file_name_list = sorted(file_name_list)
    y_parameter_error_name = y_parameter_name[:-1] + "_errors"
    y_parameter_label_name = y_parameter_name + "_label"

    for file_name in file_name_list:
        with open(simulation_results_folder_path + "/" + file_name) as file:
            results = json.load(file)
        x_parameter_name = results.get("x_parameter_name")
        data_type = results.get("data_type")

        if data_type == "aqnsim":
            aqnsim_x_data = np.array(results[x_parameter_name])
            aqnsim_y_data = np.array(results[y_parameter_name])
            aqnsim_y_error = np.array(results[y_parameter_error_name])
        elif data_type == "numerics":
            numerics_x_data = np.array(results[x_parameter_name])
            numerics_y_data = np.array(results[y_parameter_name])
        elif data_type == "experiment":
            experiment_x_data = np.array(results[x_parameter_name])
            experiment_y_data = np.array(results[y_parameter_name])
            experiment_y_error = np.array(results[y_parameter_error_name])
        else:
            raise ValueError("Unknown data type for json file.")

    # Plot results
    if x_parameter_name == "coincidence_window":
        fig, axs = plt.subplots(1, figsize=(10.5, 2.5))
    else:
        fig, axs = plt.subplots(1, figsize=(5.5, 3.85))

    if y_parameter_name == "qbers":
        aqnsim_y_data = np.array(aqnsim_y_data)
        aqnsim_y_data[aqnsim_y_data == 0] = np.nan
        if x_parameter_name == "coincidence_window":
            experiment_y_data = np.array(experiment_y_data)
            experiment_y_data[experiment_y_data == 0] = np.nan

    relative_difference_aqnsim_exp = (
        aqnsim_y_data - experiment_y_data
    ) / experiment_y_data
    delta_difference_aqnsim_exp = np.sqrt(
        (aqnsim_y_error) ** 2 + (experiment_y_error) ** 2
    )
    exp_aqnsim_y_error = relative_difference_aqnsim_exp * np.sqrt(
        (delta_difference_aqnsim_exp / (aqnsim_y_data - experiment_y_data)) ** 2
        + (experiment_y_error / experiment_y_data) ** 2
    )

    relative_difference_theory_exp = (
        numerics_y_data - experiment_y_data
    ) / experiment_y_data
    exp_theory_y_error = relative_difference_theory_exp * np.sqrt(
        (experiment_y_error / (numerics_y_data - experiment_y_data)) ** 2
        + (experiment_y_error / experiment_y_data) ** 2
    )

    x_start = 1e-10  # In seconds
    experiment_x_data_trunc = experiment_x_data[experiment_x_data > x_start]
    relative_difference_aqnsim_exp_trunc = relative_difference_aqnsim_exp[
        experiment_x_data > x_start
    ]
    relative_difference_theory_exp_trunc = relative_difference_theory_exp[
        experiment_x_data > x_start
    ]
    exp_aqnsim_y_error_trunc = exp_aqnsim_y_error[experiment_x_data > x_start]
    exp_theory_y_error_trunc = exp_theory_y_error[experiment_x_data > x_start]

    # Find the mean squared errors for the region of interest
    aqnsim_exp_diff = aqnsim_y_data - experiment_y_data
    theory_exp_diff = numerics_y_data - experiment_y_data
    aqnsim_exp_diff_trunc = aqnsim_exp_diff[experiment_x_data > x_start]
    theory_exp_diff_trunc = theory_exp_diff[experiment_x_data > x_start]

    rmse_aqnsim_exp = np.sqrt(np.mean(aqnsim_exp_diff_trunc**2))
    rmse_theory_exp = np.sqrt(np.mean(theory_exp_diff_trunc**2))

    unit_text = "bps" if y_parameter_name == "raw_key_rates" else ""
    text_y_pos = -0.08 if y_parameter_name == "raw_key_rates" else 0.25

    plt.axhline(y=0, color="black", linestyle="dotted")
    plt.text(
        8e-9,
        text_y_pos,
        f"AQNSim/Exp RMSE: {rmse_aqnsim_exp:.1e} {unit_text}\nTheory/Exp RMSE: {rmse_theory_exp:.1e} {unit_text}",
        fontsize=12,
        fontstyle="italic",
        color="black",
    )
    p1 = axs.errorbar(
        experiment_x_data_trunc,
        relative_difference_aqnsim_exp_trunc,
        np.abs(exp_aqnsim_y_error_trunc),
        color="#ff8701",
        capsize=2,
        markersize=5,
        label="(AQNSim-Exp)/Exp",
        linewidth=2,
        linestyle="",
    )
    p1 = axs.errorbar(
        experiment_x_data_trunc,
        relative_difference_theory_exp_trunc,
        np.abs(exp_theory_y_error_trunc),
        color="#80c32d",
        capsize=2,
        markersize=5,
        label="(Theory-Exp)/Exp",
        linewidth=1,
        linestyle="",
    )
    first_step = experiment_x_data_trunc[1] - experiment_x_data_trunc[0]
    last_step = experiment_x_data_trunc[-1] - experiment_x_data_trunc[-2]

    plt.xlim(
        [
            experiment_x_data_trunc[0] - first_step,
            experiment_x_data_trunc[-1] + last_step,
        ]
    )
    axs.set_xscale("log")

    legend_font_size = 14
    x_parameter_label = results.get("x_parameter_label")

    axs.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    axs.legend(fontsize=legend_font_size, loc="best", frameon=False)
    axs.grid(True, linestyle="--", linewidth=0.5)

    axs.set_xlabel(x_parameter_label)
    axs.set_ylabel("Relative error")
    plt.tight_layout()

    plt.savefig(
        f"{simulation_results_folder_path}/{x_parameter_name}_{y_parameter_name}_relative_error.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    """
    Plot average time until generation entanglement vs loss.
    """
    # Parse in folder name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder-name", help="Folder name with simulation results"
    )
    parser.add_argument(
        "-y",
        "--y-parameter-name",
        help="Parameter name to plot as the y axis. Should be secure_key_rates, raw_key_rates, or qbers",
    )

    args = parser.parse_args()

    plot_comparison(
        simulation_results_folder_path=args.folder_name,
        y_parameter_name=args.y_parameter_name,
    )
