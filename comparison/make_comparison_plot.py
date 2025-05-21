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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
Make a plot which will compare key rate, secure key rate, or qber from the numerical
simulations, the aqnsim simulations, and the experimental results.

Usage:
`python make_comparison_plot.py -f [FOLDER_NAME] -y [Y_PARAMETER_NAME]`
where FOLDER_NAME is the name of the folder containing
json files with numerical results, aqnsim results, and experimental results,
Y_PARAMETER is the name of the parameter to plot on the y axis (should be either "qbers",
"raw_key_rates" or "secure_key_rates"), e.g. "link_loss_in_db", "minimum_time_resolution", etc.
"""
matplotlib.rcParams.update({"font.size": 16})
LARGE_EXP = False


def plot_comparison(results_folder_path: str, y_parameter_name: str):
    """
    Plot comparison of simulations and experiment.

    :param results_folder_path: Folder path for the simulation results contained in a dict.
    :param y_parameter_name: The name of the parameter to plot on the y-axis, e.g. either "qbers",
        "raw_key_rates" or "secure_key_rates".
    """
    # Get list of json files
    file_name_list = [
        f
        for f in os.listdir(results_folder_path)
        if os.path.isfile(os.path.join(results_folder_path, f)) and f.endswith(".json")
    ]

    file_name_list = sorted(file_name_list)
    y_parameter_error_name = y_parameter_name[:-1] + "_errors"
    y_parameter_label_name = y_parameter_name + "_label"

    for file_name in file_name_list:
        with open(results_folder_path + "/" + file_name) as file:
            results = json.load(file)
        x_parameter_name = results.get("x_parameter_name")

        data_type = results.get("data_type")

        if data_type == "aqnsim":
            aqnsim_x_data = results[x_parameter_name]
            aqnsim_y_data = results[y_parameter_name]
            aqnsim_y_error = results[y_parameter_error_name]
        elif data_type == "numerics":
            numerics_x_data = results[x_parameter_name]
            numerics_y_data = results[y_parameter_name]
        elif data_type == "experiment":
            experiment_x_data = results[x_parameter_name]
            experiment_y_data = results[y_parameter_name]
            experiment_y_error = results[y_parameter_error_name]
        else:
            raise ValueError("Unknown data type for json file.")

    # Plot results
    if x_parameter_name == "coincidence_window":
        fig, axs = plt.subplots(1, figsize=(14.5, 3.75))
    else:
        fig, axs = plt.subplots(1, figsize=(5.5, 3.75))

    colors = plt.colormaps["Set1"]
    colors = [colors(0), colors(1), colors(8)]

    if y_parameter_name == "qbers":
        aqnsim_y_data = np.array(aqnsim_y_data)
        aqnsim_y_data[aqnsim_y_data == 0] = np.nan
        if x_parameter_name == "coincidence_window":
            experiment_y_data = np.array(experiment_y_data)
            experiment_y_data[experiment_y_data == 0] = np.nan

    experiment_y_lower_error = np.fmax(np.array(experiment_y_error), 0)
    experiment_y_upper_error = np.array(experiment_y_error)
    if not LARGE_EXP:
        p1 = axs.errorbar(
            experiment_x_data,
            experiment_y_data,
            [experiment_y_lower_error, experiment_y_upper_error],
            color=colors[0],
            capsize=2,
            markersize=5,
            label="Experiment",
            linewidth=1,
            linestyle="",
        )
    if x_parameter_name in [
        "detector_dead_time",
        "dark_count_rate",
        "source_pair_rate",
        "coincidence_window",
    ]:
        axs.set_xscale("log")

    p2 = axs.plot(
        numerics_x_data,
        numerics_y_data,
        "--",
        color=colors[1],
        label="Theory",
        linewidth=2,
    )

    aqnsim_y_lower_err = np.fmax(np.array(aqnsim_y_error), 0)
    aqnsim_y_upper_err = np.array(aqnsim_y_error)

    p3 = axs.errorbar(
        aqnsim_x_data,
        aqnsim_y_data,
        [aqnsim_y_lower_err, aqnsim_y_upper_err],
        capsize=2,
        markersize=5,
        color=colors[2],
        label="AQNSim",
        linewidth=1,
        linestyle="",
    )
    if LARGE_EXP:
        p1 = axs.errorbar(
            experiment_x_data,
            experiment_y_data,
            [experiment_y_lower_error, experiment_y_upper_error],
            color=colors[0],
            capsize=4,
            markersize=5,
            label="Experiment",
            linewidth=4,
            linestyle="",
        )
        legend_font_size = 14
    else:
        legend_font_size = 10
    x_parameter_label = results.get("x_parameter_label")
    y_parameter_label = results.get(y_parameter_label_name)
    axs.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    axs.legend(fontsize=legend_font_size, loc="best", frameon=False)
    axs.grid(True, linestyle="--", linewidth=0.5)

    axs.set_xlabel(x_parameter_label)
    axs.set_ylabel(y_parameter_label)
    if y_parameter_name != "qbers":
        plt.yscale("log")
    plt.tight_layout()

    plt.savefig(
        f"{results_folder_path}/{x_parameter_name}_{y_parameter_name}.png",
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
        results_folder_path=args.folder_name,
        y_parameter_name=args.y_parameter_name,
    )
