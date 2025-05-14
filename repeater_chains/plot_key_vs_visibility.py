import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import os

"""
To run: 
`python plot_key_vs_visibility.py -f [FOLDER_NAME]`
where FOLDER_NAME is the name of the file generated from running `run_repeater_chain_visibility_sweep.py`.
"""
SMALL_FIG = True

def calc_secure_key_rates(fidelities, time_means, time_errs=None):
    """
    Calculate secure key rates and secure key rate errors from final state fidelities,
    entanglement generation time means and errors.
    :param fidelities: The final measured fidelity for the entangled pair.
    :param time_means: The mean time to entanglement generation.
    :param time_errs: The error for the mean time to entanglement generation.
    """
    # Convert fidelities into QBERs
    qbers = (1 - fidelities) * (2/3)

    # Calculate predicted secure key rates
    # 1/2 for sifting step
    raw_key_rate_means = (1/2) * 1/time_means
    if time_errs is not None:
        raw_key_rate_errs = (1/2) * (1/time_means)**2 * time_errs

    def binary_entropy_function(x):
        if 0 < x < 1:
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        else:
            return 0

    binary_entropy = np.array([binary_entropy_function(q) for q in qbers])
   
    # Find secure key rate
    secure_key_rate_means = raw_key_rate_means * (1 - 2.1 * binary_entropy)
    secure_key_rate_means = [max(0, rate) for rate in secure_key_rate_means]

    if time_errs is not None:
        secure_key_rate_errs = raw_key_rate_errs * (1 - 2.1 * binary_entropy)
        secure_key_rate_errs = [max(0, err) for err in secure_key_rate_errs]
    else:
        secure_key_rate_errs = None
    return secure_key_rate_means, secure_key_rate_errs


def plot_rate_vs_loss_n_repeaters_with_theory(simulation_results_folder_path):
    """
    :param simulation_results_folder_path: File path for the simulation results contained in a dict.
    """
    sim_file_names = [f for f in os.listdir(simulation_results_folder_path)
                      if os.path.isfile(os.path.join(simulation_results_folder_path, f))
                      and f.lower().endswith('.json')]

    sorted_sim_file_names = sorted(sim_file_names)

    # Plot results
    if SMALL_FIG:
        font_size = 12
        plt.rcParams.update({"font.size": font_size})

    fig, axs = plt.subplots(1)

    if SMALL_FIG:
        fig.set_size_inches(6, 6)

    colors = plt.cm.get_cmap("tab20b")

    for sim_file_name in sorted_sim_file_names:
        with open(simulation_results_folder_path + sim_file_name) as file:
            simulation_results = json.load(file)

        num_repeaters = simulation_results["num_repeaters"]
        elementary_link_quantum_delay = simulation_results["elementary_link_quantum_delay"]
        num_shots = simulation_results["num_shots"]
        total_loss_in_db = simulation_results["total_loss_in_db"]

        depolarizing_probs = np.array(simulation_results["depolarizing_probs"])
        ent_gen_times = np.array(simulation_results["ent_gen_times"])
        fidelities = np.array(simulation_results["fidelities"])
        BSM_efficiency = simulation_results["BSM_efficiency"]
        total_loss_in_db = simulation_results["total_loss_in_db"]
        link_loss_in_db = total_loss_in_db / (num_repeaters + 1)
        elementary_link_transmittivity = pow(10, -link_loss_in_db / 10)
        P0 = elementary_link_transmittivity * BSM_efficiency

        sim_fidelity_means = np.mean(fidelities, axis=1)
        sim_fidelity_errs = np.std(fidelities, axis=1) / np.sqrt(
            np.size(fidelities, axis=1)
        )
        # The fidelity error is approx 0 so we only consider the key rate error
        assert np.all(sim_fidelity_errs < 1e-8)

        sim_time_means = np.mean(ent_gen_times, axis=1)
        sim_time_errs = np.std(ent_gen_times, axis=1) / np.sqrt(
            np.size(ent_gen_times, axis=1)
        )

        sim_secure_key_rates, sim_secure_key_rate_errs = calc_secure_key_rates(sim_fidelity_means, sim_time_means, sim_time_errs)

        color = colors((num_repeaters) + (int(total_loss_in_db/5)-1) * 4)
        print((num_repeaters) + (int(total_loss_in_db/5)-1) * 4)

        elementary_pair_fidelities = 1 + (-(3/2) + (3/4)*depolarizing_probs)*depolarizing_probs

        axs.errorbar(
            elementary_pair_fidelities,
            sim_secure_key_rates,
            sim_secure_key_rate_errs,
            fmt='none',
            color=color,
            capsize=2,
            linewidth=2,
            label=f"AQNSim: r = {num_repeaters}, $\\beta$ = {total_loss_in_db} dB",
        )

        axs.set_ylabel("Secure key rate ($c/L_0$)")

        ## Calculate and plot theory curve
        p1 = 1 # Single qubit depolarizing during entanglement swap
        p2 = 1 # Two qubit depolarizing during entanglement swap
        eta = 1 # Measurement error
        L = num_repeaters + 1 # Number of elementary links

        # Below, from Briegel 1998
        elementary_pair_fidelities_theory = np.linspace(elementary_pair_fidelities[0], elementary_pair_fidelities[-1], 500)
        theory_fidelities = (1/4) + (3/4)*(p1 * p2 *(4 * eta**2 - 1)/3)**(L-1)*((4*elementary_pair_fidelities_theory - 1)/3)**L

        # Calculate analytical rates
        nesting_level = np.log2(num_repeaters + 1)
        if num_repeaters == 1:
            # Analytical expression for special case of ONE repeater
            theory_rate = (3 - 2 * P0) * elementary_link_quantum_delay / ((2 - P0) * P0)
        else:
            # Approximation for average time to entanglement for n repeaters
            theory_rate = (3 / 2) ** nesting_level * elementary_link_quantum_delay * 1 / P0

        if num_repeaters > 1:
            theory_label = f"Est. theory: r = {num_repeaters}, $\\beta$ = {total_loss_in_db} dB"
        else:
            theory_label = f"Theory: r = {num_repeaters}, $\\beta$ = {total_loss_in_db} dB"

        theory_secure_key_rates, _ = calc_secure_key_rates(theory_fidelities, theory_rate)
        axs.plot(
            elementary_pair_fidelities_theory,
            theory_secure_key_rates,
            color=color,
            linewidth=1,
            linestyle=':',
            label=theory_label,
        )
    
    axs.grid(True, linestyle="--", linewidth=0.5)
    fig.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    plt.subplots_adjust(top=0.80)
    axs.set_xlabel("Elementary link Werner state fidelity")
    plt.yscale("log")
    plt.ylim(bottom=1e-4)

    # Reverse x axis
    plt.xlim(max(elementary_pair_fidelities), min(elementary_pair_fidelities))

    plt.savefig(
        f"{simulation_results_folder_path}key_vs_loss_n_repeaters.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    """
    Plot BBM92 secure key rate vs visibility.
    """
    main_folder_name = "repeater_chain_results"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder-name", help="Folder name with simulation result files")
    args = parser.parse_args()

    plot_rate_vs_loss_n_repeaters_with_theory(simulation_results_folder_path=main_folder_name + "/" + args.folder_name + "/")
