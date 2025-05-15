from datetime import datetime
import numpy as np
import copy

import uuid
import os
import json
from repeater_chain import run_repeater_chain
from elementary_link import run_elementary_link_simulation


"""
To run:
`python run_repeater_chain_visibility_sweep.py`
"""


def run_n_repeater_chain_sims(num_repeaters, total_link_loss):
    """
    Simulate repeater chain with loss.
    """
    # Set up directory to save results
    main_folder_name = "repeater_chain_results/"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)
    dir_path = main_folder_name + datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Get filename based on a uuid.
    uuid_string = str(uuid.uuid4())
    file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_" + uuid_string
    file_path = dir_path + "/" + file_name_string

    random_seed = 1
    num_shots = 10  # 1000 # Number of shots to simulate #10
    elementary_link_quantum_delay = 1
    elementary_link_classical_delay = 0  # Can modify this from 0 as needed
    BSM_efficiency = 1 / 2
    BSM_loss = -10 * np.log10(BSM_efficiency)

    num_data_points = 4  # 30 #4
    depol_probs = list(np.linspace(0, 0.05, num_data_points))

    fidelities = []
    ent_gen_times = []

    for i, depol_prob in enumerate(depol_probs):
        # Run simulation with passed in simulation parameters
        # Run many simulations and find the average simulated time and ending fidelity.

        link_loss_in_db = total_link_loss / (num_repeaters + 1)

        # Run simulations and save data to a CSV file
        simulation_parameters = {
            "random_seed": random_seed, # Seed to seed the simulation
            "num_shots": num_shots, # Number of shots in the simulation
            "num_repeaters": num_repeaters, # Number of repeaters in the repeater chain
            "elementary_link_loss_in_db": link_loss_in_db
            + BSM_loss,  # Channel loss probability, in dB with added BSM loss, in dB
            "elementary_link_quantum_delay": elementary_link_quantum_delay,  # Channel delay for qubits, in seconds
            "elementary_link_classical_delay": elementary_link_classical_delay,  # Channel delay for classical messages, in seconds
            "H_delay": 0,  # Hadamard delay, in seconds
            "X_delay": 0,  # X gate delay, in seconds
            "Z_delay": 0,  # Z gate delay, in seconds
            "CNOT_delay": 0,  # CNOT gate delay, in seconds
            "meas_delay": 0,  # Measurement delay, in seconds
            "bsm_delay": 0,  # Delay for BSM application, in seconds
            "depolarizing_prob": depol_prob,  # Depolarizing probability per qubit
            "file_name": file_path, # File path name
        }

        fidelity_set, ent_gen_set = run_repeater_chain(**simulation_parameters)

        fidelities += [fidelity_set]
        ent_gen_times += [ent_gen_set]

    # Save compiled results
    simulation_results = copy.deepcopy(simulation_parameters)
    del simulation_results["depolarizing_prob"]
    simulation_results["depolarizing_probs"] = depol_probs
    simulation_results["total_loss_in_db"] = total_link_loss
    simulation_results["fidelities"] = fidelities
    simulation_results["ent_gen_times"] = ent_gen_times
    simulation_results["BSM_efficiency"] = BSM_efficiency

    with open(file_path + "_sim_results" + ".json", "w") as file:
        file.write(json.dumps(simulation_results, indent=2))


def run_elementary_link_sim(total_link_loss):
    """
    Simulate elementary link entanglement generation with loss, given loss is implemented through qubit deletion.
    """
    # Set up directory to save results
    main_folder_name = "repeater_chain_results/"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)
    dir_path = main_folder_name + datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Get filename based on a uuid.
    uuid_string = str(uuid.uuid4())
    file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_" + uuid_string
    file_path = dir_path + "/" + file_name_string

    random_seed = 1
    num_shots = 3  # Number of shots to simulate #10 #1000
    elementary_link_quantum_delay = 1
    BSM_efficiency = 1 / 2
    BSM_loss = -10 * np.log10(BSM_efficiency)

    num_data_points = 2  # 4 #30
    depol_probs = list(np.linspace(0, 0.05, num_data_points))

    fidelities = []
    ent_gen_times = []

    for i, depol_prob in enumerate(depol_probs):
        link_loss_in_db = total_link_loss

        # Run simulations and save data to a CSV file
        simulation_parameters = {
            "random_seed": random_seed,
            "num_shots": num_shots,
            "elementary_link_loss_in_db": link_loss_in_db
            + BSM_loss,  # Channel loss probability, in dB
            "elementary_link_quantum_delay": elementary_link_quantum_delay,  # Channel delay for qubits, in seconds
            "depolarizing_prob": depol_prob,  # Depolarizing probability per qubit
        }

        fidelity_set, ent_gen_set = run_elementary_link_simulation(
            **simulation_parameters
        )

        fidelities += [fidelity_set]
        ent_gen_times += [ent_gen_set]

    # Save compiled results
    simulation_results = copy.deepcopy(simulation_parameters)
    del simulation_results["depolarizing_prob"]
    simulation_results["num_repeaters"] = 0
    simulation_results["depolarizing_probs"] = depol_probs
    simulation_results["total_loss_in_db"] = total_link_loss
    simulation_results["fidelities"] = fidelities
    simulation_results["ent_gen_times"] = ent_gen_times
    simulation_results["BSM_efficiency"] = BSM_efficiency

    with open(file_path + "_sim_results" + ".json", "w") as file:
        file.write(json.dumps(simulation_results, indent=2))


if __name__ == "__main__":
    num_repeaters_list = [
        0,
        1,
        2,
        3,
    ]  # Number of repeaters (Alice and Bob, the end nodes, are counted separately)
    total_list_loss_list = [5, 20]  # Total loss for all links, in dB
    for n in num_repeaters_list:
        for t in total_list_loss_list:
            if n == 0:
                run_elementary_link_sim(t)
            else:
                run_n_repeater_chain_sims(n, t)
