from datetime import datetime
import numpy as np
import copy

import uuid
import os
import json
from repeater_chain import run_repeater_chain


"""
To run:
`python run_repeater_chain_loss_sweep.py`
"""


if __name__ == "__main__":
    """
    Simulate repeater chain with loss
    """
    # Set up directory to save results
    main_folder_name = "repeater_chain_results/"
    if not os.path.isdir(main_folder_name):
        os.mkdir(main_folder_name)
    dir_path = main_folder_name + datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    random_seed = 4
    num_shots = 1000  # Number of shots to simulate: 1000 for fig in paper, 20 to test
    num_repeaters = (
        1  # Number of repeaters (Alice and Bob, the end nodes, are counted separately)
    )
    elementary_link_quantum_delay = 1
    elementary_link_classical_delay = 0
    BSM_efficiency = 1 / 2
    BSM_loss = -10 * np.log10(BSM_efficiency)

    num_data_points = 8
    link_losses_in_db = list(np.linspace(0, 7, num_data_points))

    fidelities = []
    ent_gen_times = []

    for i, link_loss_in_db in enumerate(link_losses_in_db):
        # Get filename based on a uuid.
        uuid_string = str(uuid.uuid4())
        file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_" + uuid_string
        file_path = dir_path + "/" + file_name_string

        # Run simulations and save data to a CSV file
        simulation_parameters = {
            "random_seed": random_seed, # Seed to seed the simulation with
            "num_shots": num_shots, # Number of shots in the simulation
            "num_repeaters": num_repeaters, # Number of repeaters in the repeater chain
            "elementary_link_loss_in_db": link_loss_in_db
            + BSM_loss,  # Channel loss probability, in dB with added BSM loss, in dB.
            "elementary_link_quantum_delay": elementary_link_quantum_delay,  # Channel delay for qubits, in seconds
            "elementary_link_classical_delay": elementary_link_classical_delay,  # Channel delay for classical messages, in seconds
            "H_delay": 0,  # Hadamard delay, in seconds
            "X_delay": 0,  # X gate delay, in seconds
            "Z_delay": 0,  # Z gate delay, in seconds
            "CNOT_delay": 0,  # CNOT gate delay, in seconds
            "meas_delay": 0,  # Measurement delay, in seconds
            "bsm_delay": 0,  # Delay for BSM application, in seconds
            "depolarizing_prob": 0,  # Depolarizing probability per qubit
            "file_name": file_path, # File path name
        }

        fidelity_set, ent_gen_set = run_repeater_chain(**simulation_parameters)

        fidelities += [fidelity_set]
        ent_gen_times += [ent_gen_set]

    # Save compiled results
    simulation_results = copy.deepcopy(simulation_parameters)
    del simulation_parameters["elementary_link_loss_in_db"]
    simulation_results["link_losses_in_db"] = link_losses_in_db
    simulation_results["fidelities"] = fidelities
    simulation_results["ent_gen_times"] = ent_gen_times
    simulation_results["BSM_efficiency"] = BSM_efficiency

    with open(file_path + "_sim_results" + ".json", "w") as file:
        file.write(json.dumps(simulation_results, indent=2))
