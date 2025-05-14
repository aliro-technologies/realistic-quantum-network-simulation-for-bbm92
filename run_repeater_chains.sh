#!/bin/bash

# Bash script to run Aliro Simulator python scripts to generate json files for repeater chain simulations
#python repeater_chains/run_repeater_chain_loss_sweep.py
python repeater_chains/run_repeater_chain_visibility_sweep.py


# Plot previously run simulations
#python repeater_chains/plot_key_vs_visibility.py -f repeater_chain_results/visibility_sweep
#python repeater_chains/plot_rate_vs_loss.py -f repeater_chain_results/single_node_simulation/21_29_18_b2886cac-0af9-4606-bb49-fb7b48af4efd_sim_results
