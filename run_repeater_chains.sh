#!/bin/bash

# Bash script to run Aliro Simulator python scripts to generate json files for repeater chain simulations
python repeater_chains/run_repeater_chain_loss_sweep.py
python repeater_chains/run_repeater_chain_visibility_sweep.py


# Plot previously run simulations
python repeater_chains/plot_rate_vs_loss.py -f repeater_chain_results/single_node_simulation/15_01_11_84beabc6-5c12-4166-9d99-c13629272886_sim_results
python repeater_chains/plot_key_vs_visibility.py -f repeater_chain_results/visibility_sweep