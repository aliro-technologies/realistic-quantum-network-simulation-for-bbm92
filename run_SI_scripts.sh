#!/bin/bash

# Bash script to run SeQUeNCe script to reproduce plot in the SI
python SI_sequence_elementary_link/run_two_node_sequence_ent_gen.py

# Plot previously run simulations
python SI_sequence_elementary_link/plot_rate_vs_loss_and_sequence.py -f SI_sequence_elementary_link/01_56_24_fc6b19a6-890f-4ceb-82cf-eda83243e86c_sequence_SI_sim_results_elementary_link
