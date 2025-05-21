#!/bin/bash

# Bash script to run Aliro Simulator python scripts to generate json files for BBM92 simulations
python simulation/run_sim_deadtime_sweep.py
python simulation/run_sim_jitter_sweep.py
python simulation/run_sim_loss_sweep.py
python simulation/run_sim_source_brightness_sweep.py
python simulation/run_sim_visibility_sweep.py
python simulation/run_sim_coinc_window_sweep.py
python simulation/run_sim_dark_counts_sweep.py
