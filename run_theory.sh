#!/bin/bash

# Bash script to run theory python scripts to generate json files
python theory/run_theory_dark_count_sweep.py
python theory/run_theory_dead_time_sweep.py
python theory/run_theory_detector_jitter_sweep.py
python theory/run_theory_loss_sweep.py
python theory/run_theory_source_brightness_sweep.py
python theory/run_theory_visibility_sweep.py
python theory/run_theory_coinc_window_sweep.py