#!/bin/bash

# Bash script to run theory python scripts to generate json files
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_coinc_window -y qbers
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_coinc_window -y raw_key_rates

python comparison/make_error_plot.py -f comparison_results/mar20_run2_coinc_window_errors -y qbers
python comparison/make_error_plot.py -f comparison_results/mar20_run2_coinc_window_errors -y raw_key_rates

python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_dark_counts -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_deadtimes -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_jitter -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_loss -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_source_brightness -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/mar20_run2_visibility -y secure_key_rates

