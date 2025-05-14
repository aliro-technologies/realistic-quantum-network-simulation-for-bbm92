#!/bin/bash

# Bash script to run theory python scripts to generate json files
python comparison/make_comparison_plot.py -f comparison_results/2025-04-22/mar20_run2_single_count_per_bin -y qbers
python comparison/make_comparison_plot.py -f comparison_results/2025-04-22/mar20_run2_single_count_per_bin -y raw_key_rates

python comparison/make_error_plot.py -f comparison_results/2025-04-22/mar20_run2_single_count_per_bin -y qbers
python comparison/make_error_plot.py -f comparison_results/2025-04-22/mar20_run2_single_count_per_bin -y raw_key_rates

python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_dark_counts -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_deadtimes -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_jitter -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_loss -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_source_brightness -y secure_key_rates
python comparison/make_comparison_plot.py -f comparison_results/2025-04-25/mar20_run2_visibility -y secure_key_rates

