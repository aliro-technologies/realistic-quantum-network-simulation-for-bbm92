#!/bin/bash

# Run python scripts to plot coincidence histograms from saved experimental data
python experiment/make_histograms.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm

# Run python scripts to plot correlation matrix for probabilities of measuring each outcome in each basis (X, Z) from saved experimental data
python experiment/make_correlation_matrix.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm

# Find key rates and QBERs for coincidence window = 2 ns for saved experimental data
python experiment/analyze_data.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm

# Sweep over coincidence windows and save key rates
python experiment/run_exp_coinc_window_sweep.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -dca 500 -dcb 1800

# Save experimental data (coincidence window = 2 ns) with parameters estimated from experimental data and with equipment data sheet parameters
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn dark_count_rate -pv 1150
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn detector_dead_time -pv 4.5e-8
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn detection_resolution -pv 6.9e-10
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn loss_per_link -pv 12.0
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn source_pair_rate -pv 1500000.0
python experiment/run_exp_single_point.py -f Mar_20_RUN_2_0P8vTHRESHOLD_BBM92_TIMETAG_STREAM_16HIST_UBENCHPC_12345678_DAHVDAHV_bin_width_1ns_4_36pm -pn visibility -pv 0.94
