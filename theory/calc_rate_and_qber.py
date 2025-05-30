# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import numpy as np
from scipy.special import erf


def calc_rate_and_qber(
    fractional_loss_A,
    fractional_loss_B,
    detector_efficiency,
    detector_dead_time,
    coincidence_window,
    rate_source,
    dark_count_rate_A,
    dark_count_rate_B,
    probability_optical_error,
    num_detectors,
    detection_resolution,
):
    """
    Return the coincidence count rate, secret key rate, and the QBER for BBM-92 with
    supplied parameters. Uses equations adapted from Neumann et al, PRA 104 2021.

    :param fractional_loss_A: Loss factor from Alice's link.
    :param fractional_loss_B: Loss factor from Bob's link.
    :param detector_efficiency: Detector efficiency (fraction).
    :param detector_dead_time: Detector dead time (in seconds).
    :param coincidence_window: The minimum resolution of the FPGA or timetagger (seconds).
    :param rate_source: The brightness (coincidence pairs per second) of the source.
    :param dark_count_rate_A: The total dark count rate for Alice, in counts per second.
        Note here this will be 4 * the dark count per detector.
    :param dark_count_rate_B: The total dark count rate for Bob, in counts per second.
        Note here this will be 4 * the dark count per detector.
    :param probability_optical_error: The probability of optical error.
    :param num_detectors: The number of detectors in the setup per communication partner.
    :param detection_resolution: The FWHM of the detection resolution, including
        a convolution of detector jitter, chromatic dispersion and coherence time (seconds).
    """

    # Efficiency due to dead time
    dead_time_efficiency_A = 1 / (
        1
        + (
            (rate_source * detector_efficiency * fractional_loss_A + dark_count_rate_A)
            * detector_dead_time
        )
        / num_detectors
    )
    dead_time_efficiency_B = 1 / (
        1
        + (
            (rate_source * detector_efficiency * fractional_loss_B + dark_count_rate_B)
            * detector_dead_time
        )
        / num_detectors
    )

    # Find singles rates
    singles_rate_A = (
        fractional_loss_A * detector_efficiency * dead_time_efficiency_A * rate_source
    )
    singles_rate_B = (
        fractional_loss_B * detector_efficiency * dead_time_efficiency_B * rate_source
    )

    # Find proportion of true coincidences which will fall into the true coincidence window
    resolution_efficiency = erf(
        np.sqrt(np.log(2)) * coincidence_window / detection_resolution
    )

    # Find coincidence rate and key rate
    true_coinc_rate = (
        fractional_loss_A
        * fractional_loss_B
        * detector_efficiency**2
        * dead_time_efficiency_A
        * dead_time_efficiency_B
        * rate_source
        * resolution_efficiency
    )

    # Find error rate due to errors in the optics
    opt_error_rate = probability_optical_error * true_coinc_rate

    # Find error rate due to accidental coincidences (includes dark counts)
    measured_singles_rate_A = singles_rate_A + dark_count_rate_A
    measured_singles_rate_B = singles_rate_B + dark_count_rate_B

    # Probability of at least one detection in one coincidence window for Alice
    # times probability of at least one detection in one coincidence window for Bob, per coincidence window
    acc_error_rate = (
        (1 - np.exp(-measured_singles_rate_A * coincidence_window))
        * (1 - np.exp(-measured_singles_rate_B * coincidence_window))
        / coincidence_window
    )

    # Approx half of all accidental coincidences will be wrong, half will be right
    # (Different from sifting)
    total_error_rate = opt_error_rate + 0.5 * acc_error_rate
    meas_coinc_rate = true_coinc_rate + acc_error_rate
    qber = total_error_rate / meas_coinc_rate

    # Multiply by 0.5 for sifting step
    raw_meas_key_rate = 0.5 * meas_coinc_rate
    non_acc_key_rate = 0.5 * true_coinc_rate

    print(f"QBER: {qber}")
    print(f"Raw measured key rate: {raw_meas_key_rate}")
    print(f"Non-accidentals key rate: {non_acc_key_rate}")

    # Some of the key is error and is used to correct the error
    binary_entropy = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
    secure_key_rate = max(raw_meas_key_rate * (1 - 2.1 * binary_entropy), 0)

    return raw_meas_key_rate, secure_key_rate, qber
