# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
import math
import numpy as np


def get_secure_key_rate_error(raw_key_rate, raw_key_rate_error, qber, qber_error):
    """
    This function returns the predicted secure key rate error given a raw key rate, raw key rate error,
    QBER, and QBER error.

    For raw key rate :math:`k_r` and qber :math:`q`, the secure key rate :math:`k_s` is given by:
    :math:`k_s = k_r (1 - 2.1 H(q))`,

    where H is the binary entropy function,
    :math:`H(x) = -x\\log_2 x - (1 - x) \\log_2(1-x)`.

    The secure key rate error :math:`\\Delta k_s` is calculated through propagation of
    error. For a general function :math:`z = f(x, y, ...)`, the error :math:`\\Delta f` is given by
    :math:`(\\Delta z)^2 = \\left( \\frac{\\partial f}{x} \\right)^2 (\\Delta x)^2 + \\left( \\frac{\\partial f}{y} \\right)^2 (\\Delta y)^2 + ...`

    :param raw_key_rate: The raw key rate for BBM-92.
    :param raw_key_rate_error: The calculated error on the raw key rate for BBM-92.
    :param qber: The QBER for BBM-92.
    :param qber_error: The calculated error for the BBM-92 QBER.
    """
    if qber == 1 or qber == 0:
        # This means there were not enough statistics to determine QBER accurately
        qber_entropy = float("nan")
        delta_qber_entropy = 0
    else:
        qber_entropy = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)
        delta_qber_entropy = np.abs(np.log2((1 - qber) / qber)) * qber_error

    A = 1 - 2.1 * qber_entropy
    delta_A = 2.1 * delta_qber_entropy

    secure_key_rate = raw_key_rate * (1 - 2.1 * qber_entropy)
    secure_key_rate_error = secure_key_rate * np.sqrt(
        (raw_key_rate_error / raw_key_rate) ** 2 + (delta_A / A) ** 2
    )

    if math.isnan(secure_key_rate_error) or secure_key_rate <= 0:
        secure_key_rate_error = 0

    return float(np.abs(secure_key_rate_error))
