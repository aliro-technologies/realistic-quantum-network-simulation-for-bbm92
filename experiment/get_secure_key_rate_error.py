import math
import numpy as np

def get_secure_key_rate_error(raw_key_rate, raw_key_rate_error, qber, qber_error):
    """
    This function returns the secure key rate error given a raw key rate, raw key rate error,
    QBER, and QBER error.
    
    For raw key rate :math:`k_r` and qber :math:`q`, the secure key rate :math:`k_s` is given by:
    :math:`k_s = k_r (1 - 2.1 H(q))`,

    where H is the binary entropy function,
    :math:`H(x) = -x\\log_2 x - (1 - x) \\log_2(1-x)`

    The secure key rate error :math:`\\Delta k_s` is calculated through standard propagation of
    error techniques. For a general function :math:`z = f(x, y, ...)`, the error :math:`\\Delta f` is given by
    :math:`(\\Delta z)^2 = \\left( \\frac{\\partial f}{x} \\right)^2 (\\Delta x)^2 + \\left( \\frac{\\partial f}{y} \\right)^2 (\\Delta y)^2 + ...`
    
    :param raw_key_rate: The raw key rate for BBM-92.
    :param raw_key_rate_error: The calculated error on the raw key rate for BBM-92.
    :param qber: The QBER for BBM-92.
    :param qber_error: The calculated error for the BBM-92 QBER.
    """
    kappa = np.sqrt((1/(np.log(2) * (1 - qber)))**2 * qber_error**2)
    gamma = np.sqrt((1/(np.log(2) * qber))**2 * qber_error**2)
    beta = - (1 - qber) * np.log2(1 - qber)
    delta_beta = beta * np.sqrt( (qber_error / (1 - qber))**2 + (kappa / np.log2(1 - qber))**2 )
    alpha = -qber * np.log2(qber)
    delta_alpha = alpha * np.sqrt( (qber_error / qber)**2 + (gamma / np.log2(qber))**2 )

    qber_entropy = alpha + beta
    delta_qber_entropy = np.sqrt( delta_alpha**2 + delta_beta**2 )

    A = 1 - 2.1 * qber_entropy
    delta_A = 2.1 * delta_qber_entropy

    secure_key_rate = raw_key_rate * ( 1 - 2.1 * qber_entropy)
    secure_key_rate_error = secure_key_rate * np.sqrt( (raw_key_rate_error / raw_key_rate)**2 + (delta_A / A)**2 )

    if math.isnan(secure_key_rate_error) or secure_key_rate <= 0:
        secure_key_rate_error = 0

    return float(np.abs(secure_key_rate_error))
