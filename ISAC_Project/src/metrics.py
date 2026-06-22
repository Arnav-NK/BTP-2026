import numpy as np

def ber(bits,
        detected_bits):

    return np.mean(
        bits != detected_bits
    )