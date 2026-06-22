import numpy as np

def generate_channels(M):

    h_BI = (np.random.randn(M) +
            1j*np.random.randn(M))/np.sqrt(2)

    h_IU = (np.random.randn(M) +
            1j*np.random.randn(M))/np.sqrt(2)

    return h_BI, h_IU


def cascaded_channel(h_BI, h_IU):

    return h_BI * h_IU