import numpy as np

def generate_sefdm_matrix(
        N,
        alpha):

    n = np.arange(N).reshape(-1,1)

    k = np.arange(N)

    F = np.exp(
        1j*2*np.pi*n*k*alpha/N
    )/np.sqrt(N)

    return F


def mmse_detector(
        y,
        F,
        h,
        noise_var):

    gram = F.conj().T @ F

    snr = abs(h)**2/noise_var

    A = gram + (1/snr)*np.eye(F.shape[1])

    b = F.conj().T @ (y/h)

    return np.linalg.solve(A,b)