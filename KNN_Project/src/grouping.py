import numpy as np

def knn_grouping(M, Q=250):
    """
    Groups adjacent elements (legacy KNN grouping).
    M: total number of IRS elements
    Q: number of groups
    """
    L = M // Q
    groups = []
    for q in range(Q):
        groups.append(np.arange(q*L, (q+1)*L))
    return groups

def grouped_channel(h_BI, h_IU, groups):
    """
    Phase alignment for grouped elements.
    """
    h_eff = 0
    for idx in groups:
        cascaded = h_BI[idx] * h_IU[idx]
        phase = np.exp(-1j * np.angle(np.sum(cascaded)))
        h_eff += phase * np.sum(cascaded)
    return h_eff