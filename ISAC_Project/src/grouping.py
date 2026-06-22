import numpy as np
from sklearn.cluster import KMeans

def sbl_grouping(cascaded_channel,
                 Q=250,
                 max_iter=50):

    M = len(cascaded_channel)

    # Feature vector

    X = np.column_stack([
        np.abs(cascaded_channel),
        np.angle(cascaded_channel)
    ])

    # Initial variance
    gamma = np.ones(M)

    for _ in range(max_iter):

        Gamma = np.diag(gamma)

        mu = Gamma @ np.linalg.pinv(Gamma + np.eye(M))

        gamma_new = np.real(np.diag(mu))

        if np.linalg.norm(gamma_new-gamma) < 1e-5:
            break

        gamma = gamma_new

    # dominant elements

    importance = gamma

    # cluster according to importance

    features = np.column_stack([
        importance,
        np.angle(cascaded_channel)
    ])

    kmeans = KMeans(
        n_clusters=Q,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(features)

    groups = []

    for q in range(Q):

        groups.append(
            np.where(labels == q)[0]
        )

    return groups

def grouped_channel(
        h_BI,
        h_IU,
        groups):

    h_eff = 0

    for idx in groups:

        cascaded = h_BI[idx] * h_IU[idx]

        phase = np.exp(
            -1j*np.angle(
                np.sum(cascaded)
            )
        )

        h_eff += phase*np.sum(cascaded)

    return h_eff