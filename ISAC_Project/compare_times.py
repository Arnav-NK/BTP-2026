import time
import numpy as np
from sklearn.cluster import KMeans
import os

from src.grouping import sbl_grouping

def knn_grouping(cascaded_channel, Q=250):
    """
    Standard grouping using K-Means directly on features (often referred to as KNN-based in some literature)
    """
    features = np.column_stack([
        np.abs(cascaded_channel),
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
        groups.append(np.where(labels == q)[0])
        
    return groups

def main():
    dataset_path = 'sbl_input_dataset.npz'
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run generate_sbl_dataset.py first.")
        return
        
    data = np.load(dataset_path)
    cascaded_channel = data['cascaded_channel']
    
    Q = 250
    
    print("Running SBL + K-Means Grouping...")
    start_time = time.time()
    sbl_groups = sbl_grouping(cascaded_channel, Q=Q)
    sbl_time = time.time() - start_time
    print(f"SBL + K-Means took: {sbl_time:.4f} seconds")
    
    print("\nRunning KNN (K-Means) Grouping...")
    start_time = time.time()
    knn_groups = knn_grouping(cascaded_channel, Q=Q)
    knn_time = time.time() - start_time
    print(f"KNN took: {knn_time:.4f} seconds")
    
    time_diff = sbl_time - knn_time
    print(f"\nTime Difference (SBL - KNN): {time_diff:.4f} seconds")
    
    # Save the time difference
    with open("time_comparison_results.txt", "w") as f:
        f.write(f"SBL + K-Means Time: {sbl_time:.4f} seconds\n")
        f.write(f"KNN Time: {knn_time:.4f} seconds\n")
        f.write(f"Time Difference (SBL - KNN): {time_diff:.4f} seconds\n")
    print("Results saved to time_comparison_results.txt")

if __name__ == "__main__":
    main()
