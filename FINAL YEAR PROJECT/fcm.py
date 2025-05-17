import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from typing import Optional

class FuzzyCMeans:
    def __init__(self, n_clusters: int = 3, m: float = 2, max_iter: int = 150,
                 error: float = 1e-5, random_state: Optional[int] = None):
        self._validate_parameters(n_clusters, m, max_iter, error)
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.centers_ = None
        self.labels_ = None
        self.membership_ = None
        self.n_iter_ = 0

        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    def _validate_parameters(n_clusters: int, m: float, max_iter: int, error: float):
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer >= 2")
        if m <= 1:
            raise ValueError("Fuzziness coefficient (m) must be greater than 1")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if error <= 0:
            raise ValueError("error must be positive")

    def _initialize_membership_matrix(self, n_samples: int) -> np.ndarray:
        u = np.random.rand(self.n_clusters, n_samples)
        return u / np.sum(u, axis=0)

    def _update_centers(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        um = membership ** self.m
        return (um @ data) / np.sum(um, axis=1, keepdims=True)

    def _update_membership(self, distances: np.ndarray) -> np.ndarray:
        power = 2 / (self.m - 1)
        inv_distances = 1.0 / np.maximum(distances, np.finfo(float).eps)
        inv_distances_m = inv_distances ** power
        return inv_distances_m / np.sum(inv_distances_m, axis=0, keepdims=True)

    def fit(self, X: np.ndarray) -> 'FuzzyCMeans':
        X = np.asarray(X)
        n_samples, _ = X.shape
        self.membership_ = self._initialize_membership_matrix(n_samples)

        for iteration in range(self.max_iter):
            old_membership = self.membership_.copy()
            self.centers_ = self._update_centers(X, self.membership_)
            distances = cdist(self.centers_, X)
            self.membership_ = self._update_membership(distances)
            if np.linalg.norm(self.membership_ - old_membership) < self.error:
                break
            self.n_iter_ = iteration + 1

        self.labels_ = np.argmax(self.membership_, axis=0)
        return self

    def xie_beni_index(self, X: np.ndarray) -> float:
        distances = cdist(X, self.centers_) ** 2
        numerator = np.sum((self.membership_.T ** 2) * distances)
        center_distances = cdist(self.centers_, self.centers_)
        np.fill_diagonal(center_distances, np.inf)
        min_separation = np.min(center_distances) ** 2
        return numerator / (X.shape[0] * min_separation)

    def davies_bouldin_index(self, X: np.ndarray) -> float:
        cluster_scatters = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.any(mask):
                cluster_scatters[i] = np.mean(np.linalg.norm(X[mask] - self.centers_[i], axis=1))
        max_ratios = []
        for i in range(self.n_clusters):
            ratios = []
            for j in range(self.n_clusters):
                if i != j:
                    separation = np.linalg.norm(self.centers_[i] - self.centers_[j])
                    if separation > 0:
                        ratio = (cluster_scatters[i] + cluster_scatters[j]) / separation
                        ratios.append(ratio)
            if ratios:
                max_ratios.append(max(ratios))
        return np.mean(max_ratios) if max_ratios else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = cdist(self.centers_, X)
        memberships = self._update_membership(distances)
        return np.argmax(memberships, axis=0)


def main():
    file_name = input("Enter the CSV file name (with extension): ")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Please check the file name and try again.")
        return

    data = df.select_dtypes(include=[np.number]).values
    data = np.nan_to_num(data, nan=np.nanmedian(data))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    fcm = FuzzyCMeans(n_clusters=3, random_state=42)
    fcm.fit(data_scaled)

    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=fcm.labels_, cmap='viridis')
    plt.scatter(fcm.centers_[:, 0], fcm.centers_[:, 1], color='red', marker='X')
    print(f"Validation Indices for {file_name}:")
    print(f"XB Index: {fcm.xie_beni_index(data_scaled):.4f}")
    print(f"DB Index: {fcm.davies_bouldin_index(data_scaled):.4f}")

    plt.title('Fuzzy C-Means Clustering')
    plt.show()

if __name__ == "__main__":
    main()
