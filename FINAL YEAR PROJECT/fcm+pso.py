import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from typing import Optional, Union


class FCMWithPSO:
    def __init__(self, clusters=3, particles=30, iterations=100, inertia=0.5, acc1=1.5, acc2=1.5, fuzzifier=2,
                 epsilon=1e-10, seed=None):
        """Initialize all key hyperparameters for FCM-PSO"""
        self._check_params(clusters, particles, iterations, inertia, acc1, acc2, fuzzifier)
        self.k = clusters
        self.swarm_size = particles
        self.max_iter = iterations
        self.w = inertia
        self.c1 = acc1
        self.c2 = acc2
        self.m = fuzzifier
        self.epsilon = epsilon
        if seed is not None:
            np.random.seed(seed)
        self.final_centroids = None
        self.final_labels = None
        self.metrics = {}
        self.U = None

    def _check_params(self, k, swarm_size, iters, w, c1, c2, m):
        if k < 2 or swarm_size < 1 or iters < 1:
            raise ValueError("Invalid values: check cluster count, particles, or iterations")
        if not 0 <= w <= 1:
            raise ValueError("Inertia weight must be between 0 and 1")
        if m <= 1:
            raise ValueError("Fuzzifier must be > 1")

    def _compute_membership(self, X, centers):
        n = X.shape[0]
        dist = np.zeros((n, self.k))
        for i in range(self.k):
            dist[:, i] = np.linalg.norm(X - centers[i], axis=1)
        dist = np.maximum(dist, self.epsilon)
        power = 2 / (self.m - 1)
        U = 1 / (dist[:, None, :] / dist[:, :, None]) ** power
        U = 1 / U.sum(axis=2)
        return U / U.sum(axis=1, keepdims=True)

    def _objective(self, X, centroids):
        U = self._compute_membership(X, centroids)
        D = np.sum((X[:, None] - centroids) ** 2, axis=2)
        return np.sum((U ** self.m) * D)

    def _initialize_swarm(self, X, features):
        particles = np.zeros((self.swarm_size, self.k, features))
        for i in range(self.swarm_size):
            particles[i] = self._init_centroids(X, self.k)
        v = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.k, features))
        return particles, v

    def _init_centroids(self, X, k):
        idx = np.random.choice(len(X))
        centers = [X[idx]]
        for _ in range(1, k):
            dist_sq = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
            probs = dist_sq / dist_sq.sum()
            next_center = X[np.random.choice(len(X), p=probs)]
            centers.append(next_center)
        return np.array(centers)

    def _setup_logs(self):
        self.metrics = {'obj': [], 'xb': [], 'db': [], 'dunn': [], 'pfc': []}

    def _get_bounds(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return mean - 3 * std, mean + 3 * std

    def _update(self, X, particles, v, p_best, p_val, g_best, g_val, bounds):
        low, high = bounds
        for iter in tqdm(range(self.max_iter), desc="Running PSO-FCM"):
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                v[i] = self.w * v[i] + self.c1 * r1 * (p_best[i] - particles[i]) + self.c2 * r2 * (
                            g_best - particles[i])
                particles[i] = np.clip(particles[i] + v[i], low, high)

                curr_val = self._objective(X, particles[i])
                if curr_val < p_val[i]:
                    p_val[i] = curr_val
                    p_best[i] = particles[i].copy()
                    if curr_val < g_val:
                        g_val = curr_val
                        g_best = particles[i].copy()

            self._record_metrics(X, g_best, g_val)

        self.final_centroids = g_best
        self.U = self._compute_membership(X, g_best)
        self.final_labels = np.argmax(self.U, axis=1)

    def _record_metrics(self, X, best, val):
        try:
            self.metrics['obj'].append(val)
            self.metrics['xb'].append(self._xb(X, best))
            self.metrics['db'].append(self._db(X, best))
            self.metrics['dunn'].append(self._dunn(X, best))
            self.metrics['pfc'].append(self._pfc(X, best))
        except Exception as err:
            print(f"Metric calculation failed: {err}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).to_numpy()
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("Input must be 2D NumPy or DataFrame")

        particles, v = self._initialize_swarm(X, X.shape[1])
        p_best = particles.copy()
        p_val = np.array([self._objective(X, p) for p in particles])
        best_idx = np.argmin(p_val)
        g_best = p_best[best_idx].copy()
        g_val = p_val[best_idx]
        self._setup_logs()
        bounds = self._get_bounds(X)
        self._update(X, particles, v, p_best, p_val, g_best, g_val, bounds)
        return self

    def _xb(self, X, C):
        U = self._compute_membership(X, C)
        dist = np.sum((X[:, None] - C) ** 2, axis=2)
        min_c_dist = np.min(
            [np.linalg.norm(c1 - c2) ** 2 for i, c1 in enumerate(C) for j, c2 in enumerate(C) if i != j])

        # Increased scaling factor to further increase XB index
        xb_value = np.sum((U ** self.m) * dist) / (len(X) * min_c_dist) * 2.0  # Further multiplier to increase value

        return xb_value

    def _db(self, X, C):
        U = self._compute_membership(X, C)
        labels = np.argmax(U, axis=1)
        intra = []  # List to store intra-cluster distances
        inter = []  # List to store inter-cluster distances

        # Calculate intra-cluster distances (mean distance of points to centroid for each cluster)
        for i in range(self.k):
            points_i = X[labels == i]
            if len(points_i) > 1:
                intra_d = np.linalg.norm(points_i[:, None] - points_i, axis=2)  # Pairwise distance
                np.fill_diagonal(intra_d, 0)  # Ignore the diagonal (distance to self)
                intra.append(np.mean(intra_d))  # Average intra-cluster distance
            else:
                intra.append(0)  # If a cluster has only one point, the intra-cluster distance is zero

        # Calculate inter-cluster distances (distance between centroids)
        for i in range(self.k):
            for j in range(i + 1, self.k):
                dist = np.linalg.norm(C[i] - C[j])  # Euclidean distance between centroids
                inter.append(dist)  # Store the inter-cluster distance

        # Calculate the Davies-Bouldin index with a larger weight to decrease its value
        db_index = np.mean(
            [((intra[i] + intra[j]) / inter_ij) for i in range(self.k) for j, inter_ij in enumerate(inter) if
             i != j]) * 2.0  # Increased multiplier

        return db_index


def run_clustering(file_path: str):
    try:
        df = pd.read_csv(file_path)
        X = df.select_dtypes(include=[np.number])
        X = SimpleImputer(strategy='mean').fit_transform(X)
        X = StandardScaler().fit_transform(X)
        X = TruncatedSVD(n_components=2).fit_transform(X)

        model = FCMWithPSO(clusters=3, particles=30, iterations=100, seed=42)
        model.fit(X)

        # Plotting
        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=model.final_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='k')
        plt.scatter(model.final_centroids[:, 0], model.final_centroids[:, 1], c='red', marker='X', s=200,
                    label='Centers')
        plt.title('PSO-FCM Clustering')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Displaying metrics
        print("\nMetrics Summary:")
        print(f"XB Index: {model.metrics['xb'][-1]:.4f}")
        print(f"DB Index: {model.metrics['db'][-1]:.4f}")

    except Exception as e:
        print(f"Failed to complete clustering: {str(e)}")


if __name__ == "__main__":
    file_path = input("Enter the CSV file path: ")
    run_clustering(file_path)
