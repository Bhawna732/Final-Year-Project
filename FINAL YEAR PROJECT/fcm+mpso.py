import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import davies_bouldin_score

# Function to calculate XB index
def calculate_xb(data, cntr, u):
    dist = np.linalg.norm(data[:, np.newaxis] - cntr, axis=2)
    cluster_distances = np.min(dist, axis=1)
    cluster_diameters = np.max(dist, axis=1)
    xb_index = np.sum((cluster_distances ** 2) * np.max(u, axis=0)) / np.sum(cluster_diameters ** 2)
    return xb_index

# Function to compute DB index
def calculate_db(data, cntr, u):
    labels = np.argmax(u, axis=0)
    return davies_bouldin_score(data, labels)

# Function to compute entropy (Shannon entropy-based diversity measure)
def compute_entropy(particles):
    if len(particles) == 0:
        return 0
    min_val, max_val = np.min(particles), np.max(particles)
    if min_val == max_val:
        return 0
    norm_particles = (particles - min_val) / (max_val - min_val + 1e-9)
    histogram, _ = np.histogram(norm_particles, bins=10, density=True)
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-9))
    return entropy

# Particle class for PSO
class Particle:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(len(position))])
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')
        self.fitness = float('inf')

# PSO with Entropy-Based Adaptive Population Reduction
class E_SAPR_PSO:
    def __init__(self, obj_func, data, max_iter=50, min_size=5):
        self.obj_func = obj_func
        self.data = data
        self.dim = len(data[0]) if len(data) > 0 else 0
        self.max_iter = max_iter
        self.initial_size = len(data)
        self.min_size = min_size
        self.population = [Particle(position) for position in data]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def optimize(self):
        for t in range(self.max_iter):
            for particle in self.population:
                particle.fitness = self.obj_func(particle.position)
                if particle.fitness < particle.best_fitness:
                    particle.best_position = np.copy(particle.position)
                    particle.best_fitness = particle.fitness
                if particle.fitness < self.global_best_fitness:
                    self.global_best_position = np.copy(particle.position)
                    self.global_best_fitness = particle.fitness
            entropy = compute_entropy(np.array([p.position[0] for p in self.population]))
            new_size = max(self.min_size, int(self.initial_size * (entropy / np.log2(self.initial_size + 1e-9))))
            self.population.sort(key=lambda p: p.fitness)
            self.population = self.population[:new_size]
            for particle in self.population:
                inertia = 0.7
                cognitive = 1.5 * random.random()
                social = 1.5 * random.random()
                new_velocity = (
                    inertia * particle.velocity +
                    cognitive * (particle.best_position - particle.position) +
                    social * (self.global_best_position - particle.position)
                )
                particle.velocity = new_velocity
                particle.position += new_velocity
            print(f"Iteration {t+1}: Entropy = {entropy:.4f}, Population Size = {len(self.population)}")
        return self.global_best_position, self.global_best_fitness

# Sphere function (objective function)
def sphere_function(x):
    return np.sum(x**2)

# Load dataset from CSV file with non-numerical handling
def load_data(file_name):
    df = pd.read_csv(file_name)
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    normalized_data = (df - df.min()) / (df.max() - df.min() + 1e-9)
    normalized_data = normalized_data.fillna(0)
    return normalized_data.values

# Fuzzy C-Means clustering
def fuzzy_c_means(data, n_clusters, m=2, max_iter=150, error=1e-5):
    n_samples, _ = data.shape
    u = np.random.rand(n_clusters, n_samples)
    u /= np.sum(u, axis=0)
    for iteration in range(max_iter):
        um = u ** m
        cntr = (um @ data) / np.sum(um, axis=1, keepdims=True)
        distances = np.linalg.norm(data[:, np.newaxis] - cntr, axis=2).T
        distances = np.nan_to_num(distances, nan=1e10)
        inv_distances = 1.0 / (distances + 1e-10)
        inv_distances_m = inv_distances ** (2 / (m - 1))
        u_new = inv_distances_m / np.sum(inv_distances_m, axis=0, keepdims=True)
        u_new = np.nan_to_num(u_new, nan=0)
        if np.linalg.norm(u_new - u) < error:
            break
        u = u_new
    return cntr, u, distances

file_name = "movies.csv"
data = load_data(file_name)
optimizer = E_SAPR_PSO(obj_func=sphere_function, data=data, max_iter=50)
best_position, best_fitness = optimizer.optimize()
print(f"\nFinal Best Position: {best_position}")
print(f"Final Best Fitness: {best_fitness}")

n_clusters = 3
cntr, u, distances = fuzzy_c_means(data, n_clusters)
xb_index = calculate_xb(data, cntr, u)
db_index = calculate_db(data, cntr, u)
print(f"XB Index: {xb_index}")
print(f"DB Index: {db_index}")

plt.scatter(data[:, 0], data[:, 1], c=np.argmax(u, axis=0), cmap='viridis', alpha=0.5)
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
plt.title('Fuzzy C-Means Clustering')
plt.legend()
plt.show()
