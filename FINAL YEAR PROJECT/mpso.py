import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
                if particle.fitness < particle.best_position.any():
                    particle.best_position = np.copy(particle.position)
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

# Sample objective function: Sphere function (minimization problem)
def sphere_function(x):
    return np.sum(x**2)

# Load dataset from CSV file with improved non-numerical handling
def load_data(file_name):
    df = pd.read_csv(file_name)
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    normalized_data = (df - df.min()) / (df.max() - df.min() + 1e-9)
    return normalized_data.values.tolist()

# Run E-SAPR PSO
file_name = "movies.csv"
data = load_data(file_name)
optimizer = E_SAPR_PSO(obj_func=sphere_function, data=data, max_iter=50)
best_position, best_fitness = optimizer.optimize()

print(f"\nFinal Best Position: {best_position}")
print(f"Final Best Fitness: {best_fitness}")
