import numpy as np

def gwo(objective_function, dim, bounds, num_wolves=10, max_iter=100):
    # Initialize parameters
    lb, ub = bounds
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")

    # Main optimization loop
    for iteration in range(max_iter):
        # Evaluate fitness of all wolves
        for wolf in wolves:
            fitness = objective_function(wolf)
            if fitness < alpha_score:
                alpha_score, alpha = fitness, wolf.copy()
            elif fitness < beta_score:
                beta_score, beta = fitness, wolf.copy()
            elif fitness < delta_score:
                delta_score, delta = fitness, wolf.copy()

        # Update positions of wolves
        a = 2 - iteration * (2 / max_iter)  # Decreasing factor
        for i, wolf in enumerate(wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * alpha - wolf)
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = abs(C2 * beta - wolf)
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = abs(C3 * delta - wolf)
            X3 = delta - A3 * D_delta

            # Update position
            wolves[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)

    # Return the best solution
    return alpha, alpha_score

# Objective function (example: sphere function)
def sphere_function(x):
    return sum(x**2)

# Usage example
dim = 5
bounds = (-10, 10)
best_position, best_score = gwo(sphere_function, dim, bounds)
print("Ganashree C M-1BM22CS097")
print(f"Best Position: {best_position}, Best Score: {best_score}")
