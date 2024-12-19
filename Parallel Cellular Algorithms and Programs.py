import numpy as np

def parallel_cellular_algorithm(objective_function, grid_size, bounds, max_iter, neighborhood="Moore"):
    rows, cols = grid_size
    lb, ub = bounds

    # Initialize grid with random positions
    grid = np.random.uniform(lb, ub, (rows, cols))
    fitness_grid = np.full((rows, cols), float("inf"))
    best_solution = None
    best_score = float("inf")

    def get_neighbors(row, col):
        if neighborhood == "Moore":
            # Include all 8 neighbors (diagonal + adjacent)
            neighbors = [
                ((row + dr) % rows, (col + dc) % cols)
                for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                if not (dr == 0 and dc == 0)
            ]
        elif neighborhood == "Von Neumann":
            # Include only up, down, left, right neighbors
            neighbors = [
                ((row + dr) % rows, (col + dc) % cols)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
        return neighbors

    # Main optimization loop
    for iteration in range(max_iter):
        for row in range(rows):
            for col in range(cols):
                # Evaluate current fitness
                cell_value = grid[row, col]
                fitness = objective_function(cell_value)
                fitness_grid[row, col] = fitness

                # Track the best solution
                if fitness < best_score:
                    best_score = fitness
                    best_solution = cell_value

                # Update state based on neighbors
                neighbors = get_neighbors(row, col)
                neighbor_values = [grid[r, c] for r, c in neighbors]
                grid[row, col] = np.clip(
                    np.mean(neighbor_values) + 0.1 * np.random.randn(), lb, ub
                )

    return best_solution, best_score

# Objective function (example: Sphere function)
def sphere_function(x):
    return x**2

# Usage example
grid_size = (5, 5)
bounds = (-10, 10)
max_iter = 100
best_solution, best_score = parallel_cellular_algorithm(sphere_function, grid_size, bounds, max_iter)
print(f"Best Solution: {best_solution}, Best Score: {best_score}")

