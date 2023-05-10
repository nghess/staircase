import random
import numpy as np
import cv2

def generate_numbers(N):
    return [random.uniform(1, 2) for _ in range(N)]


def converge_numbers(numbers, convergence_factor=0.5, bounds=(1, 2)):
    min_value, max_value = bounds
    center = (min_value + max_value) / 2
    converged_numbers = []

    for number in numbers:
        pseudo_random_shift = random.uniform(-convergence_factor, convergence_factor)
        new_number = number + (center - number) * pseudo_random_shift
        new_number = max(min(new_number, max_value), min_value)  # Ensure the number stays within bounds
        converged_numbers.append(new_number)

    return converged_numbers


# Example usage
N = 100
random_numbers = generate_numbers(N)
matrix = np.empty((0,N), int)
#print("Random numbers:", random_numbers)

for i in range(N):
    converged_numbers = converge_numbers([random_numbers])
    matrix = np.append(matrix, converged_numbers, axis=0)

#print("Converged numbers:", converged_numbers)
