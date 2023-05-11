import random
import numpy as np
import cv2

hi = 1
lo = 0

def generate_numbers(N, lo, hi):
    return [random.uniform(lo, hi) for _ in range(N)]

def converge_numbers(numbers, target, convergence_factor):

    converged_numbers = []

    for number in numbers:
        pseudo_random_shift = random.uniform(0, convergence_factor)

        if number > target:
            new_number = number - pseudo_random_shift
        elif number < target:
            new_number = number + pseudo_random_shift
        elif number - convergence_factor == target:
            new_number = target
        elif number + convergence_factor == target:
            new_number = target

        converged_numbers.append(new_number)

    return converged_numbers

# Example usage
target = 0.5
N = 64
random_numbers = generate_numbers(N, 0, 1)
matrix = np.empty((0, N), int)

for i in range(1, round(N+1/10)): #rather than loop, call each routine
    converged_numbers = converge_numbers(random_numbers, target, convergence_factor=0.05**(i/N))
    matrix = np.append(matrix, [converged_numbers], axis=0)
    random_numbers = converged_numbers

cv2.imshow('staircase', matrix)
cv2.waitKey(0)
