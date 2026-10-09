import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Problem 1: Two-site LCAO with finite overlap

# Question 1

# Given model parameters
epsilon_s = -5.34
h = -1.80
s = 0.18

# Construct the Hamiltonian matrix H
H = np.array([
    [epsilon_s, h],
    [h, epsilon_s]
], dtype=float)

# Construct the overlap matrix S
S = np.array([
    [1.0, s],
    [s, 1.0]
], dtype=float)

# Solve the generalized eigenvalue problem H c = E S c
energies, C = eigh(H, S)

print("Question 1")
print("Energies (eV):")
print(energies)
print("Eigenvectors:")
print(C)

# Question 2

# Extract the two eigenvectors
c1 = C[:, 0]
c2 = C[:, 1]

# Check the S-normalization of each eigenvector
print("\nQuestion 2")
print("c1^T S c1 =")
print(c1.T @ S @ c1)

print("c2^T S c2 =")
print(c2.T @ S @ c2)

# Question 3

# Compute the analytic energies for comparison
E_plus = (epsilon_s + h) / (1 + s)
E_minus = (epsilon_s - h) / (1 - s)

print("\nQuestion 3")
print("Analytic E_plus (eV):")
print(E_plus)

print("Analytic E_minus (eV):")
print(E_minus)

print("Symmetric bonding eigenvector:")
print(c1)

print("Antisymmetric antibonding eigenvector:")
print(c2)

# Question 4

# Repeat the calculation using the orthogonal approximation S = I
energies_orth = np.linalg.eigvalsh(H)

print("\nQuestion 4")
print("Orthogonal-approximation energies (eV):")
print(energies_orth)

# Calculate the energy shifts caused by finite overlap
shifts = energies - energies_orth

print("Energy shifts due to overlap (eV):")
print(shifts)

# Question 5

# Sweep the overlap parameter from s = 0.00 to s = 0.40
s_values = np.linspace(0.0, 0.40, 201)

# Evaluate the symmetric and antisymmetric energy branches
E_plus_values = (epsilon_s + h) / (1 + s_values)
E_minus_values = (epsilon_s - h) / (1 - s_values)

# Plot both energy branches as functions of overlap
plt.figure()

plt.plot(s_values, E_plus_values, label="Symmetric state")
plt.plot(s_values, E_minus_values, label="Antisymmetric state")

# Mark the assigned overlap value s = 0.18
plt.axvline(0.18, linestyle="--", label="s = 0.18")

plt.xlabel("Overlap s")
plt.ylabel("Energy (eV)")
plt.legend()
plt.tight_layout()

# Save the required 600-DPI figure
plt.savefig("problem1_question5.png", dpi=600)
plt.close()

# Check

# Final numerical check of S-orthonormality
print("\nCheck")
print("C^T S C =")
print(C.T @ S @ C)
