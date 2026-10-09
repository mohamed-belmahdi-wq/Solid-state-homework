import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# Define the Rayleigh-Ritz basis function
def b(j, x):
    return x**(j + 1) * (1 - x)


# Derivative of the basis function
def db(j, x):
    return (j + 1) * x**j - (j + 2) * x**(j + 1)


# Generate 200-point Gauss-Legendre quadrature points and weights
t, w = leggauss(200)

# Convert the quadrature interval from [-1, 1] to [0, 1]
x = (t + 1) / 2
w = w / 2

lowest_energies = []


# Question 2
# Construct S and H and solve Hc = ESc for M = 0,...,5
for M in range(6):

    n = M + 1

    S = np.zeros((n, n))
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            S[i, j] = np.sum(
                w * b(i, x) * b(j, x)
            )

            H[i, j] = 0.5 * np.sum(
                w * db(i, x) * db(j, x)
            )

    energies, coeffs = eigh(H, S)

    lowest_energies.append(energies[0])

    print(f"\nM = {M}")
    print("Eigenvalues =")
    print(energies)


# Question 3
# Compare the lowest eigenvalue with the exact ground-state energy
exact_energy = np.pi**2 / 2

print("\nM        E_M              E_M - pi^2/2")

for M, E_M in enumerate(lowest_energies):

    error = E_M - exact_energy

    print(
        f"{M}   {E_M:.12f}   {error:.12e}"
    )


# Question 4
# After the loop, S, H, and coeffs correspond to M = 5.
# The first eigenvector corresponds to the lowest eigenvalue.
c = coeffs[:, 0]

# Check normalization in the nonorthogonal basis
normalization = c.T @ S @ c
print("\nM = 5 normalization =", normalization)


# Construct the M = 5 trial wavefunction on 1000 points
x_plot = np.linspace(0, 1, 1000)

psi5 = np.zeros_like(x_plot)

for j in range(6):
    psi5 += c[j] * b(j, x_plot)


# Exact ground-state wavefunction
psi_exact = np.sqrt(2) * np.sin(np.pi * x_plot)


# Calculate the maximum absolute difference
max_diff = np.max(
    np.abs(psi5 - psi_exact)
)

print(
    "Maximum absolute difference =",
    max_diff
)


# Save the full wavefunction comparison plot
plt.figure()

plt.plot(
    x_plot,
    psi5,
    label="M=5 trial"
)

plt.plot(
    x_plot,
    psi_exact,
    label="Exact"
)

plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.legend()

plt.savefig(
    "wavefunction_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Use a denser grid for the zoomed plot near the maximum
x_zoom = np.linspace(
    0.499,
    0.5013,
    2000
)

psi5_zoom = np.zeros_like(x_zoom)

for j in range(6):
    psi5_zoom += c[j] * b(j, x_zoom)

psi_exact_zoom = (
    np.sqrt(2)
    * np.sin(np.pi * x_zoom)
)


# Save the zoomed comparison plot
plt.figure()

plt.plot(
    x_zoom,
    psi5_zoom,
    label="M=5 trial"
)

plt.plot(
    x_zoom,
    psi_exact_zoom,
    label="Exact"
)

plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.legend()

plt.savefig(
    "wavefunction_zoom.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()