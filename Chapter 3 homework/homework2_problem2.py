import numpy as np
import matplotlib.pyplot as plt

# model parameters
epsilon = 0.0
t = 1.0

# values of U/t used in the problem
U_values = [0, 1, 2, 4, 8, 16]


# Question 1
# build and diagonalize the even singlet Hamiltonian

for U in U_values:
    H_even = np.array([
        [2 * epsilon, -2 * t],
        [-2 * t, 2 * epsilon + U]
    ], dtype=float)

    energies, eigenvectors = np.linalg.eigh(H_even)

    print("\nU/t =", U / t)
    print("H_even =")
    print(H_even)
    print("Energies:")
    print(energies)
    print("Eigenvectors:")
    print(eigenvectors)


# Question 2
# calculate ground state energy ionic probability and singlet triplet gap

results = []

for U in U_values:
    H_even = np.array([
        [2 * epsilon, -2 * t],
        [-2 * t, 2 * epsilon + U]
    ], dtype=float)

    energies, eigenvectors = np.linalg.eigh(H_even)

    # lowest eigenvalue is the ground state
    E_GS = energies[0]

    # second component is the ionic coefficient c_I
    c_I = eigenvectors[1, 0]

    # ionic probability
    ionic_probability = abs(c_I)**2

    # triplet energy
    E_T = 2 * epsilon

    # singlet triplet gap
    J = E_T - E_GS

    results.append([
        U / t,
        E_GS / t,
        ionic_probability,
        J / t
    ])

print("\nU/t    E_GS/t      |c_I|^2      J/t")

for row in results:
    print(
        f"{row[0]:<6.1f} "
        f"{row[1]:<12.6f} "
        f"{row[2]:<12.6f} "
        f"{row[3]:<12.6f}"
    )


# Question 3
# compare numerical ground state energy with exact formula

print("\nGround-state energy check")

for U in U_values:
    E_exact = (
        2 * epsilon
        + U / 2
        - np.sqrt(4 * t**2 + U**2 / 4)
    )

    H_even = np.array([
        [2 * epsilon, -2 * t],
        [-2 * t, 2 * epsilon + U]
    ], dtype=float)

    energies, eigenvectors = np.linalg.eigh(H_even)

    E_numerical = energies[0]

    print(
        f"U/t = {U/t:.1f}, "
        f"E_numerical/t = {E_numerical/t:.6f}, "
        f"E_exact/t = {E_exact/t:.6f}, "
        f"difference = {(E_numerical - E_exact)/t:.2e}"
    )


# Question 4
# compare exact gap with the superexchange approximation

comparison = []

for U in U_values:
    if U > 0:
        H_even = np.array([
            [2 * epsilon, -2 * t],
            [-2 * t, 2 * epsilon + U]
        ], dtype=float)

        energies, eigenvectors = np.linalg.eigh(H_even)

        E_GS = energies[0]
        E_T = 2 * epsilon

        J_exact = E_T - E_GS
        J_superexchange = 4 * t**2 / U

        comparison.append([
            U / t,
            J_exact / t,
            J_superexchange / t
        ])

print("\nU/t    J_exact/t    4t^2/(U t)")

for row in comparison:
    print(
        f"{row[0]:<6.1f} "
        f"{row[1]:<12.6f} "
        f"{row[2]:<12.6f}"
    )


# Question 5
# calculate values over a larger range and make the two plots

U_plot = np.linspace(0.25, 20.0, 400)

J_exact_plot = []
ionic_plot = []

for U in U_plot:
    H_even = np.array([
        [2 * epsilon, -2 * t],
        [-2 * t, 2 * epsilon + U]
    ], dtype=float)

    energies, eigenvectors = np.linalg.eigh(H_even)

    E_GS = energies[0]
    c_I = eigenvectors[1, 0]

    J_exact_plot.append((2 * epsilon - E_GS) / t)
    ionic_plot.append(abs(c_I)**2)

# superexchange approximation for J/t
J_superexchange_plot = 4 * t / U_plot

fig, axes = plt.subplots(2, 1, figsize=(7, 8))

# exact gap and superexchange
axes[0].plot(
    U_plot / t,
    J_exact_plot,
    label="Exact J/t"
)

axes[0].plot(
    U_plot / t,
    J_superexchange_plot,
    linestyle="--",
    label="4t/U"
)

axes[0].set_xlabel("U/t")
axes[0].set_ylabel("J/t")
axes[0].legend()

# ionic probability
axes[1].plot(
    U_plot / t,
    ionic_plot
)

axes[1].set_xlabel("U/t")
axes[1].set_ylabel(r"$|c_I|^2$")

plt.tight_layout()

# save figure at 600 dpi
plt.savefig("problem2_question5.png", dpi=600)

plt.show()

