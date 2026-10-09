import numpy as np
import matplotlib.pyplot as plt


# Question 1

# Oxygen s and p orbital energies
epsilon_s = -34.02
epsilon_p = -16.77

# Constant used in the Harrison equations
hbar2_over_me = 7.62

# Energy matrix for one oxygen atom
D = np.diag([
    epsilon_s,
    epsilon_p,
    epsilon_p,
    epsilon_p
])

# O-O distances used in the calculation
d_values = [1.0, 1.208, 1.5, 2.0]

for d in d_values:

    # Calculate the coupling values
    V_ss_sigma = -1.32 * hbar2_over_me / d**2
    V_sp_sigma = 1.42 * hbar2_over_me / d**2
    V_pp_sigma = 2.22 * hbar2_over_me / d**2
    V_pp_pi = -0.63 * hbar2_over_me / d**2

    # Coupling matrix between the two oxygen atoms
    H12 = np.array([
        [V_ss_sigma, 0.0, 0.0, V_sp_sigma],
        [0.0, V_pp_pi, 0.0, 0.0],
        [0.0, 0.0, V_pp_pi, 0.0],
        [-V_sp_sigma, 0.0, 0.0, V_pp_sigma]
    ])

    # Full 8 by 8 Hamiltonian
    H = np.block([
        [D, H12],
        [H12.T, D]
    ])

    # Check that the Hamiltonian is Hermitian
    hermiticity_error = np.max(np.abs(H - H.T))

    # Find the energies and states
    energies, eigenvectors = np.linalg.eigh(H)

    print("\nd =", d, "Angstrom")
    print("Hermiticity error =", hermiticity_error)

    for i in range(8):

        # Current state
        c = eigenvectors[:, i]

        # Amount coming from s and pz orbitals
        sigma_weight = (
            abs(c[0])**2
            + abs(c[3])**2
            + abs(c[4])**2
            + abs(c[7])**2
        )

        # Amount coming from px and py orbitals
        pi_weight = (
            abs(c[1])**2
            + abs(c[2])**2
            + abs(c[5])**2
            + abs(c[6])**2
        )

        # Split the state between oxygen 1 and oxygen 2
        c1 = c[:4]
        c2 = c[4:]

        # Use the coupling energy to tell bonding from antibonding
        coupling_energy = 2 * c1 @ H12 @ c2

        # Decide if the state is sigma or pi
        if sigma_weight > pi_weight:
            orbital_type = "sigma"
        else:
            orbital_type = "pi"

        # Add a star for an antibonding state
        if coupling_energy > 0:
            orbital_type += "*"

        print(
            f"state {i + 1}: "
            f"E = {energies[i]:.6f} eV, "
            f"type = {orbital_type}"
        )


# Question 2

# Number of electrons in each molecular orbital
occupations = np.array([2, 2, 2, 2, 2, 1, 1, 0])


# Question 3

# Known O-O bond distance
d0 = 1.208

# Small step used for the derivative
h = 1.0e-4

# Energy of two separated oxygen atoms
E_sep = 2 * (2 * epsilon_s + 4 * epsilon_p)

# Calculate the band energy for any O-O distance
def E_band(d):

    # Calculate the coupling values
    V_ss_sigma = -1.32 * hbar2_over_me / d**2
    V_sp_sigma = 1.42 * hbar2_over_me / d**2
    V_pp_sigma = 2.22 * hbar2_over_me / d**2
    V_pp_pi = -0.63 * hbar2_over_me / d**2

    # Coupling matrix between the two oxygen atoms
    H12 = np.array([
        [V_ss_sigma, 0.0, 0.0, V_sp_sigma],
        [0.0, V_pp_pi, 0.0, 0.0],
        [0.0, 0.0, V_pp_pi, 0.0],
        [-V_sp_sigma, 0.0, 0.0, V_pp_sigma]
    ])

    # Full 8 by 8 Hamiltonian
    H = np.block([
        [D, H12],
        [H12.T, D]
    ])

    # Find the orbital energies
    energies = np.linalg.eigvalsh(H)

    # Add the energies of the occupied orbitals
    return np.sum(occupations * energies)


# Find the slope of the band energy at d0
dEband_dd = (
    E_band(d0 + h) - E_band(d0 - h)
) / (2 * h)

# Find C so the total energy slope is zero at d0
C = d0**5 * dEband_dd / 4

# Repulsive energy at d0
V_rep_d0 = C / d0**4

# Total energy at d0
E_tot_d0 = E_band(d0) + V_rep_d0

# Energy difference from two separated oxygen atoms
delta_E_d0 = E_tot_d0 - E_sep

print("\nE_sep =", E_sep, "eV")
print("E_band(d0) =", E_band(d0), "eV")
print("dE_band/dd =", dEband_dd, "eV/Angstrom")
print("C =", C, "eV Angstrom^4")
print("V_rep(d0) =", V_rep_d0, "eV")
print("E_tot(d0) =", E_tot_d0, "eV")
print("Delta E(d0) =", delta_E_d0, "eV")


# Question 4

# Make a fine list of O-O distances
d_grid = np.linspace(0.8, 2.5, 10001)

# Calculate the total energy at every distance
E_tot_grid = np.array([
    E_band(d) + C / d**4
    for d in d_grid
])

# Find where the total energy is smallest
minimum_index = np.argmin(E_tot_grid)

# Equilibrium distance
d_eq = d_grid[minimum_index]

# Total energy at the equilibrium distance
E_tot_eq = E_tot_grid[minimum_index]

# Cohesive energy
E_coh = E_sep - E_tot_eq

print("\nd_eq =", d_eq, "Angstrom")
print("E_tot(d_eq) =", E_tot_eq, "eV")
print("E_coh =", E_coh, "eV")


# Question 5

# Band energy compared with two separated oxygen atoms
E_band_relative = np.array([
    E_band(d) - E_sep
    for d in d_grid
])

# Repulsive energy at every distance
V_rep_grid = C / d_grid**4

# Total energy difference at every distance
delta_E_grid = E_band_relative + V_rep_grid

# Plot the three energies
plt.figure()

plt.plot(
    d_grid,
    E_band_relative,
    label=r"$E_{\mathrm{band}}-E_{\mathrm{sep}}$"
)

plt.plot(
    d_grid,
    V_rep_grid,
    label=r"$V_{\mathrm{rep}}$"
)

plt.plot(
    d_grid,
    delta_E_grid,
    label=r"$\Delta E$"
)

plt.xlabel(r"$d$ ($\AA$)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.grid()

# Save the plot at 600 DPI
plt.savefig(
    "problem4_question5.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

