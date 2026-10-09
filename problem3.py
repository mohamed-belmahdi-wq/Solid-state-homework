import numpy as np
import matplotlib.pyplot as plt

# Conversion from Hartree atomic units to electronvolts
HARTREE_TO_EV = 27.2114


# Question 1

# Exchange function F(x).
# The analytic limiting values are used at x = 0 and x = 1
# because the direct expression is undefined at those points.
def F(x):
    if x == 0.0:
        return 2.0
    elif x == 1.0:
        return 1.0
    else:
        return 1 + ((1 - x**2) / (2 * x)) * np.log(
            abs((1 + x) / (1 - x))
        )


# Array version of F(x) used for the dispersion plots.
# The special values at x = 0 and x = 1 are handled separately.
def F_array(x):
    y = np.empty_like(x)

    at_zero = x == 0.0
    at_one = x == 1.0
    regular = ~(at_zero | at_one)

    y[at_zero] = 2.0
    y[at_one] = 1.0

    xr = x[regular]

    y[regular] = 1 + ((1 - xr**2) / (2 * xr)) * np.log(
        np.abs((1 + xr) / (1 - xr))
    )

    return y


# Check the two analytic limits and one ordinary value
print("Question 1")
print("F(0) =", F(0.0))
print("F(1) =", F(1.0))
print("F(0.5) =", F(0.5))
print()


# Question 2

# Calculate and plot the free-electron and Hartree-Fock
# dispersions for r_s = 1 and r_s = 2.
for rs in [1.0, 2.0]:

    # Fermi wavevector in Hartree atomic units
    kF = (9 * np.pi / 4)**(1/3) / rs

    # x = k/kF, covering the occupied region and states
    # above the Fermi surface
    x = np.linspace(0.0, 1.5, 500)

    # Convert the dimensionless x values into k values
    k = x * kF

    # Free-electron orbital energy
    eps_free = k**2 / 2

    # Hartree-Fock exchange self-energy
    sigma_x = -(kF / np.pi) * F_array(x)

    # Hartree-Fock orbital energy
    eps_HF = eps_free + sigma_x

    # Plot both dispersions
    plt.figure()

    plt.plot(x, eps_free, label="Free electron")
    plt.plot(x, eps_HF, label="Hartree-Fock")

    # The interval k/kF <= 1 is occupied at T = 0
    plt.axvspan(0, 1, alpha=0.2, label="Occupied region")

    # Mark the Fermi surface
    plt.axvline(1, linestyle="--", label="$k=k_F$")

    plt.xlabel("$k/k_F$")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"Dispersion for $r_s={rs:g}$")
    plt.legend()

    # Save one figure for each density
    plt.savefig(
        f"problem3_rs{int(rs)}_dispersion.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Question 3

# Calculate the requested Fermi-surface and average energies
# for r_s = 1 and r_s = 2.
for rs in [1.0, 2.0]:

    # Fermi wavevector
    kF = (9 * np.pi / 4)**(1/3) / rs

    # Free-electron orbital energy at k = kF
    eps_free_kF = kF**2 / 2

    # At the Fermi surface x = 1 and F(1) = 1,
    # so the exchange self-energy is -kF/pi.
    sigma_kF = -kF / np.pi

    # Hartree-Fock orbital energy at the Fermi surface
    eps_HF_kF = eps_free_kF + sigma_kF

    # Average kinetic energy per electron
    avg_kinetic = 3 * kF**2 / 10

    # Exchange contribution to the total energy per electron
    exchange_per_electron = -3 * kF / (4 * np.pi)

    # Total Hartree-Fock energy per electron
    E_HF_per_electron = avg_kinetic + exchange_per_electron

    # Report all energies in electronvolts
    print(f"r_s = {rs}")
    print(f"kF = {kF}")
    print(
        f"free energy at kF = "
        f"{eps_free_kF * HARTREE_TO_EV} eV"
    )
    print(
        f"HF orbital energy at kF = "
        f"{eps_HF_kF * HARTREE_TO_EV} eV"
    )
    print(
        f"average kinetic energy = "
        f"{avg_kinetic * HARTREE_TO_EV} eV"
    )
    print(
        f"exchange energy per electron = "
        f"{exchange_per_electron * HARTREE_TO_EV} eV"
    )
    print(
        f"HF total energy per electron = "
        f"{E_HF_per_electron * HARTREE_TO_EV} eV"
    )
    print()


# Question 4

# The slope of the Hartree-Fock dispersion is examined
# at the Fermi surface for r_s = 2.
rs = 2.0

kF = (9 * np.pi / 4)**(1/3) / rs


# Hartree-Fock orbital energy written as a function
# of x = k/kF.
def eps_HF_x(x):
    k = x * kF

    sigma_x = -(kF / np.pi) * F(x)

    return k**2 / 2 + sigma_x


# Step sizes used for the symmetric finite difference
deltas = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6
]


# Estimate d epsilon_HF / dx at x = 1.
# If the derivative were finite, these values would
# approach a constant as delta becomes smaller.
print("Question 4")

for delta in deltas:

    slope = (
        eps_HF_x(1 + delta)
        - eps_HF_x(1 - delta)
    ) / (2 * delta)

    print(delta, slope)
