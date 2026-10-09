import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

hbar = 1.054571817e-34
m_e = 9.1093837139e-31
eV = 1.602176634e-19

N = 54
L = 1.00e-9

# Question 1

shells = defaultdict(list)

for nx in range(-8, 9):
    for ny in range(-8, 9):
        for nz in range(-8, 9):
            q = nx**2 + ny**2 + nz**2
            shells[q].append((nx, ny, nz))

first12_q = sorted(shells)[:12]

print("q   g(q)   2g(q)")

for q in first12_q:
    g = len(shells[q])
    print(f"{q:<3} {g:<6} {2*g}")


# Question 2

electrons = 0

print("\nq   capacity   cumulative electrons")

for q in sorted(shells):
    g = len(shells[q])
    capacity = 2*g
    electrons += capacity

    print(f"{q:<3} {capacity:<10} {electrons}")

    if electrons >= N:
        qF = q
        break

kF_finite = (2*np.pi/L) * np.sqrt(qF)

EF_finite_J = hbar**2 * kF_finite**2 / (2*m_e)
EF_finite_eV = EF_finite_J / eV

print("\nqF =", qF)
print("Finite-cell kF =", kF_finite, "m^-1")
print("Finite-cell EF =", EF_finite_eV, "eV")


# Question 3

n = N / L**3

kF_cont = (3*np.pi**2*n)**(1/3)

EF_cont_J = hbar**2 * kF_cont**2 / (2*m_e)
EF_cont_eV = EF_cont_J / eV

percent_diff = abs(EF_cont_eV - EF_finite_eV) / EF_finite_eV * 100

print("\nElectron density =", n, "m^-3")
print("Continuum kF =", kF_cont, "m^-1")
print("Continuum EF =", EF_cont_eV, "eV")
print("Percent difference =", percent_diff, "%")


# Question 4

capacities = [2 * len(shells[q]) for q in first12_q]

plt.figure()
plt.stem(first12_q, capacities)
plt.xlabel("q")
plt.ylabel("Shell capacity 2g(q)")
plt.title("Shell Capacity for First 12 Allowed Shells")
plt.savefig("problem2_shell_capacity.png", dpi=300, bbox_inches="tight")
plt.close()

E0_J = (hbar**2 / (2*m_e)) * (2*np.pi/L)**2
E0_eV = E0_J / eV

energies_dos = []

for q in first12_q:
    capacity = 2 * len(shells[q])
    energy = E0_eV * q
    energies_dos.extend([energy] * capacity)

bins = np.arange(-0.5, 13.5, 1) * E0_eV

plt.figure()
plt.hist(energies_dos, bins=bins, rwidth=0.65, edgecolor="black")

plt.axvline(EF_finite_eV, linestyle="--", label="Finite-cell $E_F$")
plt.axvline(EF_cont_eV, linestyle=":", label="Continuum $E_F$")

plt.xlabel("Energy (eV)")
plt.ylabel("Number of states")
plt.title("Density of States for First 12 Allowed Shells")
plt.legend()

plt.savefig("problem2_dos.png", dpi=300, bbox_inches="tight")
plt.close()