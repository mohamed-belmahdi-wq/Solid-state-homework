import numpy as np
import matplotlib.pyplot as plt

# Problem 3: Three-dimensional Bernal graphite

# Parameters
a = 2.46
d = a / np.sqrt(3)
c = 6.70
t = -2.70
gamma1 = 0.390

# Nearest-neighbor vectors within one graphene layer
delta1 = d * np.array([0.0, 1.0])
delta2 = d * np.array([np.sqrt(3)/2, -1/2])
delta3 = d * np.array([-np.sqrt(3)/2, -1/2])


# In-plane structure factor
def f(kx, ky):
    k = np.array([kx, ky])

    return (
        np.exp(1j * np.dot(k, delta1))
        + np.exp(1j * np.dot(k, delta2))
        + np.exp(1j * np.dot(k, delta3))
    )


# Effective vertical hopping
def g(kz):
    return 2 * gamma1 * np.cos(kz * c / 2)


# Bulk Hamiltonian in the basis (A1, B1, A2, B2)
def H(kx, ky, kz):
    fk = f(kx, ky)
    gk = g(kz)

    return np.array([
        [0.0, t * fk, 0.0, 0.0],
        [t * np.conjugate(fk), 0.0, gk, 0.0],
        [0.0, gk, 0.0, t * fk],
        [0.0, 0.0, t * np.conjugate(fk), 0.0]
    ], dtype=complex)


# Question 2: high-symmetry band structure

Gamma = np.array([0.0, 0.0, 0.0])

K = np.array([
    4 * np.pi / (3 * np.sqrt(3) * d),
    0.0,
    0.0
])

M = np.array([
    np.pi / (np.sqrt(3) * d),
    np.pi / (3 * d),
    0.0
])

A = np.array([
    0.0,
    0.0,
    np.pi / c
])

H_point = np.array([
    K[0],
    K[1],
    np.pi / c
])

L = np.array([
    M[0],
    M[1],
    np.pi / c
])

path1 = np.linspace(Gamma, K, 150)
path2 = np.linspace(K, M, 150)
path3 = np.linspace(M, Gamma, 150)
path4 = np.linspace(Gamma, A, 150)
path5 = np.linspace(A, H_point, 150)
path6 = np.linspace(H_point, L, 150)
path7 = np.linspace(L, A, 150)

k_path = np.vstack((
    path1,
    path2,
    path3,
    path4,
    path5,
    path6,
    path7
))

bands = []

for kpoint in k_path:
    kx, ky, kz = kpoint
    energies = np.linalg.eigvalsh(H(kx, ky, kz))
    bands.append(energies)

bands = np.array(bands)

x = np.arange(len(k_path))

plt.figure(figsize=(10, 6))

for i in range(4):
    plt.plot(x, bands[:, i])

plt.xticks(
    [0, 149, 299, 449, 599, 749, 899, 1049],
    [
        r"$\Gamma$",
        r"$K$",
        r"$M$",
        r"$\Gamma$",
        r"$A$",
        r"$H$",
        r"$L$",
        r"$A$"
    ]
)

plt.ylabel("Energy (eV)")
plt.title("Three-dimensional Bernal graphite band structure")

plt.tight_layout()
plt.savefig("problem3_question2.png", dpi=600)
plt.show()


# Question 4: full-zone density of states

Nk = 24

# Primitive reciprocal lattice vectors
b1 = (2 * np.pi / a) * np.array([
    1.0,
    -1 / np.sqrt(3)
])

b2 = (2 * np.pi / a) * np.array([
    0.0,
    2 / np.sqrt(3)
])

b3z = 2 * np.pi / c

# Shifted uniform coordinates inside one primitive reciprocal cell
u_values = (np.arange(Nk) + 0.5) / Nk - 0.5
v_values = (np.arange(Nk) + 0.5) / Nk - 0.5
w_values = (np.arange(Nk) + 0.5) / Nk - 0.5

energies = []

for u in u_values:
    for v in v_values:
        kxy = u * b1 + v * b2

        for w in w_values:
            kx = kxy[0]
            ky = kxy[1]
            kz = w * b3z

            energies.extend(
                np.linalg.eigvalsh(H(kx, ky, kz))
            )

energies = np.array(energies)

# Gaussian broadening
eta = 0.08

E_grid = np.linspace(
    -9.5,
    9.5,
    1200
)

DOS = np.zeros_like(E_grid)

for E in energies:
    DOS += (
        np.exp(-((E_grid - E) / eta)**2)
        / (eta * np.sqrt(np.pi))
    )

# Normalize per primitive cell
DOS /= Nk**3

integral = np.trapezoid(
    DOS,
    E_grid
)

print("Mesh =", Nk, "x", Nk, "x", Nk)
print("Number of k-points =", Nk**3)
print("Gaussian broadening =", eta, "eV")
print("Integrated DOS =", integral)

plt.figure(figsize=(8, 5))
plt.plot(E_grid, DOS)

plt.xlabel("Energy (eV)")
plt.ylabel("DOS (states/eV per primitive cell)")
plt.title("Bernal graphite density of states")

plt.tight_layout()
plt.savefig("problem3_question4.png", dpi=600)
plt.show()
