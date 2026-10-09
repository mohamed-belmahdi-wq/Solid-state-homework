import numpy as np
import matplotlib.pyplot as plt

# Graphene parameters
a = 2.46
d = a / np.sqrt(3)
t = -2.70
gamma1 = 0.390

# Nearest-neighbor vectors
delta1 = d * np.array([0.0, 1.0])
delta2 = d * np.array([np.sqrt(3)/2, -1/2])
delta3 = d * np.array([-np.sqrt(3)/2, -1/2])

# Graphene structure factor
def f(k):
    return (
        np.exp(1j * np.dot(k, delta1))
        + np.exp(1j * np.dot(k, delta2))
        + np.exp(1j * np.dot(k, delta3))
    )

# High-symmetry points
Gamma = np.array([0.0, 0.0])

K = np.array([
    4 * np.pi / (3 * np.sqrt(3) * d),
    0.0
])

M = np.array([
    np.pi / (np.sqrt(3) * d),
    np.pi / (3 * d)
])

# Question 3
# Plot the four AB-bilayer bands and compare them with two uncoupled layers

path1 = np.linspace(Gamma, K, 200)
path2 = np.linspace(K, M, 200)
path3 = np.linspace(M, Gamma, 200)

k_path = np.vstack((path1, path2, path3))
x = np.arange(len(k_path))

bands = []

for kpoint in k_path:
    tf = abs(t * f(kpoint))
    root = np.sqrt(tf**2 + (gamma1 / 2)**2)

    energies = [
        -(root + gamma1 / 2),
        -(root - gamma1 / 2),
         root - gamma1 / 2,
         root + gamma1 / 2
    ]

    bands.append(energies)

bands = np.array(bands)

# Two uncoupled graphene layers have doubly degenerate monolayer bands
uncoupled_plus = np.array([
    abs(t) * abs(f(kpoint)) for kpoint in k_path
])

uncoupled_minus = -uncoupled_plus

plt.figure(figsize=(8, 6))

for i in range(4):
    plt.plot(x, bands[:, i])

plt.plot(x, uncoupled_plus, "--", label="Uncoupled layers")
plt.plot(x, uncoupled_minus, "--")

plt.xticks(
    [0, 199, 399, 599],
    [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"]
)

plt.ylabel("Energy (eV)")
plt.title("AB-stacked bilayer graphene")
plt.legend()

plt.tight_layout()
plt.savefig("problem2_question3.png", dpi=600)
plt.show()

# Question 4
# Calculate the four eigenstates at K and their orbital weights

H_K = np.array([
    [0.0, 0.0,    0.0,    0.0],
    [0.0, 0.0,    gamma1, 0.0],
    [0.0, gamma1, 0.0,    0.0],
    [0.0, 0.0,    0.0,    0.0]
])

eigenvalues, eigenvectors = np.linalg.eigh(H_K)

# Each column of eigenvectors is an eigenstate
# Squaring the amplitudes gives the orbital weights
weights = np.abs(eigenvectors)**2
weights = weights.T

print("Question 4 eigenvalues:")
print(eigenvalues)

print("\nQuestion 4 orbital weights:")
print(weights)

plt.figure(figsize=(7, 4))

plt.imshow(weights, aspect="auto")

plt.xticks(
    [0, 1, 2, 3],
    [r"$A_1$", r"$B_1$", r"$A_2$", r"$B_2$"]
)

plt.yticks(
    [0, 1, 2, 3],
    [f"E = {E:.3f} eV" for E in eigenvalues]
)

plt.xlabel("Orbital")
plt.ylabel("Eigenstate")
plt.title("Orbital weights at K")

plt.colorbar(label=r"$|c_i|^2$")

for i in range(4):
    for j in range(4):
        plt.text(
            j,
            i,
            f"{weights[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("problem2_question4.png", dpi=600)
plt.show()
