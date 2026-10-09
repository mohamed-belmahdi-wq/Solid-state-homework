import numpy as np
import matplotlib.pyplot as plt

# Graphene parameters
a = 2.46
d = a / np.sqrt(3)
t = -2.70

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


# Question 2
# Verify the analytical eigenvalues using numpy.linalg.eigvalsh

k = np.array([0.30, 0.40])

fk = f(k)

H = np.array([
    [0.0, t * fk],
    [t * np.conjugate(fk), 0.0]
], dtype=complex)

analytical = np.array([
    -abs(t) * abs(fk),
     abs(t) * abs(fk)
])

numerical = np.linalg.eigvalsh(H)

print("Question 2")
print("Analytical:", analytical)
print("Numerical: ", numerical)
print()


# Question 3
# Evaluate the structure factor and energies at Gamma, K, and M

Gamma = np.array([0.0, 0.0])

K = np.array([
    4 * np.pi / (3 * np.sqrt(3) * d),
    0.0
])

M = np.array([
    np.pi / (np.sqrt(3) * d),
    np.pi / (3 * d)
])

points = {
    "Gamma": Gamma,
    "K": K,
    "M": M
}

print("Question 3")

for name, kpoint in points.items():
    fk = f(kpoint)
    abs_fk = abs(fk)

    E_minus = -abs(t) * abs_fk
    E_plus = abs(t) * abs_fk

    print(name)
    print("k =", kpoint)
    print("f(k) =", fk)
    print("|f(k)| =", abs_fk)
    print("E- =", E_minus, "eV")
    print("E+ =", E_plus, "eV")
    print()


# Question 4
# Plot both bands along Gamma -> K -> M -> Gamma

path1 = np.linspace(Gamma, K, 200)
path2 = np.linspace(K, M, 200)
path3 = np.linspace(M, Gamma, 200)

k_path = np.vstack((path1, path2, path3))

E_plus = np.array([abs(t) * abs(f(k)) for k in k_path])
E_minus = -E_plus

x = np.arange(len(k_path))

# Positive band over a 2D region containing K and K'

kx = np.linspace(-2.2, 2.2, 300)
ky = np.linspace(-1.6, 1.6, 300)

KX, KY = np.meshgrid(kx, ky)

E2D = np.zeros_like(KX)

for i in range(len(ky)):
    for j in range(len(kx)):
        kpoint = np.array([KX[i, j], KY[i, j]])
        E2D[i, j] = abs(t) * abs(f(kpoint))

Kp = -K

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Band structure plot
axes[0].plot(x, E_plus)
axes[0].plot(x, E_minus)

axes[0].set_xticks([0, 199, 399, 599])
axes[0].set_xticklabels([r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
axes[0].set_ylabel("Energy (eV)")
axes[0].set_title("Graphene band structure")

# Positive-band 2D plot
plot = axes[1].contourf(KX, KY, E2D, levels=100)

axes[1].plot(K[0], K[1], "o")
axes[1].plot(Kp[0], Kp[1], "o")

axes[1].text(K[0] + 0.05, K[1], r"$K$")
axes[1].text(Kp[0] + 0.05, Kp[1], r"$K'$")

axes[1].set_xlabel(r"$k_x$ ($\AA^{-1}$)")
axes[1].set_ylabel(r"$k_y$ ($\AA^{-1}$)")
axes[1].set_title(r"$E_+(\mathbf{k})$")

fig.colorbar(plot, ax=axes[1], label="Energy (eV)")

plt.tight_layout()
plt.savefig("problem1_question4.png", dpi=600)
plt.show()
