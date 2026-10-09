import numpy as np
import matplotlib.pyplot as plt

# Question 1
# define the bond vector and calculate its length

R12 = np.array([3.2, 2.3, 2.6])

d = np.linalg.norm(R12)

# direction cosines
l = R12[0] / d
m = R12[1] / d
n = R12[2] / d

# check that the direction vector is normalized
direction_check = l**2 + m**2 + n**2

print("Question 1")
print("d =", d, "Angstrom")
print("l =", l)
print("m =", m)
print("n =", n)
print("l^2 + m^2 + n^2 =", direction_check)


# Question 2
# calculate the four Slater Koster coupling parameters

hbar2_over_me = 7.62

V_ss_sigma = -1.32 * hbar2_over_me / d**2
V_sp_sigma =  1.42 * hbar2_over_me / d**2
V_pp_sigma =  2.22 * hbar2_over_me / d**2
V_pp_pi    = -0.63 * hbar2_over_me / d**2

print("\nQuestion 2")
print("V_ss_sigma =", V_ss_sigma, "eV")
print("V_sp_sigma =", V_sp_sigma, "eV")
print("V_pp_sigma =", V_pp_sigma, "eV")
print("V_pp_pi =", V_pp_pi, "eV")


# Question 3
# build the full 4 by 4 intersite coupling matrix

Rhat = np.array([l, m, n])

# build the 3 by 3 p orbital block
H_pp = (
    V_pp_pi * np.eye(3)
    + (V_pp_sigma - V_pp_pi) * np.outer(Rhat, Rhat)
)

H12 = np.zeros((4, 4))

# s to s coupling
H12[0, 0] = V_ss_sigma

# s to p and p to s couplings
H12[0, 1:] = V_sp_sigma * Rhat
H12[1:, 0] = -V_sp_sigma * Rhat

# p to p couplings
H12[1:, 1:] = H_pp

labels = ["s", "px", "py", "pz"]

print("\nQuestion 3")
print("       ", "  ".join(f"{x:>10}" for x in labels))

for label, row in zip(labels, H12):
    print(
        f"{label:>3}   "
        + "  ".join(f"{value:10.6f}" for value in row)
    )


# Question 4
# reverse the bond direction and rebuild the matrix

Rhat_reverse = -Rhat

H_pp_reverse = (
    V_pp_pi * np.eye(3)
    + (V_pp_sigma - V_pp_pi)
    * np.outer(Rhat_reverse, Rhat_reverse)
)

H12_reverse = np.zeros((4, 4))

H12_reverse[0, 0] = V_ss_sigma

H12_reverse[0, 1:] = V_sp_sigma * Rhat_reverse
H12_reverse[1:, 0] = -V_sp_sigma * Rhat_reverse

H12_reverse[1:, 1:] = H_pp_reverse

print("\nQuestion 4")
print("H12 for reversed bond:")
print(H12_reverse)

print("\nH12 transpose:")
print(H12.T)

# check the required transpose relation
max_difference = np.max(
    np.abs(H12_reverse - H12.T)
)

print("\nMaximum difference:")
print(max_difference)


# Question 5
# find the matrix elements that change sign under bond reversal

print("\nQuestion 5")
print("Matrix elements that change sign:")

for i in range(4):
    for j in range(4):
        if not np.isclose(
            H12[i, j],
            H12_reverse[i, j]
        ):
            print(
                f"{labels[i]}1 -> {labels[j]}2: "
                f"{H12[i, j]:.6f} eV -> "
                f"{H12_reverse[i, j]:.6f} eV"
            )

# make the labeled heat map of the forward bond matrix

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(H12)

ax.set_xticks(range(4))
ax.set_yticks(range(4))

ax.set_xticklabels(
    ["s2", "px2", "py2", "pz2"]
)

ax.set_yticklabels(
    ["s1", "px1", "py1", "pz1"]
)

ax.set_xlabel("Site 2 orbitals")
ax.set_ylabel("Site 1 orbitals")

# write the numerical coupling value inside each square
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{H12[i, j]:.3f}",
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=ax,
    label="Coupling energy (eV)"
)

plt.tight_layout()

# save the heat map at 600 dpi
plt.savefig(
    "problem3_question5.png",
    dpi=600
)

plt.show()
