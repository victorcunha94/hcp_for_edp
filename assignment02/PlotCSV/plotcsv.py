import pandas as pd
import matplotlib.pyplot as plt

file_path = "D:\\hcp\\rk4-001.csv"
df = pd.read_csv(file_path)

# Filter out stability = -1
df = df[df["stable"] != -1]

plt.figure(figsize=(8, 8))

# OPTION A: gradient colormap
scatter = plt.scatter(
    df["x"], df["y"],
    c=df["stable"], cmap="plasma",  # try "viridis", "inferno", etc.
    s=0.5, marker="."
)

# OPTION B: single color (uncomment to use)
# scatter = plt.scatter(
#     df["x"], df["y"],
#     color="blue",  # any valid matplotlib color
#     s=0.5, marker="."
# )

plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Estabilidade no Plano Complexo")

# Show colorbar only if using gradient
if hasattr(scatter, "get_array"):
    cbar = plt.colorbar(scatter)
    cbar.set_label("ID da Thread")

plt.show()