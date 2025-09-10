import pandas as pd
import matplotlib.pyplot as plt

# Option 1: forward slashes
file_path = "./ab4am4test.csv"

# Option 2: raw string with backslashes
# file_path = r"C:\Users\You\Documents\abm3.csv"

df = pd.read_csv(file_path)

plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    df["x"], df["y"], 
    c=df["stable"], cmap="plasma", 
    s=0.5, marker="."
)

plt.xlabel("Real part (x)")
plt.ylabel("Imaginary part (y)")
plt.title("Stability in the Complex Plane")

cbar = plt.colorbar(scatter)
cbar.set_label("Stability value")

plt.show()
