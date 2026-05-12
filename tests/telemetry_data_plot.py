import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/preprocessed_telemetry_data/dim_reduced_to_2.csv")

plt.figure(figsize=(14,14))
plt.scatter(x=df["PC1"], y=df["PC2"], c="steelblue", edgecolors="black")
plt.title("Scatter plot of telemetry data after PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/telemetry_scatter.png")
plt.show()
