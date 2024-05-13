import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.DataFrame()

directory = "rawData"
files = os.listdir(directory)
for file in files:
    if file.startswith("-0.45V") and file.endswith(".dat"):
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath, sep="\s+", header=None)
        data[file] = df.iloc[:, 5]

        quantiles = np.percentile(data, [25, 50, 75])
        print("25th percentile (Q1):", quantiles[0])
        print("50th percentile (median):", quantiles[1])
        print("75th percentile (Q3):", quantiles[2])
        print()

data['mean'] = data.mean(axis=1)

plt.figure(figsize=(10, 10))
sns.boxplot(data=data.iloc[:, :-1], orient="v", linewidth=2, width=0.5)
plt.xlabel("Input Voltage Level", fontsize=14)
plt.ylabel("Output Voltage", fontsize=14)
plt.title("Distribution of Output Voltage for Input Voltage Levels", fontsize=16)
plt.tight_layout()
plt.show()




