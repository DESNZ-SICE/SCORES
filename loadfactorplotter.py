import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

names = ["floating", "offshore", "onshore"]

for name in names:
    sns.set_theme()
    data = np.loadtxt(f"{name}loadfactors.csv", delimiter=",", skiprows=1)
    plt.scatter(data[:, 0], data[:, 1], label="Average load factor")
    plt.scatter(data[:, 0], data[:, 2], label="Maximum load factor")
    plt.title(f"{name}")
    plt.xlabel("Turbine Size (MW)")
    plt.ylabel("Load Factor")
    plt.legend()
    plt.ylim([0, 0.7])

    plt.savefig(f"{name}loadfactors.png")
    plt.show()
