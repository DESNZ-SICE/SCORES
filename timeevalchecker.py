import numpy as np
import matplotlib.pyplot as plt


faster = np.load("offshorewindtimeeval.npy")
slower = np.load("offshorewindtimeevalslow.npy")

uniquefastvals = np.unique(faster[:, 1])
uniqueslowvals = np.unique(slower[:, 1])
plt.scatter(slower[:, 1], slower[:, 0], label="looped")
plt.scatter(faster[:, 1], faster[:, 0], label="vectorised")
plt.xlabel("number of sites")
plt.ylabel("time taken (s)")
plt.xticks([1, 2, 3])
meanfastvals = [np.mean(faster[faster[:, 1] == i, 0]) for i in uniquefastvals]
meanslowvals = [np.mean(slower[slower[:, 1] == i, 0]) for i in uniqueslowvals]

plt.scatter(uniquefastvals, meanfastvals, label="vectorised mean", marker="x")
plt.scatter(uniqueslowvals, meanslowvals, label="looped mean", marker="x")
plt.legend()
plt.ylim([0, 7])
plt.show()
