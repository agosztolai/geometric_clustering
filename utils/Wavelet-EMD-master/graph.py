import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out.txt", sep=" ", header=None)
plt.plot(df[0], df[1], 'r', label="line with slope 1")
plt.plot(df[0], df[2], 'b', label="Wavelet EMD/EMD")

plt.legend(loc='upper left')
plt.grid(True)
#plt.title("Avg Time Taken to enter CS Vs No.of Threads")
plt.xlabel("EMD")
plt.ylabel("Wavelet EMD")
plt.show()
