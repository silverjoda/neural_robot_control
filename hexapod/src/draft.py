import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,100)
plt.plot(t, [0.1**p for p in t])
plt.show()