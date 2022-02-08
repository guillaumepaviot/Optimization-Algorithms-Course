import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, -0.000000001, 10000)

for mu in [1, 0.5, 0.1]:
    fx = -x - mu * np.log(-x)
    plt.plot(x, fx, label=f"mu = {mu}")

plt.legend()
plt.show()