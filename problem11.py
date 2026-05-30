import numpy as np
import matplotlib.pyplot as plt

N = 180
M = 100
R = 50.0
center_x = 0.5
center_y = 0.0
first_layer_dist = 0.01

beta = np.linspace(0, 2 * np.pi, N + 1)
x_in = 0.5 * (1 + np.cos(beta))
x_in = np.clip(x_in, 0, 1)

yt = 0.6 * (0.2969 * np.sqrt(x_in) - 0.1260 * x_in - 0.3516 * x_in**2 + 0.2843 * x_in**3 - 0.1036 * x_in**4)

y_in = np.zeros_like(x_in)
for i in range(N + 1):
    if i <= N // 2:
        y_in[i] = -yt[i]
    else:
        y_in[i] = yt[i]

theta_out = -beta
x_out = center_x + R * np.cos(theta_out)
y_out = center_y + R * np.sin(theta_out)

X = np.zeros((N + 1, M + 1))
Y = np.zeros((N + 1, M + 1))

for i in range(N + 1):
    L = np.sqrt((x_out[i] - x_in[i])**2 + (y_out[i] - y_in[i])**2)
    target_ratio = first_layer_dist / L
    
    k_low = 1.0001
    k_high = 1.5
    for loop in range(50):
        k_mid = (k_low + k_high) / 2.0
        val = (k_mid - 1.0) / (k_mid**M - 1.0)
        if val > target_ratio:
            k_low = k_mid
        else:
            k_high = k_mid
    k_final = (k_low + k_high) / 2.0
    
    S = (k_final**np.arange(M + 1) - 1.0) / (k_final**M - 1.0)
    
    X[i, :] = x_in[i] + S * (x_out[i] - x_in[i])
    Y[i, :] = y_in[i] + S * (y_out[i] - y_in[i])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for j in range(M + 1):
    ax1.plot(X[:, j], Y[:, j], 'k-', lw=0.1)
for i in range(N + 1):
    ax1.plot(X[i, :], Y[i, :], 'b-', lw=0.1)
ax1.set_aspect('equal')

for j in range(30):
    ax2.plot(X[:, j], Y[:, j], 'k-', lw=0.3)
for i in range(N + 1):
    ax2.plot(X[i, :30], Y[i, :30], 'b-', lw=0.3)
ax2.plot(x_in, y_in, 'r-', lw=1.5)
ax2.set_xlim(-0.2, 1.2)
ax2.set_ylim(-0.5, 0.5)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
