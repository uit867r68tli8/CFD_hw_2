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

def elliptic_smooth(X_init, Y_init, dS, iterations=500, mode=1):
    Xc = X_init[:-1, :].copy()
    Yc = Y_init[:-1, :].copy()
    num_i = Xc.shape[0]
    
    def compute_derivs(X_grid, Y_grid):
        X_E = np.roll(X_grid, -1, axis=0)
        X_W = np.roll(X_grid, 1, axis=0)
        Y_E = np.roll(Y_grid, -1, axis=0)
        Y_W = np.roll(Y_grid, 1, axis=0)
        
        X_N = np.zeros_like(X_grid)
        X_S = np.zeros_like(X_grid)
        Y_N = np.zeros_like(Y_grid)
        Y_S = np.zeros_like(Y_grid)
        X_N[:, :-1] = X_grid[:, 1:]
        X_N[:, -1] = X_grid[:, -1] 
        X_S[:, 1:] = X_grid[:, :-1]
        X_S[:, 0] = X_grid[:, 0]
        Y_N[:, :-1] = Y_grid[:, 1:]
        Y_N[:, -1] = Y_grid[:, -1]
        Y_S[:, 1:] = Y_grid[:, :-1]
        Y_S[:, 0] = Y_grid[:, 0]
        
        r_x = (X_E - X_W) / 2.0
        r_y = (Y_E - Y_W) / 2.0
        n_x = (X_N - X_S) / 2.0
        n_y = (Y_N - Y_S) / 2.0
        rr_x = X_E - 2*X_grid + X_W
        rr_y = Y_E - 2*Y_grid + Y_W
        nn_x = X_N - 2*X_grid + X_S
        nn_y = Y_N - 2*Y_grid + Y_S
        
        X_NE = np.roll(X_N, -1, axis=0)
        X_NW = np.roll(X_N, 1, axis=0)
        X_SE = np.roll(X_S, -1, axis=0)
        X_SW = np.roll(X_S, 1, axis=0)
        Y_NE = np.roll(Y_N, -1, axis=0)
        Y_NW = np.roll(Y_N, 1, axis=0)
        Y_SE = np.roll(Y_S, -1, axis=0)
        Y_SW = np.roll(Y_S, 1, axis=0)
        rn_x = (X_NE - X_NW - X_SE + X_SW) / 4.0
        rn_y = (Y_NE - Y_NW - Y_SE + Y_SW) / 4.0
        
        return r_x, r_y, n_x, n_y, rr_x, rr_y, nn_x, nn_y, rn_x, rn_y, X_E, X_W, X_N, X_S, Y_E, Y_W, Y_N, Y_S

    if mode == 3:
        r_x, r_y, *_ = compute_derivs(Xc, Yc)
        dx_in = r_x[:, 0]
        dy_in = r_y[:, 0]
        lengths_in = np.sqrt(dx_in**2 + dy_in**2 + 1e-12)
        nx = -dy_in / lengths_in
        ny = dx_in / lengths_in
        dot_prod = nx * (Xc[:, -1] - Xc[:, 0]) + ny * (Yc[:, -1] - Yc[:, 0])
        nx = np.where(dot_prod < 0, -nx, nx)
        ny = np.where(dot_prod < 0, -ny, ny)
        P_wall_filtered = np.zeros(num_i)
        Q_wall_filtered = np.zeros(num_i)

    for iter in range(iterations):
        (r_x, r_y, n_x, n_y, rr_x, rr_y, nn_x, nn_y, rn_x, rn_y,
         X_E, X_W, X_N, X_S, Y_E, Y_W, Y_N, Y_S) = compute_derivs(Xc, Yc)
         
        P_final = np.zeros((num_i, M + 1))
        Q_final = np.zeros((num_i, M + 1))
         
        if mode == 3:
            t_x_eta = dS * nx
            t_y_eta = dS * ny
            
            t_x_etaeta = 2 * (Xc[:, 1] - Xc[:, 0] - t_x_eta)
            t_y_etaeta = 2 * (Yc[:, 1] - Yc[:, 0] - t_y_eta)
            
            t_alpha = t_x_eta**2 + t_y_eta**2
            t_gamma = r_x[:, 0]**2 + r_y[:, 0]**2
            t_beta = 0.0
            t_J = r_x[:, 0] * t_y_eta - t_x_eta * r_y[:, 0]
            
            Rx = -(t_alpha * rr_x[:, 0] - 2 * t_beta * rn_x[:, 0] + t_gamma * t_x_etaeta)
            Ry = -(t_alpha * rr_y[:, 0] - 2 * t_beta * rn_y[:, 0] + t_gamma * t_y_etaeta)
            
            P_wall = np.clip((t_y_eta * Rx - t_x_eta * Ry) / (t_alpha * t_J + 1e-12), -8, 8)
            Q_wall = np.clip((-r_y[:, 0] * Rx + r_x[:, 0] * Ry) / (t_gamma * t_J + 1e-12), -8, 8)
            
            alpha_relax = 0.1
            P_wall_filtered = alpha_relax * P_wall + (1 - alpha_relax) * P_wall_filtered
            Q_wall_filtered = alpha_relax * Q_wall + (1 - alpha_relax) * Q_wall_filtered
            
            decay = 0.35 
            for j in range(1, M):
                P_final[:, j] = P_wall_filtered * np.exp(-decay * (j - 1))
                Q_final[:, j] = Q_wall_filtered * np.exp(-decay * (j - 1))
                
        a = n_x[:, 1:-1]**2 + n_y[:, 1:-1]**2
        g = r_x[:, 1:-1]**2 + r_y[:, 1:-1]**2
        b = r_x[:, 1:-1] * n_x[:, 1:-1] + r_y[:, 1:-1] * n_y[:, 1:-1]
        
        Pf = P_final[:, 1:-1]
        Qf = Q_final[:, 1:-1]
        
        X_update = (a * (X_E[:, 1:-1] + X_W[:, 1:-1] + Pf * r_x[:, 1:-1]) +
                    g * (X_N[:, 1:-1] + X_S[:, 1:-1] + Qf * n_x[:, 1:-1]) -
                    2 * b * rn_x[:, 1:-1]) / (2 * (a + g) + 1e-15)
                    
        Y_update = (a * (Y_E[:, 1:-1] + Y_W[:, 1:-1] + Pf * r_y[:, 1:-1]) +
                    g * (Y_N[:, 1:-1] + Y_S[:, 1:-1] + Qf * n_y[:, 1:-1]) -
                    2 * b * rn_y[:, 1:-1]) / (2 * (a + g) + 1e-15)
        
        omega = 1.0 
        Xc[:, 1:-1] = (1 - omega) * Xc[:, 1:-1] + omega * X_update
        Yc[:, 1:-1] = (1 - omega) * Yc[:, 1:-1] + omega * Y_update

    X_out = np.zeros((num_i + 1, M + 1))
    Y_out = np.zeros((num_i + 1, M + 1))
    X_out[:-1, :] = Xc
    Y_out[:-1, :] = Yc
    X_out[-1, :] = Xc[0, :]
    Y_out[-1, :] = Yc[0, :]
    return X_out, Y_out

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

modes = [1, 2]

for idx, mode in enumerate(modes):
    X_e, Y_e = elliptic_smooth(X, Y, dS=first_layer_dist, iterations=500, mode=mode)
    
    ax = axs[idx]
    for j in range(30):
        ax.plot(X_e[:, j], Y_e[:, j], 'k-', lw=0.3)
    for i in range(N + 1):
        ax.plot(X_e[i, :30], Y_e[i, :30], 'k-', lw=0.3)
    ax.plot(x_in, y_in, 'r-', lw=1.5)
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
