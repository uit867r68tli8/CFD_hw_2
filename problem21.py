import numpy as np
import matplotlib.pyplot as plt

N = 180
M_grid = 100
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

X = np.zeros((N + 1, M_grid + 1))
Y = np.zeros((N + 1, M_grid + 1))

for i in range(N + 1):
    L = np.sqrt((x_out[i] - x_in[i])**2 + (y_out[i] - y_in[i])**2)
    target_ratio = first_layer_dist / L
    
    k_low = 1.0001
    k_high = 1.5
    for loop in range(50):
        k_mid = (k_low + k_high) / 2.0
        val = (k_mid - 1.0) / (k_mid**M_grid - 1.0)
        if val > target_ratio:
            k_low = k_mid
        else:
            k_high = k_mid
    k_final = (k_low + k_high) / 2.0
    
    S = (k_final**np.arange(M_grid + 1) - 1.0) / (k_final**M_grid - 1.0)
    X[i, :] = x_in[i] + S * (x_out[i] - x_in[i])
    Y[i, :] = y_in[i] + S * (y_out[i] - y_in[i])

def elliptic_smooth(X_init, Y_init, dS, iterations=500):
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
        X_NE = np.roll(X_N, -1, axis=0); X_NW = np.roll(X_N, 1, axis=0)
        X_SE = np.roll(X_S, -1, axis=0); X_SW = np.roll(X_S, 1, axis=0)
        Y_NE = np.roll(Y_N, -1, axis=0); Y_NW = np.roll(Y_N, 1, axis=0)
        Y_SE = np.roll(Y_S, -1, axis=0); Y_SW = np.roll(Y_S, 1, axis=0)
        rn_x = (X_NE - X_NW - X_SE + X_SW) / 4.0
        rn_y = (Y_NE - Y_NW - Y_SE + Y_SW) / 4.0
        return r_x, r_y, n_x, n_y, rr_x, rr_y, nn_x, nn_y, rn_x, rn_y, X_E, X_W, X_N, X_S, Y_E, Y_W, Y_N, Y_S
    
    r_x, r_y, n_x, n_y, rr_x, rr_y, nn_x, nn_y, *_ = compute_derivs(Xc, Yc)
    P_bg = np.clip((r_x * rr_x + r_y * rr_y) / (r_x**2 + r_y**2 + 1e-12), -10, 10)
    Q_bg = np.clip((n_x * nn_x + n_y * nn_y) / (n_x**2 + n_y**2 + 1e-12), -10, 10)
    dx_in = r_x[:, 0]; dy_in = r_y[:, 0]
    lengths_in = np.sqrt(dx_in**2 + dy_in**2 + 1e-12)
    nx = -dy_in / lengths_in; ny = dx_in / lengths_in
    dot_prod = nx * (Xc[:, -1] - Xc[:, 0]) + ny * (Yc[:, -1] - Yc[:, 0])
    nx = np.where(dot_prod < 0, -nx, nx); ny = np.where(dot_prod < 0, -ny, ny)
    P_wall_filtered = np.zeros(num_i); Q_wall_filtered = np.zeros(num_i)

    for iter in range(iterations):
        (r_x, r_y, n_x, n_y, rr_x, rr_y, nn_x, nn_y, rn_x, rn_y,
         X_E, X_W, X_N, X_S, Y_E, Y_W, Y_N, Y_S) = compute_derivs(Xc, Yc)
        t_x_eta = dS * nx; t_y_eta = dS * ny
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
        P_final = np.copy(P_bg); Q_final = np.copy(Q_bg)
        decay = 0.35 
        
        for j in range(1, M_grid):
            P_final[:, j] += (P_wall_filtered - P_bg[:, 1]) * np.exp(-decay * (j - 1))
            Q_final[:, j] += (Q_wall_filtered - Q_bg[:, 1]) * np.exp(-decay * (j - 1))
            
        a = n_x[:, 1:-1]**2 + n_y[:, 1:-1]**2
        g = r_x[:, 1:-1]**2 + r_y[:, 1:-1]**2
        b = r_x[:, 1:-1] * n_x[:, 1:-1] + r_y[:, 1:-1] * n_y[:, 1:-1]
        Pf = P_final[:, 1:-1]; Qf = Q_final[:, 1:-1]
        X_update = (a * (X_E[:, 1:-1] + X_W[:, 1:-1] + Pf * r_x[:, 1:-1]) +
                    g * (X_N[:, 1:-1] + X_S[:, 1:-1] + Qf * n_x[:, 1:-1]) -
                    2 * b * rn_x[:, 1:-1]) / (2 * (a + g) + 1e-15)
        Y_update = (a * (Y_E[:, 1:-1] + Y_W[:, 1:-1] + Pf * r_y[:, 1:-1]) +
                    g * (Y_N[:, 1:-1] + Y_S[:, 1:-1] + Qf * n_y[:, 1:-1]) -
                    2 * b * rn_y[:, 1:-1]) / (2 * (a + g) + 1e-15)
        omega = 1.0 
        Xc[:, 1:-1] = (1 - omega) * Xc[:, 1:-1] + omega * X_update
        Yc[:, 1:-1] = (1 - omega) * Yc[:, 1:-1] + omega * Y_update

    X_out = np.zeros((num_i + 1, M_grid + 1)); Y_out = np.zeros((num_i + 1, M_grid + 1))
    X_out[:-1, :] = Xc; Y_out[:-1, :] = Yc
    X_out[-1, :] = Xc[0, :]; Y_out[-1, :] = Yc[0, :]
    return X_out, Y_out

X_e, Y_e = elliptic_smooth(X, Y, dS=first_layer_dist, iterations=1000)

gamma = 1.4
Xc = X_e[:-1, :]
Yc = Y_e[:-1, :]

dx_j = np.roll(Xc, -1, axis=0) - Xc
dy_j = np.roll(Yc, -1, axis=0) - Yc
ds_j = np.sqrt(dx_j**2 + dy_j**2 + 1e-15)
nx_j = -dy_j / ds_j
ny_j = dx_j / ds_j

dx_i = Xc[:, 1:] - Xc[:, :-1]
dy_i = Yc[:, 1:] - Yc[:, :-1]
ds_i = np.sqrt(dx_i**2 + dy_i**2 + 1e-15)
nx_i = dy_i / ds_i
ny_i = -dx_i / ds_i

X_0 = Xc[:, :-1]
Y_0 = Yc[:, :-1]
X_1 = np.roll(Xc, -1, axis=0)[:, :-1]
Y_1 = np.roll(Yc, -1, axis=0)[:, :-1]
X_2 = np.roll(Xc, -1, axis=0)[:, 1:]
Y_2 = np.roll(Yc, -1, axis=0)[:, 1:]
X_3 = Xc[:, 1:]
Y_3 = Yc[:, 1:]
vol = 0.5 * np.abs((X_2 - X_0)*(Y_3 - Y_1) - (X_3 - X_1)*(Y_2 - Y_0))
vol_ext = vol[..., np.newaxis]

def get_face_states(W_LL, W_L, W_R, W_RR):
    dW_L = W_L - W_LL
    dW_M = W_R - W_L
    dW_R = W_RR - W_R
    num_L = dW_M * dW_L + np.abs(dW_M * dW_L)
    den_L = dW_L**2 + dW_L * dW_M + 1e-15
    state_L = W_L + 0.5 * (num_L / den_L) * dW_L
    num_R = dW_M * dW_R + np.abs(dW_M * dW_R)
    den_R = dW_R**2 + dW_R * dW_M + 1e-15
    state_R = W_R - 0.5 * (num_R / den_R) * dW_R
    return state_L, state_R

def farfield_bc(W_in, nx, ny, Minf, alpha):
    c_inf = np.sqrt(gamma)
    u_inf = Minf * c_inf * np.cos(alpha)
    v_inf = Minf * c_inf * np.sin(alpha)
    rho_in = W_in[:, 0]
    u_in = W_in[:, 1]
    v_in = W_in[:, 2]
    p_in = W_in[:, 3]
    c_in = np.sqrt(gamma * p_in / rho_in)
    Vn_in = u_in * nx + v_in * ny
    Vn_inf = u_inf * nx + v_inf * ny
    R_plus = Vn_in + 2 * c_in / (gamma - 1)
    R_minus = Vn_inf - 2 * c_inf / (gamma - 1)
    Vn_b = 0.5 * (R_plus + R_minus)
    c_b = (gamma - 1) / 4.0 * (R_plus - R_minus)
    Vt_in = -u_in * ny + v_in * nx
    Vt_inf = -u_inf * ny + v_inf * nx
    s_in = p_in / (rho_in ** gamma)
    s_inf = 1.0
    inflow = Vn_b < 0
    Vt_b = np.where(inflow, Vt_inf, Vt_in)
    s_b = np.where(inflow, s_inf, s_in)
    rho_b = (c_b**2 / (gamma * s_b)) ** (1.0 / (gamma - 1))
    p_b = rho_b * c_b**2 / gamma
    u_b = Vn_b * nx - Vt_b * ny
    v_b = Vn_b * ny + Vt_b * nx
    W_out = np.zeros_like(W_in)
    W_out[:, 0] = rho_b
    W_out[:, 1] = u_b
    W_out[:, 2] = v_b
    W_out[:, 3] = p_b
    return W_out

def get_padded_W(W, Minf, alpha):
    W_pad = np.zeros((N + 4, M_grid + 4, 4))
    W_pad[2:N+2, 2:M_grid+2, :] = W
    vn_1 = W_pad[2:N+2, 2, 1] * nx_j[:, 0] + W_pad[2:N+2, 2, 2] * ny_j[:, 0]
    W_pad[2:N+2, 1, 0] = W_pad[2:N+2, 2, 0]
    W_pad[2:N+2, 1, 1] = W_pad[2:N+2, 2, 1] - 2 * vn_1 * nx_j[:, 0]
    W_pad[2:N+2, 1, 2] = W_pad[2:N+2, 2, 2] - 2 * vn_1 * ny_j[:, 0]
    W_pad[2:N+2, 1, 3] = W_pad[2:N+2, 2, 3]
    vn_0 = W_pad[2:N+2, 3, 1] * nx_j[:, 0] + W_pad[2:N+2, 3, 2] * ny_j[:, 0]
    W_pad[2:N+2, 0, 0] = W_pad[2:N+2, 3, 0]
    W_pad[2:N+2, 0, 1] = W_pad[2:N+2, 3, 1] - 2 * vn_0 * nx_j[:, 0]
    W_pad[2:N+2, 0, 2] = W_pad[2:N+2, 3, 2] - 2 * vn_0 * ny_j[:, 0]
    W_pad[2:N+2, 0, 3] = W_pad[2:N+2, 3, 3]
    W_far = farfield_bc(W_pad[2:N+2, M_grid+1, :], nx_j[:, -1], ny_j[:, -1], Minf, alpha)
    W_pad[2:N+2, M_grid+2, :] = W_far
    W_pad[2:N+2, M_grid+3, :] = W_far
    W_pad[0:2, :, :] = W_pad[N:N+2, :, :]
    W_pad[N+2:N+4, :, :] = W_pad[2:4, :, :]
    return W_pad

def roe_flux(W_L, W_R, nx, ny, ds):
    rho_L, u_L, v_L, p_L = W_L[...,0], W_L[...,1], W_L[...,2], W_L[...,3]
    rho_R, u_R, v_R, p_R = W_R[...,0], W_R[...,1], W_R[...,2], W_R[...,3]
    un_L = u_L * nx + v_L * ny
    ut_L = -u_L * ny + v_L * nx
    un_R = u_R * nx + v_R * ny
    ut_R = -u_R * ny + v_R * nx
    H_L = gamma / (gamma - 1) * p_L / rho_L + 0.5 * (u_L**2 + v_L**2)
    H_R = gamma / (gamma - 1) * p_R / rho_R + 0.5 * (u_R**2 + v_R**2)
    R_rho = np.sqrt(rho_R / rho_L)
    rho_hat = rho_L * R_rho
    u_hat = (u_L + u_R * R_rho) / (1.0 + R_rho)
    v_hat = (v_L + v_R * R_rho) / (1.0 + R_rho)
    H_hat = (H_L + H_R * R_rho) / (1.0 + R_rho)
    un_hat = u_hat * nx + v_hat * ny
    ut_hat = -u_hat * ny + v_hat * nx
    c2 = (gamma - 1) * (H_hat - 0.5 * (u_hat**2 + v_hat**2))
    c_hat = np.sqrt(np.maximum(c2, 1e-7))
    d_rho = rho_R - rho_L
    d_p = p_R - p_L
    d_un = un_R - un_L
    d_ut = ut_R - ut_L
    a1 = (d_p - rho_hat * c_hat * d_un) / (2 * c_hat**2)
    a2 = d_rho - d_p / c_hat**2
    a3 = rho_hat * d_ut
    a4 = (d_p + rho_hat * c_hat * d_un) / (2 * c_hat**2)
    l1 = un_hat - c_hat
    l2 = un_hat
    l3 = un_hat
    l4 = un_hat + c_hat
    delta = 0.1 * c_hat
    l1 = np.where(np.abs(l1) < delta, (l1**2 + delta**2) / (2 * delta), np.abs(l1))
    l4 = np.where(np.abs(l4) < delta, (l4**2 + delta**2) / (2 * delta), np.abs(l4))
    l2 = np.abs(l2)
    l3 = np.abs(l3)
    D1 = l1 * a1
    D2 = l2 * a2
    D3 = l3 * a3
    D4 = l4 * a4
    D_rho = D1 + D2 + D4
    D_un = D1 * (un_hat - c_hat) + D2 * un_hat + D4 * (un_hat + c_hat)
    D_ut = D1 * ut_hat + D2 * ut_hat + D3 * 1.0 + D4 * ut_hat
    D_E = D1 * (H_hat - un_hat * c_hat) + D2 * 0.5 * (un_hat**2 + ut_hat**2) + D3 * ut_hat + D4 * (H_hat + un_hat * c_hat)
    F_L_rho = rho_L * un_L
    F_L_un = rho_L * un_L**2 + p_L
    F_L_ut = rho_L * un_L * ut_L
    F_L_E = un_L * rho_L * H_L
    F_R_rho = rho_R * un_R
    F_R_un = rho_R * un_R**2 + p_R
    F_R_ut = rho_R * un_R * ut_R
    F_R_E = un_R * rho_R * H_R
    F_roe_rho = 0.5 * (F_L_rho + F_R_rho - D_rho)
    F_roe_un = 0.5 * (F_L_un + F_R_un - D_un)
    F_roe_ut = 0.5 * (F_L_ut + F_R_ut - D_ut)
    F_roe_E = 0.5 * (F_L_E + F_R_E - D_E)
    F = np.zeros_like(W_L)
    F[..., 0] = F_roe_rho * ds
    F[..., 1] = (F_roe_un * nx - F_roe_ut * ny) * ds
    F[..., 2] = (F_roe_un * ny + F_roe_ut * nx) * ds
    F[..., 3] = F_roe_E * ds
    return F

def compute_rhs(W, Minf, alpha):
    W_pad = get_padded_W(W, Minf, alpha)
    W_LL_i = W_pad[0:N, 2:M_grid+2, :]
    W_L_i  = W_pad[1:N+1, 2:M_grid+2, :]
    W_R_i  = W_pad[2:N+2, 2:M_grid+2, :]
    W_RR_i = W_pad[3:N+3, 2:M_grid+2, :]
    W_faceL_i, W_faceR_i = get_face_states(W_LL_i, W_L_i, W_R_i, W_RR_i)
    F_i = roe_flux(W_faceL_i, W_faceR_i, nx_i, ny_i, ds_i)
    
    W_LL_j = W_pad[2:N+2, 0:M_grid+1, :]
    W_L_j  = W_pad[2:N+2, 1:M_grid+2, :]
    W_R_j  = W_pad[2:N+2, 2:M_grid+3, :]
    W_RR_j = W_pad[2:N+2, 3:M_grid+4, :]
    W_faceL_j, W_faceR_j = get_face_states(W_LL_j, W_L_j, W_R_j, W_RR_j)
    F_j = roe_flux(W_faceL_j, W_faceR_j, nx_j, ny_j, ds_j)
    
    Flux_out = np.roll(F_i, -1, axis=0) - F_i + F_j[:, 1:, :] - F_j[:, :-1, :]
    return Flux_out

def compute_dt(W, CFL):
    c = np.sqrt(gamma * W[..., 3] / W[..., 0])
    lam_i = (np.abs(W[..., 1]*nx_i + W[..., 2]*ny_i) + c) * ds_i
    lam_j_bot = (np.abs(W[..., 1]*nx_j[:, :-1] + W[..., 2]*ny_j[:, :-1]) + c) * ds_j[:, :-1]
    lam_j_top = (np.abs(W[..., 1]*nx_j[:, 1:] + W[..., 2]*ny_j[:, 1:]) + c) * ds_j[:, 1:]
    lam_i_right = np.roll(lam_i, -1, axis=0)
    lambda_sum = lam_i + lam_i_right + lam_j_bot + lam_j_top
    dt = CFL * vol / (lambda_sum + 1e-15)
    return dt[..., np.newaxis]

def W_to_U(W):
    U = np.zeros_like(W)
    U[..., 0] = W[..., 0]
    U[..., 1] = W[..., 0] * W[..., 1]
    U[..., 2] = W[..., 0] * W[..., 2]
    U[..., 3] = W[..., 3] / (gamma - 1) + 0.5 * W[..., 0] * (W[..., 1]**2 + W[..., 2]**2)
    return U

def U_to_W(U):
    W = np.zeros_like(U)
    W[..., 0] = np.maximum(U[..., 0], 1e-5)
    W[..., 1] = U[..., 1] / W[..., 0]
    W[..., 2] = U[..., 2] / W[..., 0]
    p = (gamma - 1) * (U[..., 3] - 0.5 * W[..., 0] * (W[..., 1]**2 + W[..., 2]**2))
    W[..., 3] = np.maximum(p, 1e-5)
    return W

def run_solver(Minf, alpha, CFL, max_iter, global_dt=False):
    W = np.zeros((N, M_grid, 4))
    W[..., 0] = 1.0
    W[..., 1] = Minf * np.sqrt(gamma) * np.cos(alpha)
    W[..., 2] = Minf * np.sqrt(gamma) * np.sin(alpha)
    W[..., 3] = 1.0

    for step in range(max_iter):
        dt = compute_dt(W, CFL)
        if global_dt:
            dt = np.full_like(dt, dt.min())

        U0 = W_to_U(W)
        Flux_0 = compute_rhs(W, Minf, alpha)
        U1 = U0 - dt * Flux_0 / vol_ext
        W1 = U_to_W(U1)

        Flux_1 = compute_rhs(W1, Minf, alpha)
        U2 = 0.75 * U0 + 0.25 * (W_to_U(W1) - dt * Flux_1 / vol_ext)
        W2 = U_to_W(U2)

        Flux_2 = compute_rhs(W2, Minf, alpha)
        U_new = 1.0/3.0 * U0 + 2.0/3.0 * (W_to_U(W2) - dt * Flux_2 / vol_ext)
        W_new = U_to_W(U_new)

        max_err = np.max(np.abs(W_new[..., 0] - W[..., 0]))
        W = W_new

        if (step + 1) % 500 == 0:
            msg = f"  step {step + 1}: max_err = {max_err:.3e}"
            if global_dt:
                msg += f",  mean dt = {dt.mean():.3e}"
            print(msg)

        if max_err < 1e-3 and step > 100:
            break

    return W, step + 1


M_cases = [0.4,0.8]
run_local = True
run_global = False
alpha_deg = 1.25 * np.pi / 180.0
CFL = 0.9
max_iter = 40000

results = {}
for Minf in M_cases:
    W_local, steps_local = (run_solver(Minf, alpha_deg, CFL, max_iter, global_dt=False)
                            if run_local else (None, None))
    W_global, steps_global = (run_solver(Minf, alpha_deg, CFL, max_iter, global_dt=True)
                              if run_global else (None, None))
    print(f"M={Minf}, local dt: {steps_local} 步,  global dt: {steps_global} 步")
    results[Minf] = (W_local, steps_local, W_global, steps_global)

for Minf in M_cases:
    W_local, steps_local, W_global, steps_global = results[Minf]
    for W, label in [(W_local, f'M={Minf} local dt  steps={steps_local}'),
                     (W_global, f'M={Minf} global dt  steps={steps_global}')]:
        if W is None:
            continue
        Mach = np.sqrt(W[..., 1]**2 + W[..., 2]**2) / np.sqrt(gamma * W[..., 3] / W[..., 0])
        p = W[..., 3]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cf0 = axes[0].pcolormesh(X_e, Y_e, Mach, cmap='jet', shading='flat')
        axes[0].plot(x_in, y_in, 'k-', lw=1.5)
        axes[0].set_xlim(-1.0, 2.0)
        axes[0].set_ylim(-2.0, 2.0)
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('Mach')
        fig.colorbar(cf0, ax=axes[0], fraction=0.046, pad=0.04)

        cf1 = axes[1].pcolormesh(X_e, Y_e, p, cmap='jet', shading='flat')
        axes[1].plot(x_in, y_in, 'k-', lw=1.5)
        axes[1].set_xlim(-1.0, 2.0)
        axes[1].set_ylim(-2.0, 2.0)
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('Pressure (Pa)')
        fig.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()

plt.show()
