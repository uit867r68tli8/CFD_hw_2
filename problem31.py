import numpy as np
import matplotlib.pyplot as plt

def solve_cavity(Re):
    Lx, Ly = 1.0, 1.0
    nx, ny = 128, 128
    dx, dy = Lx / nx, Ly / ny
    U_Lid = 1.0
    rho = 1.0
    mu = U_Lid * Lx * rho / Re
    
    alpha_u, alpha_v, alpha_p = 0.7, 0.7, 0.3
    max_iter = 50000
    
    u = np.zeros((ny + 2, nx + 2))
    v = np.zeros((ny + 2, nx + 2))
    p = np.zeros((ny + 2, nx + 2))
    pc = np.zeros((ny + 2, nx + 2))
    d_u = np.zeros((ny + 2, nx + 2))
    d_v = np.zeros((ny + 2, nx + 2))
    
    ju, iu = slice(1, ny + 1), slice(1, nx)
    ju_plus_1, iu_plus_1 = slice(2, ny + 2), slice(2, nx + 1)
    ju_minus_1, iu_minus_1 = slice(0, ny), slice(0, nx - 1)
    
    jv, iv = slice(1, ny), slice(1, nx + 1)
    jv_plus_1, iv_plus_1 = slice(2, ny + 1), slice(2, nx + 2)
    jv_minus_1, iv_minus_1 = slice(0, ny - 1), slice(0, nx)
    
    jp, ip = slice(1, ny + 1), slice(1, nx + 1)
    jp_plus_1, ip_plus_1 = slice(2, ny + 2), slice(2, nx + 2)
    jp_minus_1, ip_minus_1 = slice(0, ny), slice(0, nx)
    
    def apply_bcs(u, v):
        v[:, 0] = -v[:, 1]
        v[:, nx + 1] = -v[:, nx]
        u[0, :] = -u[1, :]
        u[ny + 1, :] = 2.0 * U_Lid - u[ny, :]
        u[:, 0] = 0.0
        u[:, nx] = 0.0
        v[0, :] = 0.0
        v[ny, :] = 0.0
        
    apply_bcs(u, v)
    gamma = 1.0
    
    for step in range(1, max_iter + 1):
        U_E = 0.5 * (u[ju, iu] + u[ju, iu_plus_1])
        U_W = 0.5 * (u[ju, iu_minus_1] + u[ju, iu])
        V_N = 0.5 * (v[ju, iu] + v[ju, iu_plus_1])
        V_S = 0.5 * (v[ju_minus_1, iu] + v[ju_minus_1, iu_plus_1])
        
        Fe, Fw = rho * dy * U_E, rho * dy * U_W
        Fn, Fs = rho * dx * V_N, rho * dx * V_S
        D_x, D_y = mu * dy / dx, mu * dx / dy
        
        aE = D_x + np.maximum(-Fe, 0)
        aW = D_x + np.maximum(Fw, 0)
        aN = D_y + np.maximum(-Fn, 0)
        aS = D_y + np.maximum(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw + Fn - Fs)
        aP_rel = aP / alpha_u
        
        dFe_u = np.abs(Fe) * 0.5 * (u[ju, iu_plus_1] - u[ju, iu])
        dFw_u = np.abs(Fw) * 0.5 * (u[ju, iu] - u[ju, iu_minus_1])
        dFn_u = np.abs(Fn) * 0.5 * (u[ju_plus_1, iu] - u[ju, iu])
        dFs_u = np.abs(Fs) * 0.5 * (u[ju, iu] - u[ju_minus_1, iu])
        
        Su_DC = gamma * (dFw_u - dFe_u + dFs_u - dFn_u)
        
        b_u = (p[ju, iu] - p[ju, iu_plus_1]) * dy + (1 - alpha_u) * aP_rel * u[ju, iu] + Su_DC
        d_u[ju, iu] = dy / aP_rel
        
        for _ in range(3):
            u[ju, iu] = (aE * u[ju, iu_plus_1] + aW * u[ju, iu_minus_1] + 
                         aN * u[ju_plus_1, iu] + aS * u[ju_minus_1, iu] + b_u) / aP_rel
            apply_bcs(u, v)
            
        U_E_v = 0.5 * (u[jv, iv] + u[jv_plus_1, iv])
        U_W_v = 0.5 * (u[jv, iv_minus_1] + u[jv_plus_1, iv_minus_1])
        V_N_v = 0.5 * (v[jv, iv] + v[jv_plus_1, iv])
        V_S_v = 0.5 * (v[jv_minus_1, iv] + v[jv, iv])
        
        Fe_v, Fw_v = rho * dy * U_E_v, rho * dy * U_W_v
        Fn_v, Fs_v = rho * dx * V_N_v, rho * dx * V_S_v
        
        aE_v = D_x + np.maximum(-Fe_v, 0)
        aW_v = D_x + np.maximum(Fw_v, 0)
        aN_v = D_y + np.maximum(-Fn_v, 0)
        aS_v = D_y + np.maximum(Fs_v, 0)
        aP_v = aE_v + aW_v + aN_v + aS_v + (Fe_v - Fw_v + Fn_v - Fs_v)
        aP_rel_v = aP_v / alpha_v
        
        dFe_v = np.abs(Fe_v) * 0.5 * (v[jv, iv_plus_1] - v[jv, iv])
        dFw_v = np.abs(Fw_v) * 0.5 * (v[jv, iv] - v[jv, iv_minus_1])
        dFn_v = np.abs(Fn_v) * 0.5 * (v[jv_plus_1, iv] - v[jv, iv])
        dFs_v = np.abs(Fs_v) * 0.5 * (v[jv, iv] - v[jv_minus_1, iv])
        
        Sv_DC = gamma * (dFw_v - dFe_v + dFs_v - dFn_v)
        
        b_v = (p[jv, iv] - p[jv_plus_1, iv]) * dx + (1 - alpha_v) * aP_rel_v * v[jv, iv] + Sv_DC
        d_v[jv, iv] = dx / aP_rel_v
        
        for _ in range(3):
            v[jv, iv] = (aE_v * v[jv, iv_plus_1] + aW_v * v[jv, iv_minus_1] + 
                         aN_v * v[jv_plus_1, iv] + aS_v * v[jv_minus_1, iv] + b_v) / aP_rel_v
            apply_bcs(u, v)
            
        mass_src = (rho * dy * (u[jp, ip_minus_1] - u[jp, ip]) + 
                    rho * dx * (v[jp_minus_1, ip] - v[jp, ip]))
        
        aE_p = rho * dy * d_u[jp, ip]
        aW_p = rho * dy * d_u[jp, ip_minus_1]
        aN_p = rho * dx * d_v[jp, ip]
        aS_p = rho * dx * d_v[jp_minus_1, ip]
        aP_p = aE_p + aW_p + aN_p + aS_p
        
        pc.fill(0.0)
        for _ in range(40):
            pc[jp, ip] = (aE_p * pc[jp, ip_plus_1] + aW_p * pc[jp, ip_minus_1] + 
                          aN_p * pc[jp_plus_1, ip] + aS_p * pc[jp_minus_1, ip] + mass_src) / aP_p
            
            pc[:, 0] = pc[:, 1]
            pc[:, nx + 1] = pc[:, nx]
            pc[0, :] = pc[1, :]
            pc[ny + 1, :] = pc[ny, :]
            pc[1, 1] = 0.0
            
        p[jp, ip] += alpha_p * pc[jp, ip]
        u[ju, iu] += d_u[ju, iu] * (pc[ju, iu] - pc[ju, iu_plus_1])
        v[jv, iv] += d_v[jv, iv] * (pc[jv, iv] - pc[jv_plus_1, iv])
        
        apply_bcs(u, v)
        max_div = np.max(np.abs(mass_src))
        
        if max_div < 1e-6:
            break
            
    return u, v

Lx, Ly = 1.0, 1.0
nx, ny = 128, 128
dx, dy = Lx / nx, Ly / ny

u_400, v_400 = solve_cavity(400.0)
u_1000, v_1000 = solve_cavity(1000.0)

xc = np.linspace(dx / 2, Lx - dx / 2, nx)
yc = np.linspace(dy / 2, Ly - dy / 2, ny)

Uc_400 = 0.5 * (u_400[1:ny + 1, 0:nx] + u_400[1:ny + 1, 1:nx + 1])
Vc_400 = 0.5 * (v_400[0:ny, 1:nx + 1] + v_400[1:ny + 1, 1:nx + 1])
U_mag_400 = np.sqrt(Uc_400**2 + Vc_400**2)

plt.figure(figsize=(7, 6))
strm1 = plt.streamplot(xc, yc, Uc_400, Vc_400, color=U_mag_400, cmap='turbo', linewidth=1.2, density=2.0)
plt.colorbar(strm1.lines, label='Velocity Magnitude (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()

Uc_1000 = 0.5 * (u_1000[1:ny + 1, 0:nx] + u_1000[1:ny + 1, 1:nx + 1])
Vc_1000 = 0.5 * (v_1000[0:ny, 1:nx + 1] + v_1000[1:ny + 1, 1:nx + 1])
U_mag_1000 = np.sqrt(Uc_1000**2 + Vc_1000**2)

plt.figure(figsize=(7, 6))
strm2 = plt.streamplot(xc, yc, Uc_1000, Vc_1000, color=U_mag_1000, cmap='turbo', linewidth=1.2, density=2.0)
plt.colorbar(strm2.lines, label='Velocity Magnitude (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()

ghia_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
ghia_u_400 = np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000])
ghia_u_1000 = np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000])

ghia_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
ghia_v_400 = np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000])
ghia_v_1000 = np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000])

u_plot_400 = np.concatenate(([0.0], u_400[1:ny + 1, nx // 2], [1.0]))
u_plot_1000 = np.concatenate(([0.0], u_1000[1:ny + 1, nx // 2], [1.0]))
yc_plot = np.concatenate(([0.0], yc, [Ly]))

v_plot_400 = np.concatenate(([0.0], v_400[ny // 2, 1:nx + 1], [0.0]))
v_plot_1000 = np.concatenate(([0.0], v_1000[ny // 2, 1:nx + 1], [0.0]))
xc_plot = np.concatenate(([0.0], xc, [Lx]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(u_plot_400, yc_plot, 'b-', label='Re=400 (Present)')
ax1.plot(ghia_u_400, ghia_y, 'bo', mfc='none', label='Re=400 (Ghia)')
ax1.plot(u_plot_1000, yc_plot, 'r-', label='Re=1000 (Present)')
ax1.plot(ghia_u_1000, ghia_y, 'rs', mfc='none', label='Re=1000 (Ghia)')
ax1.set_xlabel('u (m/s)')
ax1.set_ylabel('y (m)')
ax1.legend()

ax2.plot(xc_plot, v_plot_400, 'b-', label='Re=400 (Present)')
ax2.plot(ghia_x, ghia_v_400, 'bo', mfc='none', label='Re=400 (Ghia)')
ax2.plot(xc_plot, v_plot_1000, 'r-', label='Re=1000 (Present)')
ax2.plot(ghia_x, ghia_v_1000, 'rs', mfc='none', label='Re=1000 (Ghia)')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('v (m/s)')
ax2.legend()

plt.tight_layout()
plt.show()
