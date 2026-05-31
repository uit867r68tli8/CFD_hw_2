import numpy as np
import matplotlib.pyplot as plt

def solve_cavity(Re):
    Lx, Ly = 1.0, 1.0
    nx, ny = 128, 128
    dx, dy = Lx / nx, Ly / ny
    U_Lid = 1.0
    rho = 1.0
    mu = U_Lid * Lx * rho / Re
    
    alpha_u, alpha_v, alpha_p = 0.5, 0.5, 0.3
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
    
    for step in range(1, max_iter + 1):
        gamma = min(step / 3000.0, 1.0)
        
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
        aP = aE + aW + aN + aS + np.maximum(Fe - Fw + Fn - Fs, 0)
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
        aP_v = aE_v + aW_v + aN_v + aS_v + np.maximum(Fe_v - Fw_v + Fn_v - Fs_v, 0)
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
        
        if step % 500 == 0:
            print(f"Iter: {step:5d} | Div: {max_div:.3e} | Gamma: {gamma:.2f}")
            
        if max_div < 2e-5 and step > 3000:
            break
            
    return u, v

Lx, Ly = 1.0, 1.0
nx, ny = 128, 128
dx, dy = Lx / nx, Ly / ny

u_5000, v_5000 = solve_cavity(5000.0)

xc = np.linspace(dx / 2, Lx - dx / 2, nx)
yc = np.linspace(dy / 2, Ly - dy / 2, ny)

Uc_5000 = 0.5 * (u_5000[1:ny + 1, 0:nx] + u_5000[1:ny + 1, 1:nx + 1])
Vc_5000 = 0.5 * (v_5000[0:ny, 1:nx + 1] + v_5000[1:ny + 1, 1:nx + 1])
U_mag_5000 = np.sqrt(Uc_5000**2 + Vc_5000**2)

plt.figure(figsize=(7, 6))
strm = plt.streamplot(xc, yc, Uc_5000, Vc_5000, color=U_mag_5000, cmap='turbo', linewidth=1.2, density=2.0)
plt.colorbar(strm.lines, label='Velocity Magnitude (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.show()
