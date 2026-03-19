import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Enable 64-bit precision for physical accuracy
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# --- Model Parameters (Active OU Particle) ---
MU = 1.0         # Mobility
D = 1.0          # Passive diffusion
D_PRIME = 2.0    # Active diffusion
TAU = 1.0        # Persistence time

# --- Control Boundary Conditions ---
EPS_START = 1.0  # Initial Stiffness
EPS_END = 5.0    # Final Stiffness

# --- Optimization Parameters ---
PROTOCOL_NODES = 1000       # Resolution of control protocol
PADDING_RATIO = 0.1        # Padding to capture boundary jumps
DT = 0.0005                 # Timestep
LEARNING_RATE = 0.01       # Adam Step size
MAX_ITER = 5000            # Iterations

# --- contral cost Smoothnes Penalty ---
REG_STRENGTH = 1e-4        # Tikhonov regularization strength

custom_vals = [1.08, 2.81, 3.56, 5.74, 100.0]

# Generate the ranges
r1 = np.arange(0.1, 1.0, 0.1)
r2 = np.arange(1.1, 10.0, 1.0)
r3 = np.arange(11.0, 100.0, 10.0)

# Combine, remove duplicates, and sort
TPS = np.sort(np.unique(np.concatenate([custom_vals, r1, r2, r3])))
TPS = r1

OUTPUT_DIR = 'optimal_heat_active_nonstationary_casert'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# 2. DIFFERENTIABLE PHYSICS ENGINE (JAX)
# =============================================================================

@jax.jit
def interpolate_protocol(alpha_nodes, t_fine, tp):
    t_nodes = jnp.linspace(0, tp, len(alpha_nodes))
    alpha_opt = jnp.interp(t_fine, t_nodes, alpha_nodes)

    mask_pre = t_fine < 0
    mask_post = t_fine > tp

    alpha_fine = jnp.where(mask_pre, EPS_START,
                 jnp.where(mask_post, EPS_END, alpha_opt))
    return alpha_fine

def compute_physics_trajectory(alpha_nodes, t_fine, dt, tp):
    """
    Runs dynamics and computes the INSTANTANEOUS cost density.
    Returns: m3_traj, cost_density, m3_init, alpha_fine
    """
    alpha_fine = interpolate_protocol(alpha_nodes, t_fine, tp)

    def dynamics_step(state, alpha):
        m2, m3 = state
        dm2 = -(MU * alpha + 1.0/TAU) * m2 + D_PRIME/TAU
        dm3 = -2.0 * MU * alpha * m3 + 2.0 * m2 + 2.0 * D
        m2_next = m2 + dt * dm2
        m3_next = m3 + dt * dm3
        return (m2_next, m3_next), (m2, m3)

    denom = 1.0 + MU * EPS_START * TAU
    m2_init = D_PRIME / denom
    m3_init = (D_PRIME + D * denom) / (MU * EPS_START * denom)
    state_init = (m2_init, m3_init)

    _, (m2_traj, m3_traj) = jax.lax.scan(dynamics_step, state_init, alpha_fine)

    # --- Cost Density Calculation ---
    d_alpha = jnp.diff(alpha_fine) / dt
    term_work = (d_alpha / 2.0) * m3_traj[1:]

    # Active term (Masked to [0, tp])
    term_diss = - alpha_fine[1:] * m2_traj[1:]

    t_eval = t_fine[1:]
    mask_active = (t_eval >= 0.0) & (t_eval <= tp)

    # Combine terms
    integrand = term_work + (term_diss * mask_active)

    return m3_traj, integrand, m3_init, alpha_fine


@jax.jit
def loss_function(alpha_nodes, t_fine, dt, tp):
    """
    Scalar loss for optimization (Includes Tikhonov regularization).
    """
    m3_traj, integrand, m3_init, alpha_fine = compute_physics_trajectory(alpha_nodes, t_fine, dt, tp)

    # 1. Integral Part
    integral_J = jnp.sum(integrand) * dt

    # 2. Boundary Terms
    B_housekeeping = (D_PRIME / (MU * TAU)) * tp

    U_start = 0.5 * EPS_START * m3_init
    m3_final = jnp.interp(tp, t_fine, m3_traj)
    U_end = 0.5 * EPS_END * m3_final

    # 3. Regularization (Tikhonov penalty on control rate)
    d_alpha = jnp.diff(alpha_fine) / dt
    reg_term = REG_STRENGTH * jnp.sum(d_alpha**2) * dt

    total_cost = integral_J + B_housekeeping + U_start - U_end + reg_term

    return total_cost

@jax.jit
def get_accumulated_cost(alpha_nodes, t_fine, dt, tp):
    """
    Returns the cumulative cost vs time array.
    """
    m3_traj, integrand, m3_init, _ = compute_physics_trajectory(alpha_nodes, t_fine, dt, tp)

    # Cumulative Sum of the integral part
    cumulative_integral = jnp.cumsum(integrand) * dt

    U_start = 0.5 * EPS_START * m3_init

    accumulated = U_start + cumulative_integral

    # Add Housekeeping linearly
    P_hk = (D_PRIME / (MU * TAU))
    t_axis = t_fine[1:]

    # Mask housekeeping to start at t=0
    t_active = jnp.maximum(0.0, jnp.minimum(tp, t_axis))
    accumulated += P_hk * t_active

    return t_axis, accumulated

# JIT Compile gradients
loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

# =============================================================================
# 3. OPTIMIZATION LOOP
# =============================================================================

class AdamState:
    def __init__(self, params):
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

def update_adam(params, grads, state, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    state.t += 1
    state.m = b1 * state.m + (1 - b1) * grads
    state.v = b2 * state.v + (1 - b2) * (grads**2)
    m_hat = state.m / (1 - b1**state.t)
    v_hat = state.v / (1 - b2**state.t)
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, state

final_results = []
plot_data = []

print(f"{'Tp':<6} | {'Iter':<8} | {'Heat <J>':<12}")
print("-" * 40)

for tp in TPS:
    # --- EXACT GRID ALIGNMENT ---
    # Critically ensures t_fine hits exactly 0.0 and tp so the regularizer 
    # accurately captures the boundary jump penalty.
    pad_time = PADDING_RATIO * tp
    n_active = int(round(tp / DT)) + 1
    t_active = jnp.linspace(0.0, tp, n_active)
    
    n_pad = int(round(pad_time / DT))
    t_pre = jnp.linspace(-pad_time, 0.0, n_pad + 1)[:-1]      # Everything strictly < 0
    t_post = jnp.linspace(tp, tp + pad_time, n_pad + 1)[1:]   # Everything strictly > tp
    
    t_fine = jnp.concatenate((t_pre, t_active, t_post))
    dt_actual = tp / (n_active - 1)

    # Initialization
    alpha_nodes = np.linspace(EPS_START, EPS_END, PROTOCOL_NODES)
    adam = AdamState(alpha_nodes)

    # Training
    for i in range(MAX_ITER):
        cost, grads = loss_and_grad(alpha_nodes, t_fine, dt_actual, tp)

        cost_val = float(cost)
        grads_val = np.array(grads)

        if np.isnan(cost_val) or cost_val > 1e6:
            print(f"{tp:<6.3f} | {i:<8} | NaN/Div")
            break

        # Update
        alpha_nodes, adam = update_adam(alpha_nodes, grads_val, adam, lr=LEARNING_RATE)
        alpha_nodes = np.clip(alpha_nodes, min(EPS_START, EPS_END), max(EPS_START, EPS_END))

    # --- Generate Results for Saving ---

    # 1. Protocol & True Physical Cost (Subtracting Regularizer)
    alpha_final_fine = interpolate_protocol(alpha_nodes, t_fine, tp)
    d_alpha_final = jnp.diff(alpha_final_fine) / dt_actual
    reg_val = REG_STRENGTH * jnp.sum(d_alpha_final**2) * dt_actual
    
    physical_heat = float(cost_val - reg_val)
    final_results.append((tp, physical_heat))
    
    print(f"{tp:<6.3f} | {MAX_ITER:<8} | {physical_heat:.6f}")

    # 2. Accumulated Cost
    t_acc, acc_cost = get_accumulated_cost(alpha_nodes, t_fine, dt_actual, tp)

    plot_data.append({
        'tp': tp,
        't': np.array(t_fine),
        'alpha': np.array(alpha_final_fine),
        't_acc': np.array(t_acc),
        'acc_cost': np.array(acc_cost)
    })

    # --- SAVE FILES ---

    filename_prot = os.path.join(OUTPUT_DIR, f"active_ns_protocol_tp_{tp:.3f}.txt")
    np.savetxt(filename_prot, np.column_stack((t_fine, alpha_final_fine)), header="Time\tAlpha1")

    filename_cost = os.path.join(OUTPUT_DIR, f"accumulated_cost_tp_{tp:.3f}.txt")
    np.savetxt(filename_cost, np.column_stack((t_acc, acc_cost)), header="Time\tAccumulated_Cost")

print("-" * 40)

# --- Save Summary Data (Total Heat vs Tp) ---
summary_filename = os.path.join(OUTPUT_DIR, "total_heat_vs_tp.txt")
np.savetxt(summary_filename, final_results, header="Tp\tTotal_Heat", fmt="%.6f")
print(f"Summary saved to: {summary_filename}")


# =============================================================================
# 4. PLOTTING
# =============================================================================

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
plt.rcParams['axes.grid'] = False 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax in (ax1, ax2):
    ax.tick_params(direction='in', top=True, right=True, which='both', length=6)

# Sample a few for the left plot so it doesn't get cluttered
display_tps = [1.08, 2.81, 3.56, 5.74, 100.0]
colors = cm.viridis(np.linspace(0.1, 0.9, len(display_tps)))

# Plot 1: Protocols
for data in plot_data:
    tp = data['tp']
    if not any(np.isclose(tp, d_tp) for d_tp in display_tps):
        continue
        
    idx = next(i for i, d_tp in enumerate(display_tps) if np.isclose(tp, d_tp))
    t = data['t']
    a = data['alpha']
    t_norm = t / tp

    ax1.plot(t_norm, a, color=colors[idx], linewidth=2.5, label=rf'$t_p={tp:.2f}$')

ax1.axvline(0, color='k', linestyle='-', alpha=0.3, lw=1)
ax1.axvline(1, color='k', linestyle='-', alpha=0.3, lw=1)
ax1.set_xlim(-0.1, 1.1)
ax1.set_title(r"\textbf{A} Optimal Stiffness (Active, Non-Stationary)")
ax1.set_xlabel(r"Normalized Time $\tau/t_p$")
ax1.set_ylabel(r"Stiffness $\alpha_1(\tau)$")
ax1.legend(loc='best', fontsize=10, frameon=False, ncol=1)

# Plot 2: Total Heat Scaling
tps, heats = zip(*final_results)
ax2.plot(tps, heats, 'o-', color='crimson', linewidth=2, markeredgecolor='k', markersize=6)
ax2.set_xscale('log')

if min(heats) > 0:
    ax2.set_yscale('log')
    
ax2.set_title(r"\textbf{B} Total Heat Dissipation vs Duration")
ax2.set_xlabel(r"Protocol Duration $t_p$")
ax2.set_ylabel(r"Total Heat $\langle \mathcal{J}_{\mathrm{diss}} \rangle$")

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'active_heat_ns_protocols.pdf')
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()
