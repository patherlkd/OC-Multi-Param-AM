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

# --- Model Parameters (Schuttler et al. 2025) ---
MU = 1.0
D = 1.0           # Passive diffusion
D_PRIME = 1.0     # Active diffusion parameter
TAU = 0.525       # Persistence time

# --- Control Constants ---
K_STIFFNESS = 1.0 # Fixed Stiffness alpha_1
A2_START = 0.0    # Initial Trap Center
A2_END = 1.0      # Final Trap Center (Delta l)

# --- Optimization Parameters ---
PROTOCOL_NODES = 7500     # Resolution of control protocol
PADDING_RATIO = 0.1       # Padding to capture boundary jumps
DT = 0.001                # Timestep
LEARNING_RATE = 0.01      # Adam Step size
MAX_ITER = 10000          # Iterations

# --- Smoothness Penalty ---
REG_STRENGTH = 1e-4       # Tikhonov regularization strength

# --- Sweep Parameters ---
TP_SINGLE = 3.0                                      # Single fixed duration
a = np.arange(-10.0, 10.0, 0.3)                      # Array of measured velocities
b = np.array([-1., 0., 0.72, 1.6])                   # Array of measured velocities
V0_ARRAY = np.sort(np.concatenate((a, b), axis=0))[::-1]


OUTPUT_DIR = 'optimal_work_closed_loop_schuttler_test'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# 2. DIFFERENTIABLE PHYSICS ENGINE (JAX)
# =============================================================================

@jax.jit
def interpolate_protocol(a2_nodes, t_fine, tp):
    """
    Interpolates trap center alpha_2 nodes onto fine grid and applies padding.
    """
    # Guarantee that control nodes exactly span [0, tp]
    t_nodes = jnp.linspace(0, tp, len(a2_nodes))
    a2_opt = jnp.interp(t_fine, t_nodes, a2_nodes)

    mask_pre = t_fine < 0
    mask_post = t_fine > tp

    # Pad with fixed boundary values. 
    # At t=0 exactly, mask_pre is False, so it takes a2_nodes[0].
    # At t=tp exactly, mask_post is False, so it takes a2_nodes[-1].
    a2_fine = jnp.where(mask_pre, A2_START,
              jnp.where(mask_post, A2_END, a2_opt))
    return a2_fine

def dynamics_step(state, inputs):
    """
    Evolves the conditional mean m1 = <x>|v0.
    CRITICAL CHANGE: State is strictly locked to initial condition for t < 0.
    """
    m1 = state
    a2, t, dt, v0 = inputs

    # Conditional mean velocity decays ONLY for t >= 0
    mv = jnp.where(t >= 0.0, v0 * jnp.exp(-t / TAU), 0.0)

    # ODE: dot_m1 = -mu*k*(m1 - a2) + mv
    # MASK: Set derivative to 0 for t < 0 to prevent drift/decay during pre-padding.
    dm1 = jnp.where(t >= 0.0, -MU * K_STIFFNESS * (m1 - a2) + mv, 0.0)

    m1_next = m1 + dt * dm1
    return m1_next, m1

@jax.jit
def loss_function(a2_nodes, t_fine, dt, tp, v0):
    """
    Calculates Work J. Captures jumps via padded diffs and applies regularization.
    """
    # 1. Protocol Construction
    a2_fine = interpolate_protocol(a2_nodes, t_fine, tp)

    # 2. Initial Condition for m1 (Conditional Mean at t=0)
    # Applied at start of padding. Since dm1=0 for t<0, it stays constant until t=0.
    denom = 1.0 + MU * K_STIFFNESS * TAU
    m1_init = (TAU * v0) / denom

    # 3. Simulation
    scan_inputs = (a2_fine, t_fine, jnp.full_like(t_fine, dt), jnp.full_like(t_fine, v0))
    _, m1_traj = jax.lax.scan(dynamics_step, m1_init, scan_inputs)

    # 4. Work Calculation
    da2 = jnp.diff(a2_fine) / dt

    # Align arrays: da2 is at step t->t+1. We align with the state at end of step.
    term = K_STIFFNESS * da2 * (a2_fine[1:] - m1_traj[1:])

    # Integral
    total_work = jnp.sum(term) * dt

    # 5. Regularization (Tikhonov penalty)
    reg_term = REG_STRENGTH * jnp.sum(da2**2) * dt

    return total_work + reg_term

# JIT Gradients
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

def optimize_protocol(tp, v0):
    """Runs Adam optimization for a specific tp and v0."""
    pad_time = PADDING_RATIO * tp
    
    # --- EXACT GRID CONSTRUCTION ---
    # We construct the arrays explicitly to guarantee exact floats at 0.0 and tp.
    t_pre = jnp.arange(-pad_time, 0.0, DT)
    t_active = jnp.arange(0.0, tp + DT, DT) # Includes tp
    t_post = jnp.arange(tp + DT, tp + pad_time + DT, DT)
    
    t_fine = jnp.concatenate((t_pre, t_active, t_post))

    # Initialize linear ramp
    a2_nodes = np.linspace(A2_START, A2_END, PROTOCOL_NODES)
    adam = AdamState(a2_nodes)

    for i in range(MAX_ITER):
        cost, grads = loss_and_grad(a2_nodes, t_fine, DT, tp, v0)

        # Update
        a2_nodes, adam = update_adam(a2_nodes, np.array(grads), adam, lr=LEARNING_RATE)
        
        # NOTE: Bounding constraints (np.clip) REMOVED. 
        # The optimal trap center is theoretically unbounded and must be allowed 
        # to overshoot/undershoot [0, 1] to properly "catch" or "whip" the particle.

    # Calculate final PHYSICAL work (subtract regularizer)
    a2_fine = interpolate_protocol(a2_nodes, t_fine, tp)
    da2 = jnp.diff(a2_fine) / DT
    reg_val = REG_STRENGTH * jnp.sum(da2**2) * DT
    physical_work = float(cost - reg_val)

    return t_fine, a2_nodes, physical_work

# =============================================================================
# 4. MAIN EXECUTION (Sweep over v0)
# =============================================================================

print(f"{'Tp':<6} | {'v0':<6} | {'Work':<10}")
print("-" * 30)

plot_data = []
results_summary = []

for v0 in V0_ARRAY:
    t_fine, a2_nodes, work = optimize_protocol(TP_SINGLE, v0)
    print(f"{TP_SINGLE:<6.1f} | {v0:<6.1f} | {work:<10.4f}")

    # Reconstruct fine protocol for plotting
    a2_fine = interpolate_protocol(a2_nodes, t_fine, TP_SINGLE)

    plot_data.append({
        'v0': v0,
        't': np.array(t_fine),
        'a2': np.array(a2_fine)
    })

    results_summary.append((v0, work))

    # Save protocol
    np.savetxt(os.path.join(OUTPUT_DIR, f"closed_loop_tp_{TP_SINGLE}_v{v0:.2f}.txt"),
               np.column_stack((t_fine, a2_fine)), header="Time\tAlpha2")

# Save summary of Work vs v0
np.savetxt(os.path.join(OUTPUT_DIR, "work_vs_v0.txt"), results_summary, header="v0\tWork")

# =============================================================================
# 5. PLOTTING
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define colors based on v0 values
norm = plt.Normalize(min(V0_ARRAY), max(V0_ARRAY))
cmap = cm.coolwarm
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plot 1: Protocols vs Time
for data in plot_data:
    v0 = data['v0']
    t = data['t']
    a2 = data['a2']
    t_norm = t / TP_SINGLE

    color = cmap(norm(v0))

    ax1.plot(t_norm, a2, color=color, linewidth=2.5, label=f'$v_0={v0:.1f}$')

ax1.set_title(rf"Closed-Loop Protocols $\alpha_2(t)$ ($t_p={TP_SINGLE}$)", fontsize=14)
ax1.set_xlabel(r"Normalized Time $t/t_p$", fontsize=12)
ax1.set_ylabel(r"Trap Center $\alpha_2$", fontsize=12)
ax1.axhline(A2_START, color='k', linestyle=':', alpha=0.3)
ax1.axhline(A2_END, color='k', linestyle=':', alpha=0.3)
ax1.axvline(0, color='k', linestyle='-', alpha=0.2)
ax1.axvline(1, color='k', linestyle='-', alpha=0.2)
ax1.grid(True, alpha=0.3)
cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label(r'Measured Velocity $v_0$', fontsize=12)

# Plot 2: Work vs v0
results_summary.sort(key=lambda x: x[0])
v0s, works = zip(*results_summary)
ax2.plot(v0s, works, 'o-', color='black', linewidth=2)
ax2.set_title(rf"Total Work vs Measured $v_0$ ($t_p={TP_SINGLE}$)", fontsize=14)
ax2.set_xlabel(r"Measured Velocity $v_0$", fontsize=12)
ax2.set_ylabel(r"Work $\mathcal{J}$", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'closed_loop_sweep.png'), dpi=300)
print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'closed_loop_sweep.png')}")
plt.show()
