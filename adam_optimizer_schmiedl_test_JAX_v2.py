import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Enable 64-bit precision for physical accuracy and stability
jax.config.update("jax_enable_x64", True)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
MU = 1.0
D = 1.0

# Boundary Conditions
EPS_START = 1.0
EPS_END = 5.0

# Optimization Settings
PROTOCOL_NODES = 200      # Optimizable points within [0, tp]
PADDING_RATIO = 0.1       # 10% padding before and after to handle boundary jumps
DT = 0.0005               # Simulation timestep
LEARNING_RATE = 0.01      # Adam Step size
MAX_ITER = 4000           # Iterations per tp

# Generate 31 points from 0.1 to 100.0 (10 points per decade)
TPS_ALL = np.logspace(-1, 2, 31)
# Specifically highlight these in the left panel
TPS_HIGHLIGHT = [0.1, 1.0, 10.0]

OUTPUT_DIR = 'optimal_schmiedl_test_JAX'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Comparison strengths
REG_STRENGTH_ON = 1e-4
REG_STRENGTH_OFF = 0.0

# =============================================================================
# 2. EXACT ANALYTICAL SOLUTION (Schmiedl & Seifert 2007)
# =============================================================================

def exact_ss2007_protocol(tau, tp, lam_i, lam_f):
    """
    Computes the exact optimal protocol from Schmiedl & Seifert (2007) Eq. 18.
    """
    term_sqrt = np.sqrt(1 + 2 * lam_i * tp + lam_f * lam_i * tp**2)
    c2_tp = (-1 - lam_f * tp + term_sqrt) / (2 + lam_f * tp)
    c2 = c2_tp / tp
    
    lam_opt = (lam_i - c2 * (1 + c2 * tau)) / ((1 + c2 * tau)**2)
    return lam_opt

def exact_ss2007_work(tp, lam_i, lam_f):
    """
    Computes the exact minimal work from Schmiedl & Seifert (2007) Eq. 16.
    """
    term_sqrt = np.sqrt(1 + 2 * lam_i * tp + lam_f * lam_i * tp**2)
    c2_tp = (-1 - lam_f * tp + term_sqrt) / (2 + lam_f * tp)
    
    w_min = ((c2_tp)**2 / (lam_i * tp)) - np.log(1 + c2_tp) + 0.5 * (lam_f / lam_i) * (1 + c2_tp)**2 - 0.5
    return w_min

# =============================================================================
# 3. DIFFERENTIABLE PHYSICS ENGINE (JAX)
# =============================================================================

@jax.jit
def interpolate_protocol(alpha_nodes, t_fine, tp):
    t_nodes = jnp.linspace(0, tp, len(alpha_nodes))
    mask_pre = t_fine < 0
    mask_post = t_fine > tp

    alpha_opt = jnp.interp(t_fine, t_nodes, alpha_nodes)
    alpha_fine = jnp.where(mask_pre, EPS_START,
                 jnp.where(mask_post, EPS_END, alpha_opt))
    return alpha_fine

def dynamics_step(state, inputs):
    m3 = state
    alpha, t, dt = inputs
    
    dm3 = 2.0 * D - 2.0 * MU * alpha * m3
    dm3 = jnp.where(t >= 0.0, dm3, 0.0)
    
    m3_next = m3 + dt * dm3
    return m3_next, m3

@jax.jit
def loss_function(alpha_nodes, t_fine, dt, tp, reg_strength):
    alpha_fine = interpolate_protocol(alpha_nodes, t_fine, tp)
    
    m3_init = D / (MU * EPS_START) 
    
    scan_inputs = (alpha_fine, t_fine, jnp.full_like(t_fine, dt))
    _, m3_traj = jax.lax.scan(dynamics_step, m3_init, scan_inputs)

    # Stratonovich Work
    m3_mid = 0.5 * (m3_traj[1:] + m3_traj[:-1])
    da = jnp.diff(alpha_fine)
    
    J_work = jnp.sum(0.5 * m3_mid * da)

    # Regularization (Tikhonov)
    reg_term = reg_strength * jnp.sum((da / dt)**2) * dt

    return J_work + reg_term

loss_and_grad = jax.jit(jax.value_and_grad(loss_function, argnums=0))

# =============================================================================
# 4. OPTIMIZATION LOOP
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

def run_optimization(tp, reg_strength):
    pad_time = PADDING_RATIO * tp
    t_total = tp + 2 * pad_time
    num_steps = int(t_total / DT)
    dt_actual = t_total / num_steps
    t_fine = jnp.linspace(-pad_time, tp + pad_time, num_steps)

    alpha_nodes = np.linspace(EPS_START, EPS_END, PROTOCOL_NODES)
    adam = AdamState(alpha_nodes)

    for i in range(MAX_ITER):
        cost, grads = loss_and_grad(alpha_nodes, t_fine, dt_actual, tp, reg_strength)
        alpha_nodes, adam = update_adam(alpha_nodes, np.array(grads), adam, lr=LEARNING_RATE)
        alpha_nodes = np.clip(alpha_nodes, min(EPS_START, EPS_END), max(EPS_START, EPS_END))

    a1_f = interpolate_protocol(alpha_nodes, t_fine, tp)
    da1 = jnp.diff(a1_f)
    reg_val = reg_strength * jnp.sum((da1 / dt_actual)**2) * dt_actual
    num_work = float(cost - reg_val)
    
    return t_fine, a1_f, num_work

results = []

print(f"Optimizing {len(TPS_ALL)} configurations...")
print(f"{'Tp':<8} | {'Reg Work':<10} | {'NoReg Work':<10} | {'Exact Work':<10}")
print("-" * 50)

for tp in TPS_ALL:
    # 1. Run with Regularization
    t_fine, a1_f_reg, num_work_reg = run_optimization(tp, REG_STRENGTH_ON)
    
    # 2. Run without Regularization
    _, a1_f_noreg, num_work_noreg = run_optimization(tp, REG_STRENGTH_OFF)
    
    # 3. Exact Work
    ex_work = exact_ss2007_work(tp, EPS_START, EPS_END)

    # Save to file
    save_data_reg = np.column_stack([t_fine, a1_f_reg])
    save_data_noreg = np.column_stack([t_fine, a1_f_noreg])
    header = "t_fine stiffness"
    
    np.savetxt(os.path.join(OUTPUT_DIR, f"protocol_tp_{tp:.4f}_reg.txt"), save_data_reg, header=header, comments='# ')
    np.savetxt(os.path.join(OUTPUT_DIR, f"protocol_tp_{tp:.4f}_noreg.txt"), save_data_noreg, header=header, comments='# ')

    results.append({
        'tp': tp,
        't': np.array(t_fine),
        'alpha_reg': np.array(a1_f_reg),
        'alpha_noreg': np.array(a1_f_noreg),
        'work_reg': num_work_reg,
        'work_noreg': num_work_noreg,
        'ex_work': ex_work
    })
    
    print(f"{tp:<8.3f} | {num_work_reg:<10.4f} | {num_work_noreg:<10.4f} | {ex_work:<10.4f}")

# =============================================================================
# 5. PLOTTING (TWO PANELS)
# =============================================================================

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
plt.rcParams['axes.grid'] = False # Remove grid globally

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Formatting common to all axes (Inward ticks) ---
for ax in axs:
    ax.tick_params(direction='in', top=True, right=True, which='both', length=6)

# --- Panel A: Protocols (Selected Tps) ---
ax = axs[0]
colors = cm.viridis(np.linspace(0.1, 0.9, len(TPS_HIGHLIGHT)))

for i, target_tp in enumerate(TPS_HIGHLIGHT):
    data = next(item for item in results if np.isclose(item['tp'], target_tp))
    
    tp = data['tp']
    t_plot = data['t']
    t_norm = t_plot / tp

    # Plot Numerical JAX (With Regularization)
    ax.plot(t_norm, data['alpha_reg'] / EPS_START, color=colors[i], linestyle='-', linewidth=2.5, alpha=1.0, label=rf'Numerical $J_R > 0$ ($t_p={tp}$)')
    
    # Plot Numerical JAX (No Regularization)
    ax.plot(t_norm, data['alpha_noreg'] / EPS_START, color=colors[i], linestyle='-', linewidth=2.5, alpha=0.5, label=rf'Numerical $J_R = 0$ ($t_p={tp}$)')
    
    # Plot Exact Analytical (Schmiedl & Seifert)
    tau_exact = np.linspace(0.0001, tp - 0.0001, 200) # Strictly interior to show jumps
    lam_exact = exact_ss2007_protocol(tau_exact, tp, EPS_START, EPS_END)
    ax.plot(tau_exact / tp, lam_exact / EPS_START, color=colors[i], linestyle='--', linewidth=2.0, zorder=5, label=rf'Exact ($t_p={tp}$)')

ax.axvline(0, color='k', linestyle='-', alpha=0.3, lw=1)
ax.axvline(1, color='k', linestyle='-', alpha=0.3, lw=1)

ax.set_title(r"\textbf{A} Optimal Trap Stiffness Protocol")
ax.set_xlabel(r"Normalized Time $\tau/t_p$")
ax.set_ylabel(r"Scaled Stiffness $\alpha_1(\tau)/\alpha_1(0)$")
# Custom legend placement to keep it clean (3 columns)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='best', fontsize=9, frameon=False, ncol=3)

# --- Panel B: Total Work vs tp ---
ax2 = axs[1]

# Plot Analytical curve (smooth)
tps_smooth = np.logspace(-1, 2, 200)
ex_works_smooth = [exact_ss2007_work(t, EPS_START, EPS_END) for t in tps_smooth]
ax2.plot(tps_smooth, ex_works_smooth, 'k--', lw=2.0, label='Exact (S\&S 2007)')

# Plot Numerical Points (all evaluated Tps)
num_tps = [r['tp'] for r in results]
work_reg = [r['work_reg'] for r in results]
work_noreg = [r['work_noreg'] for r in results]

ax2.scatter(num_tps, work_reg, color='crimson', s=60, marker='o', edgecolor='k', zorder=5, label=r'Numerical ($J_R > 0$)')
ax2.scatter(num_tps, work_noreg, color='dodgerblue', s=60, marker='s', edgecolor='k', zorder=4, label=r'Numerical ($J_R = 0$)')

ax2.set_xscale('log')
ax2.set_title(r"\textbf{B} Minimal Work vs Duration")
ax2.set_xlabel(r"Protocol Duration $t_p$")
ax2.set_ylabel(r"Minimal Total Work $\langle J_W(\alpha^*_1) \rangle$")
ax2.legend(frameon=False)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'ss2007_validation_2panel.pdf')
plt.savefig(save_path, dpi=300)
plt.show()
