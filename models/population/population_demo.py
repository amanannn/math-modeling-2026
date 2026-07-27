#!/usr/bin/env python3
"""
Purpose: Population dynamics and epidemic modeling demonstration.
         Covers three core models from the population biology module:
           1. Logistic growth -- single species with carrying capacity K
           2. Lotka-Volterra predator-prey -- two-species interaction with phase portrait
           3. SIR epidemic model -- compartmental model with R0 calculation and scenario comparison

Input:   None (all parameters hardcoded for self-contained demonstration)
Output:  Console output of key numerical results under clear section headers
         Multi-panel figure saved to output/population_result.png at 300 dpi

Usage:   python population_demo.py

Dependencies: numpy, scipy (integrate.solve_ivp), matplotlib (Agg backend)
"""

import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 1. LOGISTIC GROWTH MODEL  (single species, carrying capacity)
# ============================================================================
print("=" * 60)
print("  SECTION 1: Logistic Growth Model")
print("=" * 60)


def logistic(t, N, r, K):
    """Logistic growth: dN/dt = r * N * (1 - N / K)."""
    return r * N * (1 - N / K)


# Parameters
r = 0.5       # intrinsic growth rate
K = 100.0     # carrying capacity
N0 = 2.0       # initial population

sol_log = solve_ivp(logistic, [0, 20], [N0], args=(r, K),
                    method='RK45', max_step=0.05, dense_output=True)

t_fine_log = np.linspace(0, 20, 500)
N_num = sol_log.sol(t_fine_log)[0]
# Analytic solution
N_ana = K / (1 + (K / N0 - 1) * np.exp(-r * t_fine_log))

print(f"  Parameters:  r = {r},  K = {K},  N0 = {N0}")
print(f"  Final N (numerical): {N_num[-1]:.3f}")
print(f"  Final N (analytic):  {N_ana[-1]:.3f}")
print(f"  Carrying capacity K = {K}")
print()

# ============================================================================
# 2. LOTKA-VOLTERRA PREDATOR-PREY MODEL
# ============================================================================
print("=" * 60)
print("  SECTION 2: Lotka-Volterra Predator-Prey Model")
print("=" * 60)


def lotka_volterra(t, z, a, b, c, d):
    """Lotka-Volterra: dx/dt = a*x - b*x*y,  dy/dt = d*x*y - c*y."""
    x, y = z
    return [a * x - b * x * y, d * x * y - c * y]


# Parameters
a, b, c, d = 1.0, 0.1, 0.5, 0.02
x_eq = c / d          # 25.0
y_eq = a / b          # 10.0

print(f"  Parameters:  a = {a},  b = {b},  c = {c},  d = {d}")
print(f"  Equilibrium:  prey* = {x_eq:.1f},  predator* = {y_eq:.1f}")
print()

# Multiple initial conditions
init_conds = [(40, 9), (30, 15), (50, 5)]
lv_colors = ['tab:blue', 'tab:red', 'tab:green']

lv_solutions = []
for (x0, y0) in init_conds:
    sol = solve_ivp(lotka_volterra, [0, 150], [x0, y0],
                    args=(a, b, c, d), max_step=0.2, dense_output=True)
    lv_solutions.append(sol)

# Verify conservation quantity V = d*x - c*ln(x) + b*y - a*ln(y)
print("  Conservation quantity V(x,y) = d*x - c*ln(x) + b*y - a*ln(y):")
t_fine_lv = np.linspace(0, 150, 1000)
for i, (x0, y0) in enumerate(init_conds):
    y_fine = lv_solutions[i].sol(t_fine_lv)
    x_f, y_f = y_fine
    V = d * x_f - c * np.log(x_f) + b * y_f - a * np.log(y_f)
    print(f"    V({x0:2d},{y0:2d})   range = {V.max() - V.min():.2e}  "
          f"(max deviation from constant)")
print()

# ============================================================================
# 3. SIR EPIDEMIC MODEL
# ============================================================================
print("=" * 60)
print("  SECTION 3: SIR Epidemic Model")
print("=" * 60)


def sir(t, y, N, beta, gamma):
    """SIR model: dS/dt = -beta S I/N, dI/dt = beta S I/N - gamma I, dR/dt = gamma I."""
    S, I, R = y
    return [-beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I]


# Parameters
N_pop = 10000        # total population
beta = 0.35          # effective contact rate (per day)
gamma = 0.1          # recovery rate (per day)
R0_val = beta / gamma
I0 = 5
S0 = N_pop - I0

sol_sir = solve_ivp(sir, [0, 200], [S0, I0, 0],
                    args=(N_pop, beta, gamma), max_step=0.5, dense_output=True)

t_fine_sir = np.linspace(0, 200, 500)
S_f, I_f, R_f = sol_sir.sol(t_fine_sir)
peak_I = I_f.max()
peak_day = t_fine_sir[I_f.argmax()]
final_R = R_f[-1]

print(f"  Population N = {N_pop}")
print(f"  beta = {beta},  gamma = {gamma}")
print(f"  R0 = beta / gamma = {R0_val:.2f}")
print(f"  Peak infections: {peak_I:.0f}  on day {peak_day:.1f}")
print(f"  Final infected (ever): {final_R:.0f}  ({final_R / N_pop * 100:.1f}% of pop.)")
print(f"  Herd immunity threshold  h = 1 - 1/R0 = {(1 - 1 / R0_val) * 100:.1f}%")
print()

# ---------------------------------------------------------------------------
# Multi-scenario R0 sweep (for subplot 6)
# ---------------------------------------------------------------------------
gamma_fixed = 0.1
R0_list = [0.8, 1.5, 3.0, 5.0]
r0_colors = ['green', 'blue', 'orange', 'red']
r0_sweep_results = []

for R0_sc in R0_list:
    beta_sc = R0_sc * gamma_fixed
    sol_sc = solve_ivp(
        lambda t, y: [-beta_sc * y[0] * y[1] / N_pop,
                      beta_sc * y[0] * y[1] / N_pop - gamma_fixed * y[1],
                      gamma_fixed * y[1]],
        [0, 200], [N_pop - I0, I0, 0], max_step=0.5)
    r0_sweep_results.append(sol_sc)

# ============================================================================
# BUILD MULTI-PANEL FIGURE  (2 rows x 3 columns)
# ============================================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Population Dynamics & Epidemic Modeling', fontsize=16, y=0.98)

# ---- (1,1) Logistic Growth Time Series ------------------------------------
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(t_fine_log, N_num, 'b-', linewidth=2, label='Numerical')
ax1.plot(t_fine_log, N_ana, 'r--', linewidth=1.5, alpha=0.7, label='Analytic')
ax1.axhline(K, color='gray', linestyle=':', linewidth=1, label=f'K = {K}')
ax1.axhline(K / 2, color='orange', linestyle='--', linewidth=1, alpha=0.6,
            label=f'K/2 = {K/2}')
ax1.set_xlabel('Time t')
ax1.set_ylabel('Population N(t)')
ax1.set_title('Logistic Growth')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

# ---- (1,2) Lotka-Volterra Time Series -------------------------------------
ax2 = fig.add_subplot(2, 3, 2)
# Use the baseline solution (40, 9)
y_baseline = lv_solutions[0].sol(t_fine_lv)
ax2.plot(t_fine_lv, y_baseline[0], 'b-', linewidth=1.5, label='Prey x(t)')
ax2.plot(t_fine_lv, y_baseline[1], 'r-', linewidth=1.5, label='Predator y(t)')
ax2.set_xlabel('Time t')
ax2.set_ylabel('Population')
ax2.set_title('Lotka-Volterra Time Series')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ---- (1,3) Lotka-Volterra Phase Portrait ----------------------------------
ax3 = fig.add_subplot(2, 3, 3)
for i, (x0, y0) in enumerate(init_conds):
    y_fine = lv_solutions[i].sol(t_fine_lv)
    ax3.plot(y_fine[0], y_fine[1], color=lv_colors[i], linewidth=1.5,
             label=f'({x0}, {y0})')
ax3.plot(x_eq, y_eq, 'k*', markersize=12,
         label=f'Equilibrium ({x_eq:.0f}, {y_eq:.0f})')
ax3.set_xlabel('Prey x')
ax3.set_ylabel('Predator y')
ax3.set_title('LV Phase Portrait')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ---- (2,1) SIR Time Series ------------------------------------------------
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t_fine_sir, S_f, 'b-', linewidth=2, label='S (Susceptible)')
ax4.plot(t_fine_sir, I_f, 'r-', linewidth=2, label='I (Infectious)')
ax4.plot(t_fine_sir, R_f, 'g-', linewidth=2, label='R (Recovered)')
ax4.axvline(peak_day, color='gray', linestyle=':', alpha=0.7,
            label=f'Peak = day {peak_day:.0f}')
ax4.set_xlabel('Time (days)')
ax4.set_ylabel('Population')
ax4.set_title(f'SIR Model  (R0 = {R0_val:.2f})')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ---- (2,2) SIR Phase Portrait (S-I plane) ---------------------------------
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(S_f, I_f, 'm-', linewidth=1.5, label='Trajectory')
S_thresh = N_pop / R0_val
ax5.axvline(S_thresh, color='red', linestyle='--', linewidth=1.5,
            label=f'S* = N/R0 = {S_thresh:.0f}')
ax5.plot(S_f[0], I_f[0], 'go', markersize=7, label='Start')
ax5.plot(S_f[-1], I_f[-1], 'rs', markersize=7, label='End')
ax5.set_xlabel('S (Susceptible)')
ax5.set_ylabel('I (Infectious)')
ax5.set_title('SIR Phase Portrait (S-I)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ---- (2,3) R0 Scenario Comparison -----------------------------------------
ax6 = fig.add_subplot(2, 3, 6)
for R0_sc, color, sol_sc in zip(R0_list, r0_colors, r0_sweep_results):
    ax6.plot(sol_sc.t, sol_sc.y[1], color=color, linewidth=1.5,
             label=f'R0 = {R0_sc:.1f}')
    peak_idx = np.argmax(sol_sc.y[1])
    ax6.plot(sol_sc.t[peak_idx], sol_sc.y[1, peak_idx], 'o', color=color,
             markersize=5)
ax6.set_xlabel('Time (days)')
ax6.set_ylabel('Infectious I(t)')
ax6.set_title('R0 Comparison  (gamma = 0.1)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_dir = 'D:/虚拟C盘/数学建模培训/output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'population_result.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print("=" * 60)
print("  OUTPUT")
print("=" * 60)
print(f"  Figure saved to:  {output_path}")
print()
print("  All models executed successfully.")
