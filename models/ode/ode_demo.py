"""
用途: 微分方程数值求解演示 — SIR/LV/阻尼振动
输入: 内置模型参数
输出: output/ode_result.png
调用: python models/ode/ode_demo.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 案例1: SIR传染病模型
print("=" * 50)
print("案例1: SIR传染病模型")
print("=" * 50)

def sir(t, y, N, beta, gamma):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

N = 10000; beta = 0.3; gamma = 0.1
R0_val = beta / gamma
I_init = 10; R_init = 0; S0 = N - I_init - R_init

sol_sir = solve_ivp(sir, [0, 160], [S0, I_init, R_init], args=(N, beta, gamma), max_step=0.5)
peak_day = sol_sir.t[np.argmax(sol_sir.y[1])]
peak_infected = int(np.max(sol_sir.y[1]))
final_R = int(sol_sir.y[2, -1])
print(f"  基本再生数 R0 = {R0_val:.1f}")
print(f"  感染峰值: 第{peak_day:.1f}天, {peak_infected}人")
print(f"  最终康复率: {final_R/N*100:.1f}%")

# 案例2: Lotka-Volterra捕食者-猎物模型
print("\n" + "=" * 50)
print("案例2: Lotka-Volterra 捕食者-猎物模型")
print("=" * 50)

def lotka_volterra(t, z, a, b, d, g):
    x, y = z
    return [a*x - b*x*y, d*x*y - g*y]

a, b_v, d, g_v = 1.0, 0.1, 0.02, 0.5
sol_lv = solve_ivp(lotka_volterra, [0, 100], [40, 9],
                   args=(a, b_v, d, g_v), max_step=0.2)
print(f"  猎物均值: {np.mean(sol_lv.y[0]):.1f}, 捕食者均值: {np.mean(sol_lv.y[1]):.1f}")

# 案例3: 阻尼振动 (二阶ODE)
print("\n" + "=" * 50)
print("案例3: 阻尼振动 (二阶ODE)")
print("=" * 50)

def damped_osc(t, y):
    x, v = y
    return [v, -0.5*v - 5*x]

sol_osc = solve_ivp(damped_osc, [0, 20], [1.0, 0.0],
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=0.05)
print(f"  求解步数: {len(sol_osc.t)}, 求解器: RK45")

# 可视化: 三子图
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# SIR子图
ax = axes[0]
ax.plot(sol_sir.t, sol_sir.y[0], 'b-', label='S: Susceptible', linewidth=1.5)
ax.plot(sol_sir.t, sol_sir.y[1], 'r-', label='I: Infected', linewidth=1.5)
ax.plot(sol_sir.t, sol_sir.y[2], 'g-', label='R: Recovered', linewidth=1.5)
ax.axvline(x=peak_day, color='gray', linestyle='--', alpha=0.7)
ax.annotate(f'Peak: {peak_infected}\nDay {peak_day:.0f}',
            xy=(peak_day, peak_infected), xytext=(peak_day+20, peak_infected+500),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
ax.set_xlabel('Time (days)'); ax.set_ylabel('Population')
ax.set_title(f'Case 1: SIR Epidemic Model (R0={R0_val:.1f})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# LV子图
ax = axes[1]
ax.plot(sol_lv.t, sol_lv.y[0], 'g-', label='Prey', linewidth=1.5)
ax.plot(sol_lv.t, sol_lv.y[1], 'r-', label='Predator', linewidth=1.5)
ax.set_xlabel('Time'); ax.set_ylabel('Population')
ax.set_title('Case 2: Lotka-Volterra Predator-Prey')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 阻尼振动子图
ax = axes[2]
ax.plot(sol_osc.t, sol_osc.y[0], 'b-', label='x(t) displacement', linewidth=1.2)
ax.plot(sol_osc.t, sol_osc.y[1], 'r--', label='v(t) velocity', linewidth=1.2)
ax.set_xlabel('Time'); ax.set_ylabel('x / v')
ax.set_title('Case 3: Damped Oscillation (RK45)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ode_result.png'),
            dpi=300, bbox_inches='tight')
print("\n[OK] output/ode_result.png")
