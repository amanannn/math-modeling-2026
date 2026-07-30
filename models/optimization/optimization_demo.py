"""
用途: 线性规划与整数规划模型演示 (LP, 0-1 IP, MIP)
输入: 内置案例数据
输出: 控制台结果 + output/optimization_result.png
调用: python models/optimization/optimization_demo.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 案例1: 生产计划 LP
print("=" * 50)
print("案例1: 生产计划线性规划 (LP)")
print("=" * 50)

c = [-40, -60]
A = [[1, 2], [2, 1]]
b = [8, 10]
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
print(f"  最优解: A={res.x[0]:.2f}件, B={res.x[1]:.2f}件")
print(f"  最大利润: {-res.fun:.2f}元")
print(f"  机器利用率: {(1*res.x[0]+2*res.x[1])/8*100:.0f}%")
print(f"  人工利用率: {(2*res.x[0]+1*res.x[1])/10*100:.0f}%")

# 案例2: 0-1 整数规划 (枚举法)
print("\n" + "=" * 50)
print("案例2: 投资项目选择 (0-1 IP, 枚举法)")
print("=" * 50)

invest = [200, 150, 180, 100]
profit = [80, 60, 70, 45]
budget = 500
best_val = 0
best_sel = []

for mask in range(1 << 4):
    sel = [(mask >> i) & 1 for i in range(4)]
    total_inv = sum(invest[i] * sel[i] for i in range(4))
    total_prf = sum(profit[i] * sel[i] for i in range(4))
    # 约束: 不超预算, 项目1和3互斥
    if (total_inv <= budget and not (sel[0] and sel[2])
        and total_prf > best_val):
        best_val = total_prf
        best_sel = sel

selected = [i+1 for i in range(4) if best_sel[i]]
print(f"  选择项目: {selected}")
print(f"  总投入: {sum(invest[i-1] for i in selected)}万")
print(f"  总收益: {best_val}万")
print(f"  资金利用率: {sum(invest[i-1] for i in selected)}/{budget} = {sum(invest[i-1] for i in selected)/budget*100:.0f}%")

# 案例3: 钢管下料
print("\n" + "=" * 50)
print("案例3: 钢管下料问题 (MIP)")
print("=" * 50)

raw_len = 7
patterns = []
for a in range(int(raw_len / 2.5) + 1):
    for b in range(int(raw_len / 1.6) + 1):
        if a * 2.5 + b * 1.6 <= raw_len and (a > 0 or b > 0):
            waste = raw_len - a * 2.5 - b * 1.6
            patterns.append((a, b, waste))

# scipy MILP 求解
c_mip = np.ones(len(patterns))
A_ub = np.zeros((2, len(patterns)))
A_ub[0] = [-p[0] for p in patterns]
A_ub[1] = [-p[1] for p in patterns]
b_ub = [-100, -150]
bounds_mip = [(0, 100) for _ in patterns]

res_mip = linprog(c_mip, A_ub=A_ub, b_ub=b_ub, bounds=bounds_mip, method='highs')
# 向上取整(启发式修复)
x_mip = np.ceil(res_mip.x)
total_pipes = int(np.sum(x_mip))

print(f"  最少用料: {total_pipes}根")
print("  截取方案:")
total_25 = total_16 = 0
for i, xi in enumerate(x_mip):
    if xi > 0.5:
        a, b, waste = patterns[i]
        print(f"    模式(2.5m x{a} + 1.6m x{b}): {int(xi)}根, 废料={waste:.1f}m")
        total_25 += a * int(xi)
        total_16 += b * int(xi)
print(f"  实际产出: 2.5m x {total_25}根, 1.6m x {total_16}根")

# 可视化: 三图合一
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# LP 可行域
ax = axes[0]
x1_vals = np.linspace(0, 8, 100)
c1 = (8 - x1_vals) / 2
c2 = 10 - 2 * x1_vals
upper = np.minimum(np.maximum(c1, 0), np.maximum(c2, 0))
ax.fill_between(x1_vals, 0, upper, where=(c1 >= 0) & (c2 >= 0),
                alpha=0.3, color='lightblue')
ax.plot(x1_vals, c1, 'r-', label='Machine: x1+2x2<=8')
ax.plot(x1_vals, c2, 'b-', label='Labor: 2x1+x2<=10')
ax.plot(res.x[0], res.x[1], 'r*', markersize=15, label=f'Opt({res.x[0]:.0f},{res.x[1]:.0f})')
ax.set_xlabel('Product A'); ax.set_ylabel('Product B')
ax.set_xlim(0, 9); ax.set_ylim(0, 9)
ax.set_title('Case 1: Production LP'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 0-1 IP 投资组合
ax = axes[1]
projects = ['P1\n200', 'P2\n150', 'P3\n180', 'P4\n100']
colors_ip = ['#2ecc71' if best_sel[i] else '#e74c3c' for i in range(4)]
ax.bar(projects, invest, color=colors_ip, edgecolor='black', alpha=0.8)
for i, (inv, prof) in enumerate(zip(invest, profit)):
    status = 'Sel' if best_sel[i] else 'Skip'
    ax.text(i, inv + 8, f'{status}\nProfit:{prof}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=budget, color='gray', linestyle='--', label=f'Budget:{budget}')
ax.set_ylabel('Investment'); ax.set_title('Case 2: 0-1 IP Project Selection')
ax.legend(fontsize=8); ax.set_ylim(0, 300)

# MIP 下料方案
ax = axes[2]
used = [(a, b, int(xi)) for i, (a, b, w) in enumerate(patterns) if x_mip[i] > 0.5]
labels_mip = [f'{a}x2.5m\n{b}x1.6m' for a, b, _ in used]
counts = [c for _, _, c in used]
colors_mip = plt.cm.Blues(np.linspace(0.4, 0.9, len(used)))
ax.pie(counts, labels=labels_mip, autopct='%d pipes',
       colors=colors_mip, startangle=90)
ax.set_title(f'Case 3: Cutting Stock ({total_pipes} pipes)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimization_result.png'),
            dpi=300, bbox_inches='tight')
print("\n[OK] output/optimization_result.png")
