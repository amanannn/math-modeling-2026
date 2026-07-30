"""
用途: 决策模型演示 — AHP层次分析法 + TOPSIS综合评价 + 不确定型决策
输入: 内置案例数据
输出: 控制台 + output/decision_result.png
调用: python models/decision/decision_demo.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 案例1: AHP层次分析法
print("=" * 50)
print("Case 1: AHP - Supplier Selection")
print("=" * 50)

def ahp(matrix):
    n = matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    w = eigenvectors[:, max_idx].real
    w = w / w.sum()
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1:0, 2:0, 3:0.52, 4:0.89, 5:1.12, 6:1.26,
               7:1.36, 8:1.41, 9:1.46, 10:1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    return w, lambda_max, CI, CR, CR < 0.1

# 准则层矩阵
A_criteria = np.array([
    [1,   3,   5],
    [1/3, 1,   3],
    [1/5, 1/3, 1]
])
w_c, lam, ci, cr, ok = ahp(A_criteria)
labels_c = ['Cost', 'Quality', 'Time']
print(f"  Criteria weights:")
for lb, wt in zip(labels_c, w_c):
    print(f"    {lb}: {wt:.4f}")
print(f"  lambda_max={lam:.3f}, CI={ci:.4f}, CR={cr:.4f}, {'PASS' if ok else 'FAIL'}")

# 案例2: TOPSIS综合评价
print("\n" + "=" * 50)
print("Case 2: TOPSIS - Alternative Ranking")
print("=" * 50)

def topsis(X, weights, directions):
    m, n = X.shape
    X_pos = X.copy()
    for j in range(n):
        if directions[j] == -1:  # 成本型转效益型
            X_pos[:, j] = X[:, j].max() - X[:, j]

    Z = X_pos / np.sqrt((X_pos ** 2).sum(axis=0))
    Z_w = Z * weights

    Z_plus = Z_w.max(axis=0)
    Z_minus = Z_w.min(axis=0)

    D_plus = np.sqrt(((Z_w - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z_w - Z_minus) ** 2).sum(axis=1))

    C = D_minus / (D_plus + D_minus)
    rank = np.argsort(-C) + 1
    return C, rank

# 4个供应商, 3项指标
X = np.array([
    [80, 90, 5],   # S1
    [65, 85, 7],   # S2
    [70, 95, 4],   # S3
    [90, 80, 6]    # S4
])
directions = [-1, 1, -1]
scores, ranks = topsis(X, w_c, directions)
print("  Supplier scores and ranks:")
for i in range(4):
    print(f"    S{i+1}: score={scores[i]:.4f}, rank={ranks[i]}")

# 案例3: 不确定型决策
print("\n" + "=" * 50)
print("Case 3: Decision Under Uncertainty (5 rules)")
print("=" * 50)

payoff = np.array([
    [800, 550, 300, -100],
    [600, 500, 350,    0],
    [400, 380, 320,  150]
])
names = ['A1', 'A2', 'A3']
optimistic = names[payoff.max(axis=1).argmax()]
pessimistic = names[payoff.min(axis=1).argmax()]
regret = payoff.max(axis=0) - payoff
savage = names[regret.max(axis=1).argmin()]
laplace = names[payoff.mean(axis=1).argmax()]
print(f"  Optimistic (max-max): {optimistic}")
print(f"  Pessimistic (max-min): {pessimistic}")
print(f"  Savage (min-max regret): {savage}")
print(f"  Laplace (equal prob): {laplace}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# AHP权重图
ax = axes[0]
bars = ax.bar(labels_c, w_c, color=['#e74c3c', '#3498db', '#2ecc71'], edgecolor='black')
for bar, v in zip(bars, w_c):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Weight'); ax.set_title('Case 1: AHP Criteria Weights')
ax.set_ylim(0, 0.8); ax.grid(axis='y', alpha=0.3)

# TOPSIS得分图
ax = axes[1]
colors_t = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 4))
bars = ax.bar([f'S{i+1}' for i in range(4)], scores, color=colors_t, edgecolor='black')
for bar, s, r in zip(bars, scores, ranks):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'#{r} ({s:.3f})', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('TOPSIS Score'); ax.set_title('Case 2: TOPSIS Ranking')
ax.set_ylim(0, 0.9); ax.grid(axis='y', alpha=0.3)

# 收益矩阵热力图
ax = axes[2]
im = ax.imshow(payoff, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(['V.Good','Good','Fair','Poor'])
ax.set_yticks(range(3)); ax.set_yticklabels(names)
for i in range(3):
    for j in range(4):
        ax.text(j, i, payoff[i, j], ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if payoff[i, j] < 200 else 'black')
ax.set_title('Case 3: Payoff Matrix')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'decision_result.png'),
            dpi=300, bbox_inches='tight')
print("\n[OK] output/decision_result.png")
