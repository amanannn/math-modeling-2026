"""
用途: 代码模板集 — 数据读取 / 可视化 / 优化建模 / ODE求解
输入: 按需修改模板参数
输出: 对应结果
调用: 复制所需模板函数到工作脚本
"""
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Template 1: Data I/O
# ============================================================
def load_excel(filepath, sheet_name=0):
    """读取Excel/CSV数据"""
    import pandas as pd
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    return pd.read_excel(filepath, sheet_name=sheet_name)

def save_results(data, filepath):
    """保存结果为CSV"""
    import pandas as pd
    pd.DataFrame(data).to_csv(filepath, index=False, encoding='utf-8-sig')

# ============================================================
# Template 2: Publication-quality Figure
# ============================================================
def setup_figure(figsize=(8,5)):
    """创建论文级图表的基础设置"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax

def save_figure(fig, filepath, formats=('png','eps')):
    """保存为高清PNG+矢量EPS"""
    for fmt in formats:
        fig.savefig(filepath.replace('.png', f'.{fmt}'),
                    dpi=300, bbox_inches='tight')
    print(f"[OK] Saved to {filepath}")

# ============================================================
# Template 3: LP Optimization
# ============================================================
def solve_lp(c, A_ub, b_ub, bounds, maximize=True):
    """
    求解线性规划
    c: 目标系数
    A_ub, b_ub: 不等式约束 A_ub @ x <= b_ub
    bounds: 变量范围 [(min,max), ...]
    maximize: True=最大化, False=最小化
    """
    from scipy.optimize import linprog
    obj = [-ci for ci in c] if maximize else c
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.x, -res.fun if maximize else res.fun, res.success

# ============================================================
# Template 4: ODE Solver
# ============================================================
def solve_ode(ode_func, t_span, y0, params=(), method='RK45'):
    """
    求解ODE初值问题
    ode_func(t, y, *params): 右端函数 dy/dt = f(t,y)
    t_span: [t0, tf]
    y0: 初值
    params: 额外参数
    """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode_func, t_span, y0, args=params,
                    method=method, rtol=1e-6, atol=1e-9,
                    dense_output=True, max_step=0.1)
    return sol.t, sol.y, sol

# ============================================================
# Template 5: TOPSIS Evaluation
# ============================================================
def topsis_template(X, weights, directions):
    """
    TOPSIS综合评价
    X: m行n列 (m方案, n指标)
    weights: n个权重
    directions: n个方向 (1=极大型, -1=极小型)
    """
    m, n = X.shape
    X_pos = X.copy()
    for j in range(n):
        if directions[j] == -1:
            X_pos[:, j] = X[:, j].max() - X[:, j]
    Z = X_pos / np.sqrt((X_pos ** 2).sum(axis=0))
    Z_w = Z * weights
    D_plus = np.sqrt(((Z_w - Z_w.max(axis=0)) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z_w - Z_w.min(axis=0)) ** 2).sum(axis=1))
    C = D_minus / (D_plus + D_minus)
    return C, np.argsort(-C) + 1

# ============================================================
# Template 6: AHP Weight Calculation
# ============================================================
def ahp_template(matrix):
    """
    层次分析法权重计算
    matrix: n×n判断矩阵 (Saaty 1~9标度)
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    w = eigenvectors[:, max_idx].real
    w = w / w.sum()
    n = matrix.shape[0]
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1:0, 2:0, 3:0.52, 4:0.89, 5:1.12, 6:1.26,
               7:1.36, 8:1.41, 9:1.46}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    return w, CR, CR < 0.1

print("=" * 50)
print("Code Templates Loaded")
print("=" * 50)
print("Available: load_excel, save_results, setup_figure, save_figure,")
print("           solve_lp, solve_ode, topsis_template, ahp_template")
