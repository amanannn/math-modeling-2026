"""
用途: 智能优化算法综合演示脚本
      (Intelligent Optimization Algorithms Demo)
      涵盖四大类算法:
        [1] 手写遗传算法 (GA)            — Sphere 函数
        [2] 手写粒子群优化 (PSO)         — Sphere 函数
        [3] scipy.differential_evolution — Rastrigin 函数
        [4] scipy.dual_annealing         — Ackley 函数
      输出: 控制台打印各算法最优解/最优值/评价次数
            figure: intelligent_opt_result.png (四子图收敛曲线对比)
输入: 无 (所有参数在脚本内部定义，修改搜索范围/迭代次数等请在下方参数区调整)
输出: (1) 控制台 — 分节打印每个算法的运行结果与汇总表
      (2) 图片   — D:/虚拟C盘/数学建模培训/output/intelligent_opt_result.png
调用: python intelligent_opt_demo.py
依赖: numpy, matplotlib (Agg backend), scipy>=1.10
运行环境: Python 3.12+, Windows / Linux / macOS
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，无头环境也可运行
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing


# ============================================================
# 1. 测试函数定义 (Benchmark Functions)
# ============================================================

def sphere(x):
    """Sphere 函数 — 单峰、凸函数，最简单基准
       全局最小值: f(0,...,0) = 0
       搜索范围: x_i ∈ [-5.12, 5.12]
       Formula: f(x) = sum(x_i^2)
    """
    return np.sum(x ** 2)


def rastrigin(x):
    """Rastrigin 函数 — 多峰，大量局部极小
       全局最小值: f(0,...,0) = 0
       搜索范围: x_i ∈ [-5.12, 5.12]
       Formula: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    """
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    """Ackley 函数 — 多峰，外表面近平坦，中心一个狭长的洞
       全局最小值: f(0,...,0) ≈ 0
       搜索范围: x_i ∈ [-32, 32]
       Formula: f = -20*exp(-0.2*sqrt(sum(x^2)/n))
                -exp(sum(cos(2*pi*x))/n) + 20 + e
    """
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1.0)


# ============================================================
# 2. 遗传算法 (Genetic Algorithm, GA) — 手写实数编码实现
# ============================================================

def init_population(pop_size, n_vars, lb, ub):
    """初始化种群 (实数编码均匀随机采样)"""
    return np.random.uniform(lb, ub, size=(pop_size, n_vars))


def tournament_selection(pop, fitness, k=3):
    """锦标赛选择 (最小化) — 随机挑 k 个取最优"""
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()


def sbx_crossover(p1, p2, eta=15):
    """模拟二进制交叉 (Simulated Binary Crossover, SBX)
       eta: 分布指数，越大子代越接近父代
    """
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    for i in range(n):
        if np.random.rand() < 0.5:
            if abs(p1[i] - p2[i]) < 1e-10:
                continue
            u = np.random.rand()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
            c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
    return c1, c2


def polynomial_mutation(child, lb, ub, eta_m=20, rate=0.1):
    """多项式变异 (Polynomial Mutation)"""
    for i in range(len(child)):
        if np.random.rand() < rate:
            r = np.random.rand()
            delta = min(child[i] - lb, ub - child[i]) / (ub - lb)
            if r < 0.5:
                delta_q = (2.0 * r + (1.0 - 2.0 * r)
                           * (1.0 - delta) ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - r) + 2.0 * (r - 0.5)
                                 * (1.0 - delta) ** (eta_m + 1.0)) ** (1.0 / (eta_m + 1.0))
            child[i] += delta_q * (ub - lb)
            child[i] = np.clip(child[i], lb, ub)
    return child


def ga_minimize(func, n_vars, lb, ub, pop_size=100, max_gen=200,
                crossover_rate=0.85, mutation_rate=0.08, elite_ratio=0.05):
    """
    遗传算法主函数 (最小化)

    参数
    ----
    func : callable  — 目标函数，接受 1-D array 返回标量
    n_vars : int     — 变量维数
    lb, ub : float   — 变量下界 / 上界
    pop_size : int   — 种群大小
    max_gen : int    — 最大迭代代数
    crossover_rate, mutation_rate : float — 交叉率 / 变异率
    elite_ratio : float — 精英保留比例

    返回
    ----
    best_x : 1-D array  — 找到的最优解
    best_f : float       — 对应的最优值
    history : list       — 每代最优适应度值 (长度 = max_gen)
    """
    pop = init_population(pop_size, n_vars, lb, ub)
    n_elite = max(1, int(pop_size * elite_ratio))
    history = []

    for gen in range(max_gen):
        fit = np.array([func(ind) for ind in pop])
        # 精英：最小化的前 n_elite 个个体
        elite_idx = np.argsort(fit)[:n_elite]

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fit)
            p2 = tournament_selection(pop, fit)
            if np.random.rand() < crossover_rate:
                c1, c2 = sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = polynomial_mutation(c1, lb, ub, rate=mutation_rate)
            c2 = polynomial_mutation(c2, lb, ub, rate=mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        new_pop = np.array(new_pop)
        new_fit = np.array([func(ind) for ind in new_pop])
        # 精英替换新种群中最差的 n_elite 个
        worst_idx = np.argsort(new_fit)[-n_elite:]
        for i, ei in enumerate(elite_idx):
            new_pop[worst_idx[i]] = pop[ei].copy()

        pop = new_pop
        history.append(fit.min())

    final_fit = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(final_fit)
    return pop[best_idx], final_fit[best_idx], history


# ============================================================
# 3. 粒子群优化 (Particle Swarm Optimization, PSO) — 手写实现
# ============================================================

def pso_minimize(func, n_vars, lb, ub, n_particles=50, max_iter=200,
                 w_start=0.9, w_end=0.4, c1=1.5, c2=1.5):
    """
    粒子群优化主函数 (最小化)，带线性递减惯性权重

    参数
    ----
    func : callable      — 目标函数
    n_vars : int         — 维数
    lb, ub : float       — 下界 / 上界
    n_particles : int    — 粒子数
    max_iter : int       — 最大迭代次数
    w_start, w_end : float — 惯性权重起止值 (线性递减)
    c1, c2 : float       — 认知 / 社会系数

    返回
    ----
    gbest_pos : 1-D array — 全局最优位置
    gbest_val : float     — 全局最优值
    history : list        — 每代全局最优值 (长度 = max_iter)
    """
    dim = n_vars
    span = ub - lb

    pos = np.random.uniform(lb, ub, (n_particles, dim))
    vel = np.random.uniform(-0.1 * span, 0.1 * span, (n_particles, dim))

    pbest_pos = pos.copy()
    pbest_val = np.array([func(p) for p in pos])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = []

    for t in range(max_iter):
        w = w_start - (w_start - w_end) * t / max_iter

        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pbest_pos[i] - pos[i])
                      + c2 * r2 * (gbest_pos - pos[i]))
            vel[i] = np.clip(vel[i], -0.2 * span, 0.2 * span)
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], lb, ub)

            val = func(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = pos[i].copy()

        history.append(gbest_val)

    return gbest_pos, gbest_val, history


# ============================================================
# 4. 主程序：运行所有算法 + 输出结果 + 绘制收敛曲线
# ============================================================

def main():
    print("=" * 72)
    print("    Intelligent Optimization Algorithms Demo")
    print("    (GA  |  PSO  |  DE / scipy  |  SA / scipy)")
    print("=" * 72)

    # ---------- 全局参数 ----------
    DIM = 2            # 测试函数维度
    MAX_GEN = 150      # 最大迭代 / 进化代数
    POP_SIZE = 80      # 种群 / 粒子数
    np.random.seed(42)  # 固定随机种子，保证可复现

    histories = {}
    results = {}

    # ================================================================
    # [1] 遗传算法 GA — Sphere 函数 (最小化)
    # ================================================================
    print()
    print("─" * 72)
    print("  [1] Genetic Algorithm (hand-written) on Sphere(x) = sum(x_i^2)")
    print("─" * 72)

    LB, UB = -5.12, 5.12
    best_x_ga, best_f_ga, hist_ga = ga_minimize(
        sphere, DIM, LB, UB,
        pop_size=POP_SIZE, max_gen=MAX_GEN,
    )
    print(f"      Best x      : {best_x_ga}")
    print(f"      Best f(x)   : {best_f_ga:.8e}")
    print(f"      Theoretical : f(0, 0) = 0")

    histories['GA (Sphere)'] = hist_ga
    results['GA (Sphere)'] = (best_x_ga, best_f_ga)

    # ================================================================
    # [2] 粒子群优化 PSO — Sphere 函数 (最小化)
    # ================================================================
    print()
    print("─" * 72)
    print("  [2] Particle Swarm Optimization (hand-written) on Sphere(x)")
    print("─" * 72)

    best_x_pso, best_f_pso, hist_pso = pso_minimize(
        sphere, DIM, LB, UB,
        n_particles=POP_SIZE, max_iter=MAX_GEN,
    )
    print(f"      Best x      : {best_x_pso}")
    print(f"      Best f(x)   : {best_f_pso:.8e}")
    print(f"      Theoretical : f(0, 0) = 0")

    histories['PSO (Sphere)'] = hist_pso
    results['PSO (Sphere)'] = (best_x_pso, best_f_pso)

    # ================================================================
    # [3] 差分进化 DE / scipy — Rastrigin 函数 (最小化)
    # ================================================================
    print()
    print("─" * 72)
    print("  [3] scipy.optimize.differential_evolution on Rastrigin(x)")
    print("─" * 72)

    bounds_rastrigin = [(-5.12, 5.12)] * DIM
    de_history = []

    def de_callback(xk, convergence=None):
        de_history.append(rastrigin(xk))
        return False

    result_de = differential_evolution(
        rastrigin,
        bounds_rastrigin,
        strategy='best1bin',
        maxiter=MAX_GEN,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-12,
        seed=42,
        callback=de_callback,
    )
    print(f"      Best x      : {result_de.x}")
    print(f"      Best f(x)   : {result_de.fun:.8e}")
    print(f"      nfev        : {result_de.nfev}")
    print(f"      Theoretical : f(0, 0) = 0")

    # 若回调未满 max_gen 次（提前收敛），补齐以便绘图
    while len(de_history) < MAX_GEN:
        de_history.append(result_de.fun)
    histories['DE (Rastrigin)'] = de_history[:MAX_GEN]
    results['DE (Rastrigin)'] = (result_de.x, result_de.fun)

    # ================================================================
    # [4] 模拟退火 SA / scipy — Ackley 函数 (最小化)
    # ================================================================
    print()
    print("─" * 72)
    print("  [4] scipy.optimize.dual_annealing on Ackley(x)")
    print("─" * 72)

    bounds_ackley = [(-32.0, 32.0)] * DIM
    sa_history = []

    def sa_callback(x, f, context):
        sa_history.append(f)
        return False

    result_sa = dual_annealing(
        ackley,
        bounds_ackley,
        maxiter=MAX_GEN,
        initial_temp=5230.0,
        restart_temp_ratio=2e-5,
        seed=42,
        callback=sa_callback,
    )
    print(f"      Best x      : {result_sa.x}")
    print(f"      Best f(x)   : {result_sa.fun:.8e}")
    print(f"      nfev        : {result_sa.nfev}")
    print(f"      Theoretical : f(0, 0) ≈ 0")

    # SA 的回调次数不确定，补齐以便绘图
    while len(sa_history) < MAX_GEN:
        sa_history.append(result_sa.fun)
    histories['SA (Ackley)'] = sa_history[:MAX_GEN]
    results['SA (Ackley)'] = (result_sa.x, result_sa.fun)

    # ================================================================
    # 5. 绘制四子图收敛曲线
    # ================================================================
    print()
    print("=" * 72)
    print("    Generating convergence curves ...")
    print("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Curves of Intelligent Optimization Algorithms',
                 fontsize=15, fontweight='bold', y=0.98)

    # common style
    gen_range = range(1, MAX_GEN + 1)

    # ---- (a) GA ----
    ax = axes[0, 0]
    ax.plot(gen_range, histories['GA (Sphere)'], 'tab:blue', lw=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title('GA — Sphere Function')
    ax.grid(True, alpha=0.3)
    _, fv = results['GA (Sphere)']
    ax.text(0.97, 0.93, f'$f^* = {fv:.3e}$',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    # ---- (b) PSO ----
    ax = axes[0, 1]
    ax.semilogy(gen_range, histories['PSO (Sphere)'], 'tab:red', lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Value (log)')
    ax.set_title('PSO — Sphere Function')
    ax.grid(True, alpha=0.3)
    _, fv = results['PSO (Sphere)']
    ax.text(0.97, 0.93, f'$f^* = {fv:.3e}$',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    # ---- (c) DE ----
    ax = axes[1, 0]
    ax.plot(gen_range, histories['DE (Rastrigin)'], 'tab:green', lw=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Value')
    ax.set_title('DE (scipy) — Rastrigin Function')
    ax.grid(True, alpha=0.3)
    _, fv = results['DE (Rastrigin)']
    ax.text(0.97, 0.93, f'$f^* = {fv:.3e}$',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    # ---- (d) SA ----
    ax = axes[1, 1]
    ax.plot(gen_range, histories['SA (Ackley)'], 'tab:purple', lw=1.5)
    ax.set_xlabel('Iteration (outer loop)')
    ax.set_ylabel('Best Value')
    ax.set_title('SA (scipy) — Ackley Function')
    ax.grid(True, alpha=0.3)
    _, fv = results['SA (Ackley)']
    ax.text(0.97, 0.93, f'$f^* = {fv:.3e}$',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = r'D:\虚拟C盘\数学建模培训\output\intelligent_opt_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Figure saved to: {output_path}")
    plt.close()

    # ================================================================
    # 6. 结果汇总表
    # ================================================================
    print()
    print("=" * 72)
    print("    Result Summary")
    print("=" * 72)
    print(f"    {'Algorithm':<28} {'Best f(x)':<20} {'Status':<12}")
    print("    " + "-" * 60)
    for name, (x, f) in results.items():
        status = 'Converged' if abs(f) < 0.1 else 'Approximate'
        print(f"    {name:<28} {f:<20.6e} {status:<12}")
    print()
    print("    Test functions (all 2-D, minimized):")
    print("      Sphere    : f(x) = sum(x_i^2)               [GA, PSO]")
    print("      Rastrigin : f(x) = 10n + sum(x_i^2 - 10cos(2pi x_i))  [DE]")
    print("      Ackley    : f(x) = -20exp(...) - exp(...) + 20 + e     [SA]")
    print("=" * 72)
    print("    Demo completed successfully.")


if __name__ == '__main__':
    main()
