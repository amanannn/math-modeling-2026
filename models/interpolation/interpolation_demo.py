"""
用途: 插值与拟合方法综合演示 (多项式拟合、样条插值、非线性拟合、应力-应变实例)
输入: 无（内置合成数据）
输出: 控制台结果 + output/interpolation_result.png
调用: python interpolation_demo.py
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 绘图样式设置
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "interpolation_result.png")
np.random.seed(42)


# 1. 多项式拟合与过拟合
def section_1_polynomial_overfitting():
    """Runge 函数 f(x)=1/(1+25x^2) 的多项式拟合，演示低阶合理/高阶过拟合"""
    print("=" * 65)
    print("SECTION 1: Polynomial Fitting & Overfitting (Runge Phenomenon)")
    print("=" * 65)

    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    x_fine = np.linspace(-1, 1, 500)
    y_true = runge(x_fine)

    # 低阶拟合 (5点, 4次)
    x_pts_low = np.linspace(-1, 1, 5)
    y_pts_low = runge(x_pts_low)
    coeff_low = np.polyfit(x_pts_low, y_pts_low, deg=4)
    y_fit_low = np.polyval(coeff_low, x_fine)
    err_low = np.max(np.abs(y_fit_low - y_true))
    print(f"  Degree-4 fit (5 points):  max error = {err_low:.4f}")

    # 中阶拟合 (10点, 9次)
    x_pts_mid = np.linspace(-1, 1, 10)
    y_pts_mid = runge(x_pts_mid)
    coeff_mid = np.polyfit(x_pts_mid, y_pts_mid, deg=9)
    y_fit_mid = np.polyval(coeff_mid, x_fine)
    err_mid = np.max(np.abs(y_fit_mid - y_true))
    print(f"  Degree-9 fit (10 points): max error = {err_mid:.4f}")

    # 高阶过拟合 (16点, 15次)
    x_pts_high = np.linspace(-1, 1, 16)
    y_pts_high = runge(x_pts_high)
    coeff_high = np.polyfit(x_pts_high, y_pts_high, deg=15)
    y_fit_high = np.polyval(coeff_high, x_fine)
    err_high = np.max(np.abs(y_fit_high - y_true))
    print(f"  Degree-15 fit (16 points): max error = {err_high:.4f}  <-- severe oscillation at edges")
    print()

    return x_fine, y_true, (x_pts_low, y_pts_low, y_fit_low,
                            x_pts_mid, y_pts_mid, y_fit_mid,
                            x_pts_high, y_pts_high, y_fit_high)


# 2. 三次样条插值
def section_2_cubic_spline():
    """对温度-时间数据进行三次样条插值，比较 natural 与 not-a-knot 边界条件"""
    print("=" * 65)
    print("SECTION 2: Cubic Spline Interpolation")
    print("=" * 65)

    # 模拟每2小时的温度数据
    hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    temp = np.array([15.2, 14.8, 14.5, 16.1, 18.5, 21.3, 23.8,
                     24.5, 23.9, 21.2, 18.6, 16.5, 15.0])

    # 创建不同边界条件的样条
    cs_natural = CubicSpline(hours, temp, bc_type="natural")
    cs_notaknot = CubicSpline(hours, temp, bc_type="not-a-knot")

    # 细网格用于绘图
    hours_fine = np.linspace(0, 24, 241)
    temp_natural = cs_natural(hours_fine)
    temp_notaknot = cs_notaknot(hours_fine)

    # 一阶导数（温度变化率）
    temp_rate = cs_natural(hours_fine, nu=1)

    # 寻找最高温度时刻
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t: -cs_natural(t), bounds=(0, 24), method="bounded")
    print(f"  Max temperature:  T = {-res.fun:.2f} C  at  t = {res.x:.2f} h")
    print(f"  Data points: {len(hours)},  interpolation points: {len(hours_fine)}")
    print()

    return (hours, temp, hours_fine, temp_natural, temp_notaknot, temp_rate, cs_natural)


# 3. 非线性曲线拟合
def section_3_nonlinear_curve_fit():
    """药物浓度衰减数据的非线性最小二乘拟合（指数模型）"""
    print("=" * 65)
    print("SECTION 3: Nonlinear Curve Fitting (curve_fit)")
    print("=" * 65)

    # 模拟药物浓度数据
    t = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12])
    conc = np.array([100, 62, 40, 26, 18, 13, 10, 6.5, 4.5, 3.5])

    # 模型1: 指数衰减 C(t) = C0 * exp(-k * t)
    def exp_decay(t, C0, k):
        return C0 * np.exp(-k * t)

    popt1, pcov1 = curve_fit(exp_decay, t, conc, p0=[100, 0.2])
    C0_opt, k_opt = popt1
    C0_err, k_err = np.sqrt(np.diag(pcov1))
    half_life = np.log(2) / k_opt

    y_pred1 = exp_decay(t, *popt1)
    ss_res1 = np.sum((conc - y_pred1)**2)
    ss_tot1 = np.sum((conc - np.mean(conc))**2)
    r2_1 = 1 - ss_res1 / ss_tot1
    rmse_1 = np.sqrt(ss_res1 / len(t))

    print(f"  Model: C(t) = C0 * exp(-k * t)")
    print(f"    C0 = {C0_opt:.2f} +/- {C0_err:.2f}")
    print(f"    k  = {k_opt:.4f} +/- {k_err:.4f}")
    print(f"    Half-life = {half_life:.2f} h")
    print(f"    R^2 = {r2_1:.4f},  RMSE = {rmse_1:.4f}")

    # 模型2: 带偏移的指数衰减 C(t) = C0 * exp(-k * t) + b
    def exp_decay_offset(t, C0, k, b):
        return C0 * np.exp(-k * t) + b

    popt2, pcov2 = curve_fit(exp_decay_offset, t, conc, p0=[100, 0.2, 0])
    y_pred2 = exp_decay_offset(t, *popt2)
    ss_res2 = np.sum((conc - y_pred2)**2)
    r2_2 = 1 - ss_res2 / ss_tot1
    rmse_2 = np.sqrt(ss_res2 / len(t))
    print(f"  Model: C(t) = C0 * exp(-k * t) + b")
    print(f"    C0 = {popt2[0]:.2f},  k = {popt2[1]:.4f},  b = {popt2[2]:.4f}")
    print(f"    R^2 = {r2_2:.4f},  RMSE = {rmse_2:.4f}")

    # 细网格预测值用于绘图
    t_fine = np.linspace(0, 12, 100)
    c_fine1 = exp_decay(t_fine, *popt1)
    c_fine2 = exp_decay_offset(t_fine, *popt2)

    print()
    return t, conc, t_fine, c_fine1, c_fine2


# 4. 实际案例：材料应力-应变曲线拟合
def section_4_real_case():
    """对比二次、幂律、指数偏移三种模型拟合实验应力-应变数据"""
    print("=" * 65)
    print("SECTION 4: Real-Case Example - Stress-Strain Curve Fitting")
    print("=" * 65)

    # 实验数据：应力 (MPa) 与应变 (%)
    stress = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    strain = np.array([0.00, 0.03, 0.08, 0.15, 0.24, 0.36,
                       0.50, 0.68, 0.88, 1.12, 1.40])

    # 模型1: 二次多项式 y = a*x^2 + b*x + c
    coeff_quad = np.polyfit(stress, strain, deg=2)
    strain_quad = np.polyval(coeff_quad, stress)
    ss_res_q = np.sum((strain - strain_quad)**2)
    ss_tot = np.sum((strain - np.mean(strain))**2)
    r2_q = 1 - ss_res_q / ss_tot
    rmse_q = np.sqrt(ss_res_q / len(stress))
    print(f"  Model 1 - Quadratic:  y = {coeff_quad[0]:.6f} x^2 + {coeff_quad[1]:.6f} x + {coeff_quad[2]:.6f}")
    print(f"    R^2 = {r2_q:.4f},  RMSE = {rmse_q:.4f}")

    # 模型2: 幂律 y = a * x^b
    def power_law(x, a, b):
        return a * x**b

    popt_pl, _ = curve_fit(power_law, stress[1:], strain[1:], p0=[0.001, 1.5])
    # 排除应力为0的点以避免对数问题
    strain_pl = np.zeros_like(stress)
    strain_pl[1:] = power_law(stress[1:], *popt_pl)
    ss_res_pl = np.sum((strain - strain_pl)**2)
    r2_pl = 1 - ss_res_pl / ss_tot
    rmse_pl = np.sqrt(ss_res_pl / len(stress))
    print(f"  Model 2 - Power law:  y = {popt_pl[0]:.6f} * x^{popt_pl[1]:.4f}")
    print(f"    R^2 = {r2_pl:.4f},  RMSE = {rmse_pl:.4f}")

    # 模型3: 指数偏移 y = a * (exp(b*x) - 1)
    def exp_off(x, a, b):
        return a * (np.exp(b * x) - 1)

    popt_eo, _ = curve_fit(exp_off, stress, strain, p0=[0.1, 0.02])
    strain_eo = exp_off(stress, *popt_eo)
    ss_res_eo = np.sum((strain - strain_eo)**2)
    r2_eo = 1 - ss_res_eo / ss_tot
    rmse_eo = np.sqrt(ss_res_eo / len(stress))
    print(f"  Model 3 - Exponential off.:  y = {popt_eo[0]:.4f} * (exp({popt_eo[1]:.4f} * x) - 1)")
    print(f"    R^2 = {r2_eo:.4f},  RMSE = {rmse_eo:.4f}")

    # 细网格用于绘制平滑曲线
    stress_fine = np.linspace(0, 100, 200)
    strain_fine_q = np.polyval(coeff_quad, stress_fine)
    strain_fine_pl = np.zeros_like(stress_fine)
    strain_fine_pl[1:] = power_law(stress_fine[1:], *popt_pl)
    strain_fine_eo = exp_off(stress_fine, *popt_eo)

    # 选出最佳模型
    models_r2 = {"Quadratic": r2_q, "Power law": r2_pl, "Exp. offset": r2_eo}
    best = max(models_r2, key=models_r2.get)
    print(f"\n  >>> Best model (by R^2): {best}  (R^2 = {models_r2[best]:.4f})")
    print()

    return (stress, strain, stress_fine,
            strain_fine_q, strain_fine_pl, strain_fine_eo,
            coeff_quad, popt_pl, popt_eo,
            strain_quad, strain_pl, strain_eo)


# 主程序
def main():
    print("=" * 65)
    print("  INTERPOLATION & FITTING DEMO")
    print("  Mathematical Modeling - National Competition Training")
    print("=" * 65)
    print()

    res1 = section_1_polynomial_overfitting()
    res2 = section_2_cubic_spline()
    res3 = section_3_nonlinear_curve_fit()
    res4 = section_4_real_case()

    # 解包结果
    (x_fine, y_true,
     (x_low, y_low, y_fit_low,
      x_mid, y_mid, y_fit_mid,
      x_high, y_high, y_fit_high)) = res1

    (hours, temp, hours_fine, temp_natural, temp_notaknot, temp_rate, cs_natural) = res2
    (t_drug, conc_drug, t_fine_drug, c_fine1, c_fine2) = res3
    (stress, strain, stress_fine,
     strain_q, strain_pl, strain_eo,
     coeff_q, popt_pl, popt_eo,
     strain_quad_pts, strain_pl_pts, strain_eo_pts) = res4

    # 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: 多项式过拟合
    ax = axes[0, 0]
    ax.plot(x_fine, y_true, "k-", linewidth=2, label="True f(x)")
    ax.plot(x_fine, y_fit_low, "b--", linewidth=1.5, label="Degree 4 (5 pts)")
    ax.plot(x_fine, y_fit_mid, "g-.", linewidth=1.5, label="Degree 9 (10 pts)")
    ax.plot(x_fine, y_fit_high, "r:", linewidth=1.5, label="Degree 15 (16 pts)")
    ax.scatter(x_low, y_low, c="b", s=30, zorder=5, marker="o")
    ax.scatter(x_high, y_high, c="r", s=30, zorder=5, marker="s", alpha=0.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Polynomial Fitting — Overfitting Demo\n(Runge: $f(x)=1/(1+25x^2)$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: 三次样条插值
    ax = axes[0, 1]
    ax.plot(hours, temp, "ro", markersize=6, label="Measured data")
    ax.plot(hours_fine, temp_natural, "b-", linewidth=2, label="Natural spline")
    ax.plot(hours_fine, temp_notaknot, "g--", linewidth=1.5, label="Not-a-knot spline")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Cubic Spline Interpolation\n(Temperature vs. Time)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 内嵌一阶导数
    ax_twin = ax.twinx()
    ax_twin.plot(hours_fine, temp_rate, "k:", linewidth=1, alpha=0.6)
    ax_twin.set_ylabel("dT/dt (C/h)", color="gray", fontsize=8)
    ax_twin.tick_params(colors="gray", labelsize=7)

    # Panel C: 非线性曲线拟合
    ax = axes[1, 0]
    ax.scatter(t_drug, conc_drug, c="red", s=40, label="Measured conc.")
    ax.plot(t_fine_drug, c_fine1, "b-", linewidth=2, label="Exp decay")
    ax.plot(t_fine_drug, c_fine2, "g--", linewidth=1.5, label="Exp decay + offset")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Nonlinear Curve Fit\n(Drug Concentration Decay)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D: 实际应力-应变拟合
    ax = axes[1, 1]

    # 预计算 R^2 用于标注
    ss_tot_d = np.sum((strain - np.mean(strain))**2)
    r2_q_d = 1 - np.sum((strain - strain_quad_pts)**2) / ss_tot_d
    r2_pl_d = 1 - np.sum((strain - strain_pl_pts)**2) / ss_tot_d
    r2_eo_d = 1 - np.sum((strain - strain_eo_pts)**2) / ss_tot_d

    ax.scatter(stress, strain, c="red", s=40, label="Experimental data")
    ax.plot(stress_fine, strain_q, "b-", linewidth=2,
            label=f"Quadratic  (R^2={r2_q_d:.3f})")
    ax.plot(stress_fine, strain_pl, "g-.", linewidth=1.5,
            label=f"Power law  (R^2={r2_pl_d:.3f})")
    ax.plot(stress_fine, strain_eo, "m:", linewidth=1.5,
            label=f"Exp. offset  (R^2={r2_eo_d:.3f})")
    ax.set_xlabel("Stress (MPa)")
    ax.set_ylabel("Strain (%)")
    ax.set_title("Real-Case: Stress-Strain Curve Fitting\n(Experimental Data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {OUTPUT_PATH}")
    print()

    print("=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
