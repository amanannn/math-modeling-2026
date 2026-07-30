"""
用途: 计算电磁学与数学建模 Demo 脚本（阵列因子 / Friis 传输 / 趋肤深度）
输入: 无（参数均在脚本内定义）
输出: 控制台打印数值结果；output/electromagnetics_result.png (300dpi)
调用: python electromagnetics_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import pi, c, mu_0

# 全局绘图参数：英文标签，避免中文字体问题
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (12, 10),
})


# (1) 均匀直线阵 — 阵列因子方向图
def uniform_linear_array_factor(N, d, theta):
    """
    计算均匀直线阵的阵列因子
    输入: N（阵元数）, d（间距/波长）, theta（角度弧度）
    输出: AF（归一化线性值）, AF_dB（dB值）
    """
    psi = 2 * pi * d * np.sin(theta)
    # 阵列因子: AF = sin(N·ψ/2) / sin(ψ/2)，加小量避免除零
    AF = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2) + 1e-12))
    AF_dB = 20 * np.log10(AF + 1e-12)
    return AF, AF_dB


def demo_antenna_array():
    """演示均匀直线阵方向图"""
    N = 8                     # 阵元数
    d = 0.5                   # 间距（波长归一化）
    theta = np.linspace(-pi / 2, pi / 2, 2000)

    AF, AF_dB = uniform_linear_array_factor(N, d, theta)

    # 控制台输出
    print("=" * 60)
    print("  [1] 均匀直线阵方向图 (Array Factor)")
    print("=" * 60)
    print(f"  阵元数 N      = {N}")
    print(f"  阵元间距 d/λ  = {d}")
    print(f"  主瓣宽度 (3dB) ≈ {2 * np.rad2deg(np.arcsin(0.886 / (N * d))):.2f} deg")
    idx_max = np.argmax(AF)
    print(f"  最大指向       = {np.rad2deg(theta[idx_max]):.1f} deg")
    null_idx = np.where(AF < 0.01)[0]
    if len(null_idx) > 0:
        null_theta = theta[null_idx[0]]
        print(f"  第一零点       ≈ {np.rad2deg(null_theta):.2f} deg")
    print(f"  方向图动态范围 = {np.max(AF_dB) - np.min(AF_dB):.1f} dB")

    # 极坐标方向图
    ax1 = plt.subplot(221, projection="polar")
    ax1.plot(theta, AF, "b-", linewidth=1.5, label=f"N={N}, d/λ={d}")
    ax1.set_title("Uniform Linear Array — Polar", va="bottom", pad=12)
    ax1.legend(loc="upper right", fontsize=9)

    # 直角坐标方向图（dB）
    ax2 = plt.subplot(222)
    ax2.plot(np.rad2deg(theta), AF_dB, "b-", linewidth=1.5)
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Normalized AF (dB)")
    ax2.set_title("Uniform Linear Array — Rectangular")
    ax2.set_ylim(-40, 3)
    ax2.axhline(-3, color="gray", linestyle="--", linewidth=0.8, label="-3 dB")
    ax2.legend(fontsize=9)

    return theta, AF, AF_dB


# (2) Friis 传输方程 — 路径损耗 vs 距离
def path_loss_db(d, f):
    """
    自由空间路径损耗（Friis 传输方程）
    输入: d（距离 m）, f（频率 Hz）
    输出: L（路径损耗 dB）
    """
    return 20 * np.log10(d) + 20 * np.log10(f) - 147.55


def demo_friis():
    """演示 Friis 路径损耗"""
    f = 2.4e9                 # 2.4 GHz（Wi-Fi）
    d = np.logspace(0, 3, 200)  # 1 m ~ 1000 m

    L = path_loss_db(d, f)

    # 控制台输出
    print("\n" + "=" * 60)
    print("  [2] Friis 传输方程 — 自由空间路径损耗")
    print("=" * 60)
    print(f"  频率 f = {f / 1e9:.1f} GHz")
    print(f"  波长 λ = {c / f:.3f} m")
    print(f"  距离 10 m 时的路径损耗 : {path_loss_db(10, f):.2f} dB")
    print(f"  距离 100 m 时的路径损耗: {path_loss_db(100, f):.2f} dB")
    print(f"  距离 1000 m 时的路径损耗: {path_loss_db(1000, f):.2f} dB")

    # 绘图
    ax = plt.subplot(223)
    ax.semilogx(d, L, "r-", linewidth=1.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Path Loss (dB)")
    ax.set_title(f"Friis Free-Space Path Loss @ {f/1e9:.1f} GHz")
    ax.grid(True, which="both", alpha=0.3)
    # 标注参考点
    for dist in [10, 100, 1000]:
        loss = path_loss_db(dist, f)
        ax.plot(dist, loss, "ko", markersize=4)
        ax.annotate(f"{loss:.1f} dB", (dist, loss),
                     textcoords="offset points", xytext=(5, -12), fontsize=8)

    return d, L


# (3) 趋肤深度 vs 频率
def skin_depth(f, sigma, mu_r=1.0):
    """
    趋肤深度
    输入: f（频率 Hz）, sigma（电导率 S/m）, mu_r（相对磁导率）
    输出: delta（趋肤深度 m）
    """
    return np.sqrt(1 / (pi * f * mu_0 * mu_r * sigma))


def demo_skin_effect():
    """演示趋肤深度 vs 频率"""
    # 三种常见导体
    materials = {
        "Copper (Cu)":   {"sigma": 5.96e7,  "mu_r": 1.0},
        "Aluminum (Al)": {"sigma": 3.50e7,  "mu_r": 1.0},
        "Iron (Fe)":     {"sigma": 1.00e7,  "mu_r": 200.0},
    }
    f = np.logspace(1, 9, 500)  # 10 Hz ~ 1 GHz

    print("\n" + "=" * 60)
    print("  [3] 趋肤深度 (Skin Depth)")
    print("=" * 60)

    # 绘图
    ax = plt.subplot(224)
    for name, params in materials.items():
        delta = skin_depth(f, params["sigma"], params["mu_r"])
        ax.loglog(f, delta * 1e3, linewidth=1.5, label=name)  # 单位 mm
        f_vals = [50, 1e3, 1e6]
        for fv in f_vals:
            d_val = skin_depth(fv, params["sigma"], params["mu_r"])
            unit = "mm" if d_val < 0.1 else "m"
            factor = 1e3 if d_val < 0.1 else 1.0
            print(f"  {name:18s} @ {fv:9.0f} Hz:  δ = {d_val * factor:.4f} {unit}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Skin Depth (mm)")
    ax.set_title("Skin Depth vs Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # 标注工频（50 Hz）和 Wi-Fi（2.4 GHz）区域
    ax.axvline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(2.4e9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    return f, materials


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("  计算电磁学与数学建模 — Demo 演示")
    print("=" * 60)

    # 三个子演示（每个在 fig 上添加子图）
    demo_antenna_array()
    demo_friis()
    demo_skin_effect()

    plt.tight_layout(pad=3.0)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "electromagnetics_result.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("\n" + "-" * 60)
    print(f"  图片已保存: {output_path}")
    print("-" * 60)

    plt.close()

    print("\n  Demo 运行完毕。")
    print("=" * 60)
