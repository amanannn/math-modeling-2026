"""
用途: CT 图像重建演示 — Radon 变换 + 直接反投影 + FBP 滤波反投影
输入: 无（自动生成多圆叠加体模，Radon 正变换使用解析公式）
输出: 四面板对比图 → output/ct_reconstruction_result.png，控制台输出关键数值
调用: python ct_reconstruction_demo.py
"""

import os
import sys
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. 椭圆体模定义 — 使用解析椭圆参数

# 椭圆参数: (A, a, b, x0, y0, theta_deg)，A=强度，a/b=半轴，x0/y0=中心偏移，theta=旋转角度
SHEPP_LOGAN_PARAMS = [
    ( 1.0,  0.69, 0.92,  0.00,  0.00,   0.0),   # 外壳
    (-0.8,  0.66, 0.87,  0.00,  0.00,   0.0),   # 内部低密度
    (-0.2,  0.31, 0.22, -0.22,  0.00,  72.0),   # 左椭圆
    (-0.2,  0.41, 0.16,  0.22,  0.00, 108.0),   # 右椭圆
    ( 0.1,  0.25, 0.21,  0.00,  0.35,   0.0),   # 上小椭圆
    ( 0.1,  0.14, 0.23,  0.00,  0.10,   0.0),   # 中间小椭圆
    ( 0.1,  0.11, 0.22,  0.00, -0.10,   0.0),   # 下小椭圆
    ( 0.1,  0.16, 0.05, -0.08, -0.35,   0.0),   # 左下小椭球
    ( 0.1,  0.16, 0.05,  0.08, -0.35,   0.0),   # 右下小椭球
    ( 0.1,  0.10, 0.04,  0.00, -0.42,   0.0),   # 底部小椭球
]


def render_phantom(params, n_pixels=256):
    """将椭圆参数渲染为离散图像"""
    phantom = np.zeros((n_pixels, n_pixels))
    center = n_pixels / 2.0

    y, x = np.mgrid[0:n_pixels, 0:n_pixels]
    xc = (x - center) / center  # 归一化到 [-1, 1]
    yc = (y - center) / center

    for A, a, b, x0, y0, theta_deg in params:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xt = (xc - x0) * cos_t + (yc - y0) * sin_t
        yt = -(xc - x0) * sin_t + (yc - y0) * cos_t
        mask = (xt ** 2 / a ** 2 + yt ** 2 / b ** 2) <= 1.0
        phantom[mask] += A

    phantom = np.clip(phantom, 0.0, 1.0)
    return phantom


# 2. 解析 Radon 正变换

def radon_transform_analytic(params, angles_rad, n_detectors, half_range=1.4):
    """
    解析计算多椭圆体模的 Radon 变换
    每个椭圆的投影有闭合形式: p(s,θ) = 2*A*a*b / L² * sqrt(L² - (s - s0)²)
    """
    n_angles = len(angles_rad)
    sinogram = np.zeros((n_angles, n_detectors))

    s_pos = np.linspace(-half_range, half_range, n_detectors)

    for A, a, b, x0, y0, theta_deg in params:
        if A == 0:
            continue
        theta_e = np.deg2rad(theta_deg)
        cos_e, sin_e = np.cos(theta_e), np.sin(theta_e)

        for i, theta in enumerate(angles_rad):
            dtheta = theta - theta_e
            L_sq = (a * np.cos(dtheta)) ** 2 + (b * np.sin(dtheta)) ** 2
            L = np.sqrt(L_sq) if L_sq > 0 else 1e-16
            s0 = x0 * np.cos(theta) + y0 * np.sin(theta)
            dist_sq = (s_pos - s0) ** 2
            inside = dist_sq <= L_sq - 1e-12
            val = np.zeros(n_detectors)
            val[inside] = (2.0 * A * a * b / L_sq) * np.sqrt(
                np.maximum(0, L_sq - dist_sq[inside])
            )
            sinogram[i, :] += val

    return sinogram


# 3. 直接反投影

def back_project(sinogram, angles_rad, n_pixels, half_range=1.4):
    """直接反投影重建（不加滤波），将每个角度的投影沿对应方向"涂抹"回图像空间"""
    n_angles, n_detectors = sinogram.shape
    center = n_pixels / 2.0
    scale = 2.0 * half_range / (n_detectors - 1)

    x = np.arange(n_pixels) - center
    X, Y = np.meshgrid(x / center, x / center)

    recon = np.zeros((n_pixels, n_pixels))

    for i in range(n_angles):
        theta = angles_rad[i]
        s_proj = X * np.cos(theta) + Y * np.sin(theta)
        s_idx = (s_proj + half_range) / scale
        s_idx = np.clip(s_idx, 0, n_detectors - 1)
        s0 = np.floor(s_idx).astype(np.int64)
        s1 = np.minimum(s0 + 1, n_detectors - 1)
        frac = s_idx - s0
        interp = (1.0 - frac) * sinogram[i, s0] + frac * sinogram[i, s1]
        recon += interp

    recon *= np.pi / n_angles
    return recon


# 4. FBP — 滤波反投影

def fbp_reconstruct(sinogram, angles_rad, n_pixels, half_range=1.4):
    """
    滤波反投影 (FBP) 重建
    流程: FFT → 频域乘以 Ram-Lak 斜坡滤波器 |ω| → IFFT → 反投影
    """
    n_angles, n_detectors = sinogram.shape

    # 构建 Ram-Lak 斜坡滤波器
    freqs = fftfreq(n_detectors)
    ramp_filter = np.abs(freqs) * n_detectors  # H[k] ∝ |k|

    # 频域滤波
    filtered_sino = np.zeros_like(sinogram)
    for i in range(n_angles):
        proj_fft = fft(sinogram[i, :])
        filtered_sino[i, :] = np.real(ifft(proj_fft * ramp_filter))

    # 反投影
    recon = back_project(filtered_sino, angles_rad, n_pixels, half_range)
    recon *= 0.5  # 额外缩放因子
    return recon


# 5. 辅助: 计算重建质量指标

def normalize_for_comparison(recon, phantom, mask_thresh=0.05):
    """将重建结果缩放到与体模可比的范围（线性回归缩放）"""
    mask = phantom > mask_thresh
    if mask.sum() < 10:
        mask = phantom > 0
    A = np.column_stack([recon[mask], np.ones(mask.sum())])
    coeff, _, _, _ = np.linalg.lstsq(A, phantom[mask], rcond=None)
    recon_scaled = recon * coeff[0] + coeff[1]
    return recon_scaled, coeff[0], coeff[1]


# 6. 主函数

def main():
    print("=" * 60)
    print("  CT Image Reconstruction Demo")
    print("  Radon Transform + Back Projection + FBP")
    print("=" * 60)

    # 参数设置
    N = 256                     # 重建图像像素数
    n_angles = 180              # 投影角度数
    n_detectors = 367           # 探测器数 (≈ √2 * N)
    half_range = 1.5            # 探测器覆盖半宽
    params = SHEPP_LOGAN_PARAMS

    # 1. 渲染体模
    print("\n[Step 1/6] Rendering Shepp-Logan phantom ...", end=" ")
    phantom = render_phantom(params, N)
    print(f"done, shape={phantom.shape}, "
          f"range=[{phantom.min():.4f}, {phantom.max():.4f}]")

    # 2. 角度设置
    angles_rad = np.linspace(0, np.pi, n_angles, endpoint=False)
    angle_step_deg = np.rad2deg(angles_rad[1])
    print(f"\n[Step 2/6] Angles: {n_angles} projections over [0, pi), "
          f"step={angle_step_deg:.2f} deg")

    # 3. 解析 Radon 正变换 → 正弦图
    print("\n[Step 3/6] Analytic Radon transform (sinogram) ...", end=" ")
    sinogram = radon_transform_analytic(params, angles_rad,
                                        n_detectors, half_range)
    print(f"done, shape={sinogram.shape}")
    print(f"         projection range: [{sinogram.min():.4f}, "
          f"{sinogram.max():.4f}]")
    print(f"         projection mean:  {sinogram.mean():.4f}")

    # 4. 直接反投影
    print("\n[Step 4/6] Direct back-projection (unfiltered) ...", end=" ")
    t0 = __import__("time").time()
    bp_recon = back_project(sinogram, angles_rad, N, half_range)
    t_bp = __import__("time").time() - t0
    print(f"done, {t_bp:.2f}s")
    print(f"         BP range: [{bp_recon.min():.4f}, {bp_recon.max():.4f}]")

    # 5. FBP
    print("\n[Step 5/6] FBP (filtered back-projection) ...", end=" ")
    t0 = __import__("time").time()
    fbp_recon = fbp_reconstruct(sinogram, angles_rad, N, half_range)
    t_fbp = __import__("time").time() - t0
    print(f"done, {t_fbp:.2f}s")
    print(f"         FBP range: [{fbp_recon.min():.4f}, "
          f"{fbp_recon.max():.4f}]")

    # 6. 质量评估
    print("\n[Step 6/6] Quality assessment ...")

    bp_scaled, a_bp, b_bp = normalize_for_comparison(bp_recon, phantom)
    fbp_scaled, a_fbp, b_fbp = normalize_for_comparison(fbp_recon, phantom)

    mask = phantom > 0.05
    mse_bp = np.mean((bp_scaled[mask] - phantom[mask]) ** 2)
    mse_fbp = np.mean((fbp_scaled[mask] - phantom[mask]) ** 2)
    signal_var = np.var(phantom[mask])
    snr_bp = 10 * np.log10(signal_var / mse_bp) if mse_bp > 0 else float("inf")
    snr_fbp = 10 * np.log10(signal_var / mse_fbp) if mse_fbp > 0 else float("inf")

    rmse_bp = np.sqrt(mse_bp)
    rmse_fbp = np.sqrt(mse_fbp)

    print("-" * 60)
    print(f"  Metric         Back Projection    FBP")
    print("-" * 60)
    print(f"  RMSE           {rmse_bp:.4f}           {rmse_fbp:.4f}")
    print(f"  SNR (dB)       {snr_bp:6.2f}           {snr_fbp:6.2f}")
    print(f"  Improvement    -                  {snr_fbp - snr_bp:+.2f} dB")
    print("-" * 60)

    mid = N // 2
    print("\n" + "-" * 60)
    print(f"  Center row profile (row={mid}) — scaled values")
    print("-" * 60)
    print(f"  {'x':>5s}  {'Phantom':>8s}  {'BP':>10s}  {'FBP':>10s}")
    for col in [32, 64, 96, 128, 160, 192, 224]:
        print(f"  {col:5d}  {phantom[mid, col]:8.4f}  "
              f"{bp_scaled[mid, col]:10.4f}  {fbp_scaled[mid, col]:10.4f}")

    # 保存图片
    print("\n" + "-" * 60)
    print("  Generating 6-panel comparison figure ...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    fig.suptitle("CT Reconstruction: Radon Transform & Filtered Back-Projection",
                 fontsize=13, fontweight="bold", y=0.98)

    imshow_kw = dict(cmap="gray", interpolation="bilinear")

    # (a) 原始体模
    ax = axes[0, 0]
    im = ax.imshow(phantom, **imshow_kw, extent=[-1, 1, -1, 1])
    ax.set_title("(a) Original Phantom", fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Attenuation")

    # (b) 正弦图
    ax = axes[0, 1]
    aspect = n_detectors / n_angles * 0.5
    im = ax.imshow(sinogram, aspect=aspect, cmap="gray",
                   extent=[-half_range, half_range, n_angles, 0])
    ax.set_title("(b) Sinogram (Radon Transform)", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("Detector position $s$"); ax.set_ylabel("Angle index")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Projection $p(s,\\theta)$")

    # (c) 单个角度投影曲线（45度）
    ax = axes[0, 2]
    angle_idx = 45
    ax.plot(sinogram[angle_idx, :], "b-", linewidth=1.0)
    ax.set_title(f"(c) Projection at $\\theta = {angle_idx}^\\circ$",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Detector index $s$"); ax.set_ylabel("$p(s,\\theta)$")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_detectors)

    # (d) Ram-Lak 滤波器
    ax = axes[1, 0]
    freqs_vis = fftfreq(n_detectors)
    ramp_vis = np.abs(freqs_vis) * n_detectors
    ax.plot(freqs_vis, ramp_vis, "r-", linewidth=1.2)
    ax.set_title("(d) Ram-Lak Filter $|\\omega|$", fontsize=11,
                 fontweight="bold")
    ax.set_xlabel("Normalized frequency $\\omega/2\\pi$")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)

    # (e) 直接反投影结果
    ax = axes[1, 1]
    vmax_bp = np.percentile(bp_scaled, 99.5)
    vmin_bp = np.percentile(bp_scaled, 0.5)
    im = ax.imshow(bp_scaled, **imshow_kw, extent=[-1, 1, -1, 1],
                   vmax=vmax_bp, vmin=vmin_bp)
    ax.set_title(f"(e) Back Projection  (SNR={snr_bp:.1f}dB)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Reconstructed")

    # (f) FBP 重建结果
    ax = axes[1, 2]
    vmax_fbp = np.percentile(fbp_scaled, 99.5)
    vmin_fbp = np.percentile(fbp_scaled, 0.5)
    im = ax.imshow(fbp_scaled, **imshow_kw, extent=[-1, 1, -1, 1],
                   vmax=vmax_fbp, vmin=vmin_fbp)
    ax.set_title(f"(f) FBP Reconstruction  (SNR={snr_fbp:.1f}dB)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Reconstructed")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ct_reconstruction_result.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("  Key takeaway: FBP removes the blur inherent in direct")
    print("  back-projection by applying a ramp filter in Fourier domain.")
    print("=" * 60)


if __name__ == "__main__":
    main()
