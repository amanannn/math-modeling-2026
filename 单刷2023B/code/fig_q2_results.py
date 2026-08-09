"""
2023B第二问 结果可视化 (数据类)
图: 合成图(左W(L,β)三维曲面 + 右热力图)
输出: ../fig/fig_q2_combined.png
用法: python fig_q2_results.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'

H0 = 120.0
ALPHA = np.deg2rad(1.5)
NM = 1852.0
BETA = np.array([0, 45, 90, 135, 180, 225, 270, 315])
L_NM = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1])
t30 = np.tan(np.pi / 6)
ta = np.tan(ALPHA)

# ===== 计算 W(L,β) (与 solve_q2.py 一致) =====
W = np.zeros((len(BETA), len(L_NM)))
for i, b in enumerate(BETA):
    cb, sb = np.cos(np.deg2rad(b)), np.sin(np.deg2rad(b))
    tap = ta * abs(sb)
    for j, lnm in enumerate(L_NM):
        H = H0 - lnm * NM * ta * cb
        W[i, j] = H * (1 / (t30 + tap) + 1 / (t30 - tap))

Bm, Lm = np.meshgrid(BETA, L_NM)
Wm = W.T
Bq, Lq = np.meshgrid(np.linspace(0, 315, 80), np.linspace(0, 2.1, 80))
Wq = griddata((Bm.ravel(), Lm.ravel()), Wm.ravel(), (Bq, Lq),
              method='cubic')


def draw_surface(ax, fs=11):
    surf = ax.plot_surface(Bq, Lq, Wq, cmap='jet', edgecolor='gray',
                           linewidth=0.25, antialiased=True, alpha=0.95)
    ax.scatter(Bm, Lm, Wm, c='gray', s=8, alpha=0.6)
    ax.set_xlabel('测线方向夹角 β (°)', fontsize=fs)
    ax.set_ylabel('距中心点距离 L (海里)', fontsize=fs)
    ax.set_zlabel('覆盖宽度 W (m)', fontsize=fs)
    ax.view_init(elev=30, azim=-55)
    return surf


def draw_heatmap(ax, fs=11):
    cmap = plt.cm.viridis
    im = ax.imshow(W, origin='lower', aspect='auto', cmap=cmap)
    for i in range(len(BETA)):
        for j in range(len(L_NM)):
            r, g, b, _ = cmap((W[i, j] - W.min()) / (W.max() - W.min()))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f'{W[i,j]:.0f}', ha='center', va='center',
                    color='k' if lum > 0.6 else 'w', fontsize=fs,
                    fontweight='bold')
    ax.set_xticks(range(len(L_NM)))
    ax.set_xticklabels(L_NM)
    ax.set_yticks(range(len(BETA)))
    ax.set_yticklabels(BETA)
    ax.set_xlabel('距中心点距离 L (海里)', fontsize=fs)
    ax.set_ylabel('测线方向夹角 β (°)', fontsize=fs)
    return im


# ===== 合成图: 左曲面 + 右热力图 =====
fig = plt.figure(figsize=(15, 6.5))
ax1 = fig.add_subplot(121, projection='3d')
surf = draw_surface(ax1, fs=10)
fig.colorbar(surf, ax=ax1, shrink=0.55, label='W (m)')
ax1.set_title('(a) 覆盖宽度三维曲面', fontsize=12)

ax2 = fig.add_subplot(122)
im = draw_heatmap(ax2, fs=10)
fig.colorbar(im, ax=ax2, label='覆盖宽度 W (m)')
ax2.set_title('(b) 覆盖宽度热力图', fontsize=12)

plt.tight_layout()
plt.savefig(FIG / 'fig_q2_combined.png', dpi=300, bbox_inches='tight')
plt.close(fig)
