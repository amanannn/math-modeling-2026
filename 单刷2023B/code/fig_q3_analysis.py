"""
2023B第三问 补充分析图 (数据类, 数据源: solve_q3.py)
图1: β方向扫描 — 证明β=90°(沿等深线)总长最短
图2: 重叠率下界敏感性 — 10%/15%/20% → 36/37/40条
图3: 三维布设+蛇形航线 — 坡面+条带随水深收窄+折返航迹(垂直放大20×)
输出: ../fig/fig_q3_beta_scan.png, fig_q3_sensitivity.png, fig_q3_3d.png
用法: python fig_q3_analysis.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from solve_q3 import (layout_west, simulate_beta, width,
                      XW, XE, W_NS, NM, H0, ta, K)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'


def layout_k(k):
    """间距 = k·W(下一条,较浅); k=0.9/0.85/0.8 ↔ 重叠率10/15/20%"""
    x1 = (XW + K * H0 / 2) / (1 + K * ta / 2)
    xs = [x1]
    while True:
        xn = (xs[-1] + k * K * H0) / (1 + k * K * ta)
        xs.append(xn)
        if xn + width(xn) / 2 >= XE:
            break
    return xs


# ===== 图1: β 方向扫描 =====
betas = np.arange(0, 181, 15)
l90 = simulate_beta(90)
rels = np.array([simulate_beta(b) / l90 for b in betas])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(betas, rels, 'o-', color='#2e86ab', lw=2, ms=7)
ax.axvline(90, color='#c0392b', ls='--', lw=1.2)
ax.annotate('β=90°(沿等深线)\n总长最短', xy=(90, rels.min()),
            xytext=(106, rels.max() - 0.10),
            arrowprops=dict(arrowstyle='->', color='#c0392b'),
            fontsize=12, color='#c0392b', fontweight='bold')
ax.set_xlabel('测线方向夹角 β (°)')
ax.set_ylabel('相对测线总长 (β=90°为 1.00)')
ax.set_title('测线总长随方向 β 变化 (扫描验证)')
ax.grid(True, alpha=0.3, ls='--')
plt.tight_layout()
plt.savefig(FIG / 'fig_q3_beta_scan.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[OK] fig_q3_beta_scan.png  '
      f'β=0:{rels[0]:.2f} β=45:{rels[3]:.2f} β=90:{rels[6]:.2f} '
      f'β=180:{rels[-1]:.2f}')

# ===== 图2: 重叠率下界敏感性 =====
etas = [0.10, 0.15, 0.20]
ns = [len(layout_k(1 - e)) for e in etas]
Ls = [n * W_NS / NM for n in ns]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([f'{int(e * 100)}%' for e in etas], Ls, width=0.5,
              color=['#2e86ab', '#5fa8d3', '#8ec9e0'])
for b, n, L in zip(bars, ns, Ls):
    ax.text(b.get_x() + b.get_width() / 2, L + 1.2,
            f'{n}条\n{L:.0f}海里', ha='center', fontsize=12,
            fontweight='bold', color='#1f4e79')
ax.set_ylabel('测线总长 (海里)')
ax.set_xlabel('重叠率设计下界')
ax.set_title('重叠率下界敏感性: 每放宽5%约增1条测线')
ax.set_ylim(0, max(Ls) * 1.22)
ax.grid(True, axis='y', alpha=0.3, ls='--')
plt.tight_layout()
plt.savefig(FIG / 'fig_q3_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[OK] fig_q3_sensitivity.png  '
      f'η=10%→{ns[0]}条 η=15%→{ns[1]}条 η=20%→{ns[2]}条')

# ===== 图3: 彩色海底 + 蛇形航线 =====
ZS = 12.0                    # 垂直放大(坡度可见)
PAD = 450.0                  # 坡面外扩(容纳折返弧)
xs = np.array(layout_west())
cmap = plt.cm.jet_r          # 西深(红)→东浅(蓝)
norm = Normalize(vmin=XW, vmax=XE)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 海平面(极淡, 承载航线)
xw = np.array([XW - PAD, XE + PAD])
yw = np.array([-W_NS / 2 - PAD, W_NS / 2 + PAD])
Xw, Yw = np.meshgrid(xw, yw)
ax.plot_surface(Xw, Yw, np.zeros_like(Xw), color=[0.42, 0.68, 0.9],
                alpha=0.08, edgecolor='none')

# 海底坡面(彩色, 网格纹理)
xx = np.linspace(XW, XE, 30)
yy = np.linspace(-W_NS / 2 - PAD, W_NS / 2 + PAD, 12)
Xg, Yg = np.meshgrid(xx, yy)
Zg = -(H0 - Xg * ta) * ZS
ax.plot_surface(Xg, Yg, Zg, facecolors=cmap(norm(Xg)), alpha=0.65,
                edgecolor='gray', linewidth=0.3, shade=False)

# 蛇形航线(海面z=0): 测线黑 / 折返橙
for i, x in enumerate(xs):
    ax.plot([x, x], [-W_NS / 2, W_NS / 2], [0, 0], 'k-', lw=1.3)
for i in range(len(xs) - 1):
    xa, xb = xs[i], xs[i + 1]
    r = (xb - xa) / 2
    xm = (xa + xb) / 2
    if i % 2 == 0:                      # 北端折返: 上半圆
        th = np.linspace(np.pi, 0, 50)
        yv = W_NS / 2 + r * np.sin(th)
    else:                               # 南端折返: 下半圆
        th = np.linspace(np.pi, 2 * np.pi, 50)
        yv = -W_NS / 2 + r * np.sin(th)
    xv = xm + r * np.cos(th)
    ax.plot(xv, yv, np.zeros_like(xv), color='#e67e22', lw=1.6)

# 起终点 + 海域边界(海面虚线框)
ax.plot([xs[0]], [-W_NS / 2], [0], 'o', color='#1a7f37', ms=9)
ax.plot([xs[-1]], [-W_NS / 2], [0], 's', color='#c0392b', ms=8)
for xb in (XW, XE):
    ax.plot([xb, xb], [-W_NS / 2, W_NS / 2], [0, 0], 'k--', lw=1)
for yb in (-W_NS / 2, W_NS / 2):
    ax.plot([XW, XE], [yb, yb], [0, 0], 'k--', lw=1)

ax.set_xlabel('东西 (m)')
ax.set_ylabel('南北 (m)')
ax.set_zlabel('水深 (m, ×12)')
ax.set_title(f'第三问蛇形航线布设 ({len(xs)}条, 重叠率10%)', fontsize=13)
ax.set_zlim(-220 * ZS, 20 * ZS)
ax.set_zticks([-2400, -1200, 0])         # 刻度显示真实水深
ax.set_zticklabels(['-200', '-100', '0'])
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label('水深: 西深(红) → 东浅(蓝)')
ax.view_init(elev=25, azim=-60)
ax.set_box_aspect((2.4, 1.5, 0.55))
fig.subplots_adjust(left=0.03, right=0.95, bottom=0.05, top=0.93)
plt.savefig(FIG / 'fig_q3_3d.png', dpi=300)
plt.close(fig)
print(f'[OK] fig_q3_3d.png ({len(xs)}条航线, 彩色海底+航线)')
