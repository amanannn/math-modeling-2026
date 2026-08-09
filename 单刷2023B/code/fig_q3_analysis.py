"""
2023B第三问 三维布设+蛇形航线图 (数据源: solve_q3.py)
图: 彩色海底坡面 + 条带 + 蛇形折返航线(航线在海面)
输出: ../fig/fig_q3_3d.png
用法: python fig_q3_analysis.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from solve_q3 import layout_west, width, XW, XE, W_NS, H0, ta

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'


# ===== 彩色海底 + 蛇形航线 =====
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
