"""
2023B第四问 测线布设图 (数据源: solve_q4.py)
图1: 贪心@110°方案(黑=主线, 橙=漏测补线); 图2: PWS曲线方案对比
输出: ../fig/fig_q4_layout.png, fig_q4_pws.png
用法: python fig_q4_layout.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import solve_q4 as q

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'
TH = 110.0


def draw_bg(ax):
    """水深地形 + 海域边界"""
    im = ax.imshow(q.H, origin='lower', extent=[0, 4, 0, 5],
                   cmap='jet_r', aspect='auto', alpha=0.55)
    ax.add_patch(plt.Rectangle((0, 0), 4, 5, fill=False, edgecolor='k',
                               lw=1.2, ls='--'))
    ax.set_xlabel('东西 (海里)')
    ax.set_ylabel('南北 (海里)')
    return im


# ===== 图1: 贪心@110° (主线+补线) =====
l_main, c1 = q.solve_lines(TH)
n_main = len(l_main)
l_all, c_all = q.repair_miss(l_main.copy(), c1.copy(), TH)

fig, ax = plt.subplots(figsize=(8, 9))
im = draw_bg(ax)
for k, l in enumerate(l_all):
    col = 'k' if k < n_main else '#e67e22'
    ax.plot(l[:, 0] / q.NM, l[:, 1] / q.NM, '-', color=col, lw=0.8)
miss = (~c_all).reshape(q.NY, q.NX)
ax.imshow(np.ma.masked_where(~miss, miss.astype(float)),
          origin='lower', extent=[0, 4, 0, 5], cmap='Reds', alpha=0.5,
          aspect='auto')
ax.set_title(f'贪心@{TH:.0f}°: {len(l_all)}条, 漏测{100*(1-c_all.mean()):.2f}%')
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label('水深 (m)')
plt.tight_layout()
plt.savefig(FIG / 'fig_q4_layout.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[OK] fig_q4_layout.png ({len(l_all)}条)')

# ===== 图2: PWS 曲线方案 =====
l_pws, c_pws = q.solve_pws()
fig, ax = plt.subplots(figsize=(8, 9))
im = draw_bg(ax)
for l in l_pws:
    ax.plot(l[:, 0] / q.NM, l[:, 1] / q.NM, 'k-', lw=0.7)
ax.set_title(f'PWS惩罚加权最短路径: {len(l_pws)}条, '
             f'漏测{100*(1-c_pws.mean()):.2f}%')
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label('水深 (m)')
plt.tight_layout()
plt.savefig(FIG / 'fig_q4_pws.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[OK] fig_q4_pws.png ({len(l_pws)}条)')
