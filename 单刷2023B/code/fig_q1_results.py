"""
2023B第一问 结果可视化 (数据类)
图1: 覆盖带俯视图 (条带边框/空隙白底/中心虚线/学术标题)
图2: W与重叠率趋势
输出: ../fig/fig_coverage_overview.png, ../fig/fig_w_eta_trend.png
用法: python fig_q1_results.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'

x_lines = np.array([-800, -600, -400, -200, 0, 200, 400, 600, 800])
W = np.array([315.60, 297.41, 279.23, 261.08, 242.90,
              224.72, 206.57, 188.39, 170.20])
d = 200.0
L = 2200

# ============================================================
# 图1: 覆盖带俯视图
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7.5))

# --- 每条测线覆盖带: 浅蓝 + 黑色细虚线边框 ---
for y, w in zip(x_lines, W):
    half = w / 2
    ax.add_patch(plt.Rectangle((0, y - half), L, w,
                               facecolor='#2e86ab', alpha=0.25,
                               edgecolor='k', lw=0.8, ls='--'))
    ax.plot([0, L], [y, y], 'k:', lw=0.9)          # 测线中心轨迹

# --- 相邻测线重叠区 (棕色) ---
for i in range(len(x_lines) - 1):
    top1 = x_lines[i] + W[i] / 2
    bot2 = x_lines[i + 1] - W[i + 1] / 2
    if top1 > bot2:
        ax.add_patch(plt.Rectangle((0, bot2), L, top1 - bot2,
                                   facecolor='#e67e22', alpha=0.55,
                                   edgecolor='none'))

# --- 空隙: 先抠白底, 再画红斜线 ---
for i in range(len(x_lines) - 1):
    top1 = x_lines[i] + W[i] / 2
    bot2 = x_lines[i + 1] - W[i + 1] / 2
    if top1 < bot2:
        ax.add_patch(plt.Rectangle((0, top1), L, bot2 - top1,
                                   facecolor='white', edgecolor='none'))
        ax.add_patch(plt.Rectangle((0, top1), L, bot2 - top1,
                                   facecolor='none', hatch='//',
                                   edgecolor='#c0392b', lw=1.2))

gap = (x_lines[-1] - W[-1] / 2) - (x_lines[-2] + W[-2] / 2)
if gap > 0:
    ax.annotate(f'空隙 (未覆盖)\nΔ = {gap:.1f} m',
                xy=(L / 2, x_lines[-1] - W[-1] / 2 - gap / 2),
                xytext=(L * 0.63, x_lines[-1] - W[-1] / 2 + 90),
                arrowprops=dict(arrowstyle='->', color='#c0392b'),
                fontsize=11, color='#c0392b', fontweight='bold')

# --- 标注与图例 ---
ax.text(-60, -830, '深水侧\nW 大', ha='center', fontsize=11, color='#1f618d')
ax.text(-60, 830, '浅水侧\nW 小', ha='center', fontsize=11, color='#b03a2e')
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#2e86ab', alpha=0.25,
                           ls='--', edgecolor='k', label='单条测线覆盖带'))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#e67e22', alpha=0.55,
                           label='相邻测线重叠区'))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', hatch='//',
                           edgecolor='#c0392b', label='覆盖空隙'))
ax.plot([0, 1], [0, 0], 'k:', lw=0.9, label='测线中心轨迹')
ax.legend(loc='lower right', fontsize=10)

ax.set_xlim(-100, L + 100)
ax.set_ylim(-1000, 1000)
ax.set_xlabel('沿测线方向 (等深线方向, m)', fontsize=12)
ax.set_ylabel('垂直测线方向 (横向, m)', fontsize=12)
ax.grid(True, alpha=0.15, ls='--')
ax.set_aspect('auto')
plt.savefig(FIG / 'fig_coverage_overview.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================================
# 图2: W 与重叠率趋势
# ============================================================
eta = np.full_like(W, np.nan)
for i in range(1, len(W)):
    eta[i] = 1 - d / W[i - 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(x_lines, W, 'o-', color='#2e86ab', lw=2, ms=7)
ax1.axhline(d, color='#c0392b', ls='--', lw=1.5, label='测线间距 d=200m')
ax1.fill_between(x_lines, d, W, where=(W > d), color='#2e86ab', alpha=0.15)
ax1.set_xlabel('测线距中心点距离 (m)')
ax1.set_ylabel('覆盖宽度 W (m)')
ax1.set_title('覆盖宽度随测线位置变化')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, ls='--')
ax1.annotate('W > d: 有重叠', xy=(400, 240), fontsize=9, color='#2e86ab')
ax1.annotate('W < d: 空隙', xy=(-700, 150), fontsize=9, color='#c0392b')

ax2.plot(x_lines[1:], eta[1:] * 100, 's-', color='#e67e22', lw=2, ms=6)
ax2.axhline(0, color='gray', lw=1)
ax2.set_xlabel('测线距中心点距离 (m)')
ax2.set_ylabel('与前一条测线重叠率 (%)')
ax2.set_title('重叠率随测线位置变化')
ax2.grid(True, alpha=0.3, ls='--')
ax2.annotate('重叠率随位置递减', xy=(-550, 28), fontsize=10, color='#e67e22')
ax2.annotate('转为负值=空隙', xy=(550, -8), fontsize=10, color='#c0392b')

plt.tight_layout()
plt.savefig(FIG / 'fig_w_eta_trend.png', dpi=300, bbox_inches='tight')
plt.close(fig)
