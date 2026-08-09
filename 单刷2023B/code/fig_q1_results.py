"""
2023B第一问 结果可视化 (数据类, 数据源: solve_q1.py)
图: 覆盖带俯视图 (条带边框/空隙白底/中心虚线/学术标题)
输出: ../fig/fig_q1_coverage_overview.png
用法: python fig_q1_results.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from solve_q1 import X as x_lines, depth, width, D_LINE

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'

W = width(depth(x_lines))     # 与 solve_q1.py 同源, 不硬编码
d = D_LINE
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
plt.savefig(FIG / 'fig_q1_coverage_overview.png', dpi=300,
            bbox_inches='tight')
plt.close(fig)
