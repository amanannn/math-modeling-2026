"""
用途: 2023B第一问结果可视化
  - 图1: 俯视覆盖带图 (测线条带+重叠区域+空隙)
  - 图2: W与重叠率随测线位置变化趋势
输出: D:/虚拟C盘/数学建模培训/单刷2023B/fig/fig_coverage_overview.png, D:/虚拟C盘/数学建模培训/单刷2023B/fig/fig_w_eta_trend.png
调用: python models/mbs/fig_q1_results.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 第一问表格数据 =====
x_lines = np.array([-800, -600, -400, -200, 0, 200, 400, 600, 800])
D = np.array([90.95, 85.71, 80.47, 75.24, 70.00, 64.76, 59.53, 54.29, 49.05])
W = np.array([315.60, 297.41, 279.23, 261.08, 242.90, 224.72, 206.57, 188.39, 170.20])
d = 200  # 测线间距

# ============================================================
# 图1: 俯视覆盖带图
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

L = 2200  # 测线方向长度 (m)
line_color = '#2e86ab'
overlap_color = '#e67e22'
gap_color = 'none'

# 画每条测线的覆盖带
for i, (y, w) in enumerate(zip(x_lines, W)):
    half = w / 2
    # 覆盖带 (半透明蓝)
    ax.add_patch(plt.Rectangle((0, y - half), L, w,
                               facecolor=line_color, alpha=0.35,
                               edgecolor='none'))
    # 测线本身 (中线, 深色线)
    ax.plot([0, L], [y, y], color=line_color, lw=1.8)

# 相邻测线重叠区域 (橙色加深)
for i in range(len(x_lines) - 1):
    y1, w1 = x_lines[i], W[i]          # 深水侧
    y2, w2 = x_lines[i + 1], W[i + 1]  # 浅水侧
    top1 = y1 + w1 / 2
    bottom2 = y2 - w2 / 2
    if top1 > bottom2:
        ax.add_patch(plt.Rectangle((0, bottom2), L, top1 - bottom2,
                                   facecolor=overlap_color, alpha=0.6,
                                   edgecolor='none'))
    elif top1 < bottom2:
        # 空隙
        ax.add_patch(plt.Rectangle((0, top1), L, bottom2 - top1,
                                   facecolor='white', alpha=0.9,
                                   hatch='//', edgecolor='#c0392b', lw=1.2))

# 标注空隙
top_last = x_lines[-1] - W[-1] / 2
bottom_prev = x_lines[-2] + W[-2] / 2
if top_last > bottom_prev:
    ax.annotate('空隙 (未覆盖区)\nΔ = %.1f m' % (top_last - bottom_prev),
                xy=(L / 2, (top_last + bottom_prev) / 2),
                xytext=(L * 0.62, top_last + 60),
                arrowprops=dict(arrowstyle='->', color='#c0392b'),
                fontsize=11, color='#c0392b', fontweight='bold')

# 标注深水侧/浅水侧
ax.text(-60, -830, '深水侧\n(测线-800)', ha='center', fontsize=10, color='#1f618d')
ax.text(-60, 830, '浅水侧\n(测线+800)', ha='center', fontsize=10, color='#b03a2e')

# 图例
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=line_color, alpha=0.35,
                           label='单条测线覆盖带 (宽=W)'))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=overlap_color, alpha=0.6,
                           label='相邻测线重叠区'))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', hatch='//',
                           edgecolor='#c0392b', label='覆盖空隙'))
ax.legend(loc='lower right', fontsize=10)

ax.set_xlim(-100, L + 100)
ax.set_ylim(-1000, 1000)
ax.set_xlabel('沿测线方向 (等深线方向, m)', fontsize=11)
ax.set_ylabel('垂直测线方向 (横向, m)', fontsize=11)
ax.set_title('第一问：多波束测线覆盖带俯视图 (测线间距 d=200m)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.set_aspect('auto')

plt.tight_layout()
plt.savefig('D:/虚拟C盘/数学建模培训/单刷2023B/fig/fig_coverage_overview.png',
            dpi=300, bbox_inches='tight')
plt.close(fig)
print('[OK] fig_coverage_overview.png')

# ============================================================
# 图2: W 与重叠率趋势
# ============================================================
eta = np.full_like(W, np.nan)
for i in range(1, len(W)):
    eta[i] = 1 - d / W[i - 1]   # 与前一条测线的重叠率

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 左: W 随测线位置
ax1.plot(x_lines, W, 'o-', color='#2e86ab', lw=2, ms=7)
ax1.axhline(d, color='#c0392b', ls='--', lw=1.5, label='测线间距 d=200m')
ax1.fill_between(x_lines, d, W, where=(W > d), color='#2e86ab', alpha=0.15)
ax1.set_xlabel('测线距中心点距离 (m)')
ax1.set_ylabel('覆盖宽度 W (m)')
ax1.set_title('覆盖宽度随测线位置变化')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.annotate('W > d: 有重叠', xy=(400, 240), fontsize=9, color='#2e86ab')
ax1.annotate('W < d: 空隙', xy=(-700, 150), fontsize=9, color='#c0392b')

# 右: 重叠率随测线位置
ax2.plot(x_lines[1:], eta[1:] * 100, 's-', color='#e67e22', lw=2, ms=6)
ax2.axhline(0, color='gray', lw=1)
ax2.set_xlabel('测线距中心点距离 (m)')
ax2.set_ylabel('与前一条测线重叠率 (%)')
ax2.set_title('重叠率随测线位置变化 (瓶颈在浅水侧)')
ax2.grid(True, alpha=0.3)
ax2.annotate('重叠率随位置递减', xy=(-550, 28), fontsize=10, color='#e67e22')
ax2.annotate('转为负值=空隙', xy=(550, -8), fontsize=10, color='#c0392b')

plt.tight_layout()
plt.savefig('D:/虚拟C盘/数学建模培训/单刷2023B/fig/fig_w_eta_trend.png',
            dpi=300, bbox_inches='tight')
plt.close(fig)
print('[OK] fig_w_eta_trend.png')
