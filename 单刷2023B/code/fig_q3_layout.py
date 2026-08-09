"""
2023B第三问 测线布设与完整航迹图 (数据源: solve_q3.py 最终方案)
图1: 布局图 (36条南北向测线); 图2: 完整航迹图 (蛇形折返, 含转弯段)
输出: ../fig/fig_q3_layout.png, ../fig/fig_q3_track.png
用法: python fig_q3_layout.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from solve_q3 import layout_west, depth, width, XW, XE, W_NS, W_EW, NM

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

FIG = Path(__file__).resolve().parent.parent / 'fig'

xs = np.array(layout_west())


def draw_bg(ax):
    """水深渐变背景 + 海域边界 + 坐标"""
    xx = np.linspace(XW, XE, 300)
    yy = np.linspace(-W_NS / 2, W_NS / 2, 120)
    Hm = 110.0 - xx * np.tan(np.deg2rad(1.5))
    im = ax.imshow(np.tile(Hm, (len(yy), 1)),
                   extent=[XW, XE, -W_NS / 2, W_NS / 2],
                   aspect='auto', cmap='jet_r', alpha=0.5,
                   norm=Normalize(vmin=Hm.min(), vmax=Hm.max()))
    ax.add_patch(plt.Rectangle((XW, -W_NS / 2), W_EW, W_NS, fill=False,
                               edgecolor='k', lw=1.5, ls='--'))
    xt = np.arange(XW, XE + 1, NM)
    yt = np.arange(-W_NS / 2, W_NS / 2 + 1, NM / 2)
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{v / NM:.1f}' for v in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{v / NM:.1f}' for v in yt])
    ax.set_xlabel('东西 (海里)')
    ax.set_ylabel('南北 (海里)')
    return im


def draw_layout(ax):
    """36条测线 + 每4条一个航向箭头"""
    for i, x in enumerate(xs):
        ax.plot([x, x], [-W_NS / 2, W_NS / 2], 'k-', lw=0.9)
        if i % 4 == 0:
            ax.annotate('', xy=(x, -W_NS / 2 + 260), xytext=(x, W_NS / 2 - 260),
                        arrowprops=dict(arrowstyle='->', color='k', lw=1.2))


def draw_track(ax):
    """测线 + 蛇形折返弧(北上半圆/南下半圆) + 起终点 + 图例"""
    for i, x in enumerate(xs):
        ax.plot([x, x], [-W_NS / 2, W_NS / 2], 'k-', lw=0.9, zorder=3,
                label='测线' if i == 0 else None)
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        r = (xb - xa) / 2
        xm = (xa + xb) / 2
        if i % 2 == 0:
            th = np.linspace(np.pi, 0, 50)
            yy = W_NS / 2 + r * np.sin(th)
        else:
            th = np.linspace(np.pi, 2 * np.pi, 50)
            yy = -W_NS / 2 + r * np.sin(th)
        ax.plot(xm + r * np.cos(th), yy, color='#e67e22', lw=1.2,
                alpha=0.85, zorder=3, label='折返段' if i == 0 else None)
    ax.plot(xs[0], -W_NS / 2, 'o', color='#1a7f37', ms=9, zorder=4,
            label='起点')
    ax.plot(xs[-1], -W_NS / 2, 's', color='#c0392b', ms=8, zorder=4,
            label='终点')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=4,
              frameon=False, fontsize=11)


if __name__ == '__main__':
    # ===== 图1: 布局图 =====
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = draw_bg(ax)
    draw_layout(ax)
    fig.text(0.015, 0.985, '西(深)', ha='left', va='top', fontsize=11,
             color='navy', fontweight='bold')
    fig.text(0.985, 0.985, f'{len(xs)}条 / 总长{len(xs) * W_NS / NM:.0f}海里',
             ha='right', va='top', fontsize=11, color='dimgray',
             fontweight='bold')
    fig.text(0.985, 0.955, '东(浅)', ha='right', va='top', fontsize=11,
             color='darkred', fontweight='bold')
    ax.set_title(f'第三问测线布设 ({len(xs)}条, 重叠率10%)', fontsize=13)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('水深 (m)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_q3_layout.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {FIG / "fig_q3_layout.png"} ({len(xs)}条)')

    # ===== 图2: 完整航迹图 =====
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = draw_bg(ax)
    draw_track(ax)
    fig.text(0.015, 0.985, '西(深)', ha='left', va='top', fontsize=11,
             color='navy', fontweight='bold')
    fig.text(0.985, 0.985, '完整航迹 ≈ 78 海里', ha='right', va='top',
             fontsize=11, color='dimgray', fontweight='bold')
    fig.text(0.985, 0.955, '东(浅)', ha='right', va='top', fontsize=11,
             color='darkred', fontweight='bold')
    ax.set_title('第三问完整测量航迹 (蛇形折返, 测线72+转弯6海里)', fontsize=13)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('水深 (m)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_q3_track.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {FIG / "fig_q3_track.png"}')
