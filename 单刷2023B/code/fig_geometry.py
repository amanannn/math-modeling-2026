"""
用途: 2023B多波束测线 截面几何示意图 (Matplotlib)
输出: output/fig_geometry.png
修正: z向下为正, 图上用 -z 绘制, 海平面在上, 海底在下
调用: python models/mbs/fig_geometry.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

alpha = np.deg2rad(1.5)   # 坡面倾角
D = 70.0                  # 中心水深
t30 = np.tan(np.deg2rad(30))   # 波束边缘与水平面夹角 30°

# 水平坐标 y (向坡下为正), 竖直绘制坐标 yp = -z (向下为深)
y = np.linspace(-160, 160, 200)
z_slope = D + y * np.tan(alpha)   # 真实深度 (向下)
yp_slope = -z_slope               # 绘图坐标 (海平面=0, 向下为负)

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.set_aspect('equal')

# ===== 海平面 (yp=0) =====
ax.axhline(0, color='#1f4e79', lw=2.2)
ax.text(163, 8, '海平面', ha='right', color='#1f4e79', fontsize=11)

# ===== 海底坡面 (下方) =====
ax.plot(y, yp_slope, color='#8b4513', lw=2.5, label='海底坡面 (α=1.5°)')
ax.fill_between(y, yp_slope, -130, color='#d2b48c', alpha=0.3)

# ===== 船 (海平面上 yp=0) =====
ax.plot(0, 0, 'ks', ms=11, zorder=5)
ax.annotate('船', (0, 0), xytext=(10, 4), fontsize=11, fontweight='bold')

# ===== 波束边缘射线 (向下, 与竖直成60°) =====
for sign in (-1, 1):
    yy = np.linspace(0, sign * 130, 100)
    zz = -np.abs(yy) * t30        # 向下: yp = -|y|·tan30°
    ax.plot(yy, zz, color='#2e75b6', lw=1.8, alpha=0.85)
# 船到海底垂线 (水深)
ax.plot([0, 0], [0, -D], 'k--', lw=1.2, alpha=0.6)

# ===== 交点 =====
yL = -D / (t30 + np.tan(alpha))
yR = D / (t30 - np.tan(alpha))
zL = D + yL * np.tan(alpha)
zR = D + yR * np.tan(alpha)
ax.plot([yL, yR], [-zL, -zR], 'ro', ms=7, zorder=5)
ax.annotate('P_L', (yL, -zL), xytext=(yL - 45, -zL - 18),
            fontsize=11, color='red', fontweight='bold')
ax.annotate('P_R', (yR, -zR), xytext=(yR + 6, -zR - 18),
            fontsize=11, color='red', fontweight='bold')

# ===== 覆盖宽度 W (海平面基准, 贴在海平面上方) =====
ax.annotate('', xy=(yL, 12), xytext=(yR, 12),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.8))
ax.text((yL + yR) / 2, 24, f'W = {yR - yL:.1f} m',
        ha='center', color='red', fontsize=12, fontweight='bold')

# ===== 角度标注 (集中在船附近) =====
# 30°: 波束边缘与海平面的夹角 (数学角 -180°~-150°, 小弧)
ax.add_patch(Arc((0, 0), 30, 30, theta1=-180, theta2=-150, color='#2e75b6', lw=1.5))
ax.text(-30, -9, '30°', color='#2e75b6', fontsize=10,
        ha='center', va='center')
# 60°: 波束边缘与竖直向下方向的夹角 (数学角 -150°~-90°, 稍大弧)
ax.add_patch(Arc((0, 0), 46, 46, theta1=-150, theta2=-90, color='#2e75b6', lw=1.5))
ax.text(-23, -40, '60°', color='#2e75b6', fontsize=10,
        ha='center', va='center')
# 坡角 α (海底与水平线夹角, 紧贴坡面)
ax.add_patch(Arc((0, -D), 24, 24, theta1=-180, theta2=-180 + np.rad2deg(alpha),
                 color='#7030a0', lw=1.5))
ax.text(-14, -D + 5, f'α={np.rad2deg(alpha):.1f}°',
        color='#7030a0', fontsize=10, ha='center', va='center')

# ===== 水深标注 =====
ax.annotate('', xy=(12, 0), xytext=(12, -D),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.2, alpha=0.7))
ax.text(17, -D / 2, f'D={D}m', fontsize=10, va='center')

ax.set_xlim(-165, 165)
ax.set_ylim(-110, 60)
ax.set_xlabel('y (m)  向坡下为正', fontsize=11)
ax.set_ylabel('z (m)  深度(向下)', fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('D:/虚拟C盘/数学建模培训/单刷2023B/fig/fig_geometry.png',
            dpi=300, bbox_inches='tight')
print('[OK] output/fig_geometry.png')
