"""
2023B第四问 测线方向扫描 (诊断用)
对θ∈[0,180)每10°, 平行线族贪心布设+覆盖评估, 找全局最优方向
用法: python scan_q4_direction.py
"""
import numpy as np
import solve_q4 as q
from scipy.spatial.distance import cdist

NM = q.NM
corners = np.array([[0, 0], [4 * NM, 0], [4 * NM, 5 * NM], [0, 5 * NM]])
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]


def w_field(n):
    """测线法线n方向的条带宽度场: 视坡角=梯度在n方向的投影"""
    ta = np.abs(q.gx * n[0] + q.gy * n[1])          # tan(视坡角)
    ta = np.minimum(ta, q.t30 * 0.99)
    return q.H * (1 / (q.t30 + ta) + 1 / (q.t30 - ta))


def scan_theta(theta_deg):
    """θ方向平行线族: 贪心间距(线上W_θ中位数), 返回(总长海里, 漏测率)"""
    th = np.deg2rad(theta_deg)
    t = np.array([np.cos(th), np.sin(th)])
    n = np.array([-np.sin(th), np.cos(th)])
    Wth = w_field(n).ravel()
    cn = corners @ n
    cmin, cmax = cn.min(), cn.max()

    def line(c):
        """法线n·P=c 与海域的截线段(采样点)"""
        ss = []
        for a, b in edges:
            p1, p2 = corners[a], corners[b]
            denom = n @ (p2 - p1)
            if abs(denom) < 1e-9:
                continue
            s = (c - n @ p1) / denom
            if -1e-6 <= s <= 1 + 1e-6:
                ss.append(p1 + s * (p2 - p1))
        if len(ss) < 2:
            return None
        p, r = ss[0], ss[1]
        L = np.linalg.norm(r - p)
        k = max(2, int(L / q.DX))
        return p + np.linspace(0, 1, k)[:, None] * (r - p)

    cov = np.zeros(q.n, bool)
    total, c = 0.0, cmin
    while c < cmax - 1e-6:
        l = line(c)
        if l is None:
            break
        d = cdist(q.GRID, l).min(axis=1)
        cov |= (d <= Wth / 2)
        total += np.linalg.norm(np.diff(l, axis=0), axis=1).sum()
        w_med = np.median(Wth[np.round(l[:, 1] / q.DX).astype(int) * q.NX
                              + np.round(l[:, 0] / q.DX).astype(int)])
        c += 0.9 * w_med
    return total / NM, 1 - cov.mean()


print(f'{"θ(°)":>6}{"总长/海里":>10}{"漏测率":>9}')
for th in range(0, 180, 10):
    L, miss = scan_theta(th)
    print(f'{th:>6}{L:>10.1f}{100*miss:>8.2f}%')
