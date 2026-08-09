"""
2023B第四问 真实地形测线布设 (附件.xlsx, 南北5海里×东西4海里)
算法:
  ① 方向扫描: θ∈[0,180) 平行线族贪心布设, 漏测率最小方向为最优
     (条带宽度用沿测线法线的视坡角, 方向相关)
  ② 贪心布设: 间距=0.9·线上W中位数, 逐条推进
  ③ 漏测修复: 漏测连通块中心沿主线方向补线
指标: 测线总长/漏测率/超20%重叠长度
用法: python solve_q4.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
NM = 1852.0
t30 = np.tan(np.pi / 6)

# ===== 读取与地形场 =====
df = pd.read_excel(ROOT / '附件.xlsx', header=None)
H = df.iloc[2:, 2:].astype(float).values   # (251, 201) 行:南→北 列:西→东
NY, NX = H.shape
DX = 0.02 * NM
gy, gx = np.gradient(H, DX, DX)
GN = np.hypot(gx, gy)

n = NY * NX
Wr = (H * (1 / (t30 + GN) + 1 / (t30 - GN))).ravel()   # 沿梯度方向宽度(参考)
grid_y, grid_x = np.meshgrid(np.arange(NY) * DX, np.arange(NX) * DX,
                             indexing='ij')
GRID = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)   # (n, 2)

W_EW, W_NS = 4 * NM, 5 * NM
CORNERS = np.array([[0, 0], [W_EW, 0], [W_EW, W_NS], [0, W_NS]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def w_field(nx, ny):
    """测线法线(nx,ny)方向的条带宽度场(视坡角=梯度沿法线投影)"""
    ta = np.abs(gx * nx + gy * ny)
    ta = np.minimum(ta, t30 * 0.99)
    return H * (1 / (t30 + ta) + 1 / (t30 - ta))


def solve_lines(theta_deg, quantile=0.35):
    """θ方向平行线族贪心布设: 间距=0.9·线上W分位数, 返回(lines, cov)"""
    th = np.deg2rad(theta_deg)
    t = np.array([np.cos(th), np.sin(th)])
    nv = np.array([-np.sin(th), np.cos(th)])
    Wth = w_field(nv[0], nv[1]).ravel()
    cn = CORNERS @ nv
    cmin, cmax = cn.min(), cn.max()

    def line(c):
        """法线nv·P=c 与海域的截线段(采样点)"""
        ss = []
        for a, b in EDGES:
            p1, p2 = CORNERS[a], CORNERS[b]
            denom = nv @ (p2 - p1)
            if abs(denom) < 1e-9:
                continue
            s = (c - nv @ p1) / denom
            if -1e-6 <= s <= 1 + 1e-6:
                ss.append(p1 + s * (p2 - p1))
        if len(ss) < 2:
            return None
        p, r = ss[0], ss[1]
        L = np.linalg.norm(r - p)
        k = max(2, int(L / DX))
        return p + np.linspace(0, 1, k)[:, None] * (r - p)

    lines, cov = [], np.zeros(n, bool)
    c = cmin
    while c < cmax - 1e-6:
        l = line(c)
        if l is None:
            break
        d = cdist(GRID, l).min(axis=1)
        cov |= (d <= Wth / 2)
        lines.append(l)
        idx = np.round(l[:, 1] / DX).astype(int) * NX + \
            np.round(l[:, 0] / DX).astype(int)
        c += 0.9 * np.percentile(Wth[idx], quantile * 100)
    return lines, cov


def repair_miss(lines, cov, theta_deg):
    """漏测连通块补线: 块中心沿主线方向θ补线"""
    from scipy import ndimage
    th = np.deg2rad(theta_deg)
    t = np.array([np.cos(th), np.sin(th)])
    for _ in range(6):
        cov2 = cov.reshape(NY, NX)
        lab, cnt = ndimage.label(~cov2, structure=np.ones((3, 3)))
        if cnt == 0:
            break
        m = 0
        for k in range(1, cnt + 1):
            ys, xs = np.where(lab == k)
            if len(ys) < 60:             # 微小块忽略
                continue
            pc = np.array([np.median(xs) * DX, np.median(ys) * DX])
            # 过pc沿θ的线与海域的交点
            ss = []
            for a, b in EDGES:
                p1, p2 = CORNERS[a], CORNERS[b]
                denom = t[0] * (p2[1] - p1[1]) - t[1] * (p2[0] - p1[0])
                if abs(denom) < 1e-9:
                    continue
                s = (t[0] * (p1[1] - pc[1]) - t[1] * (p1[0] - pc[0])) / denom
                if -1e-6 <= s <= 1 + 1e-6:
                    ss.append(p1 + s * (p2 - p1))
            if len(ss) < 2:
                continue
            p, r = ss[0], ss[1]
            L = np.linalg.norm(r - p)
            kk = max(2, int(L / DX))
            line = p + np.linspace(0, 1, kk)[:, None] * (r - p)
            lines.append(line)
            d = cdist(GRID, line).min(axis=1)
            cov |= (d <= Wr / 2)
            m += 1
        if m == 0:
            break
    return lines, cov


def metrics(lines, cov, theta_deg):
    """指标: 总长/漏测率/超20%重叠长度(逐点口径)"""
    th = np.deg2rad(theta_deg)
    nv = np.array([-np.sin(th), np.cos(th)])
    Wth = w_field(nv[0], nv[1]).ravel()

    def idx(pts):
        return np.round(pts[:, 1] / DX).astype(int) * NX + \
            np.round(pts[:, 0] / DX).astype(int)

    L = sum(np.linalg.norm(np.diff(l, axis=0), axis=1).sum() for l in lines)
    miss = 1 - cov.mean()
    over = 0.0
    for l in lines:
        O = np.concatenate([m for m in lines if m is not l], axis=0)
        dd = cdist(l, O)
        d = dd.min(axis=1)
        w_adj = Wth[idx(O[dd.argmin(axis=1)])]
        w_self = Wth[idx(l)]
        eta = 1 - d / np.minimum(w_self, w_adj)
        frac = (eta > 0.2).mean()
        over += frac * np.linalg.norm(np.diff(l, axis=0), axis=1).sum()
    return L / NM, miss, over / NM, len(lines)


def scan_direction():
    """方向扫描(漏测率最小者最优)"""
    best = (1e9, None)
    for th in range(85, 126, 5):
        _, cov = solve_lines(th)
        if 1 - cov.mean() < best[0]:
            best = (1 - cov.mean(), th)
    return best[1]


# ===== 对比算法: 惩罚加权最短路径(PWS)迭代Dijkstra =====
# 8邻域图, 边权=方向惩罚(贴合等深线)+覆盖惩罚(碗形), 迭代至覆盖饱和
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

THETA_ISO = (np.arctan2(gy, gx) + np.pi / 2) % np.pi   # 等深线方向
K_DIR, P_COV = 3.0, 6000.0
rows, cols, wdir = [], [], []
for i in range(NY):
    for j in range(NX):
        u = i * NX + j
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < NY and 0 <= jj < NX:
                v = ii * NX + jj
                ell = np.hypot(di * DX, dj * DX)
                te = np.arctan2(di, dj)
                da = np.abs((te - THETA_ISO[(i + ii) // 2, (j + jj) // 2])
                            % np.pi)
                da = min(da, np.pi - da)
                w = ell * (1 + K_DIR * np.sin(da) ** 2)
                rows += [u, v]
                cols += [v, u]
                wdir += [w, w]
src, dst = n, n + 1
for j in range(NX):
    rows += [src, j, dst, (NY - 1) * NX + j]
    cols += [j, src, (NY - 1) * NX + j, dst]
    wdir += [0.0, 0.0, 0.0, 0.0]
rows, cols, wdir = np.array(rows), np.array(cols), np.array(wdir, float)

# 纯长度边权图(覆盖引导版PWS用, 无方向惩罚)
rows_l, cols_l, wlen = [], [], []
for i in range(NY):
    for j in range(NX):
        u = i * NX + j
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < NY and 0 <= jj < NX:
                ell = np.hypot(di * DX, dj * DX)
                rows_l += [u, ii * NX + jj]
                cols_l += [ii * NX + jj, u]
                wlen += [ell, ell]
for j in range(NX):
    rows_l += [src, j, dst, (NY - 1) * NX + j]
    cols_l += [j, src, (NY - 1) * NX + j, dst]
    wlen += [0.0, 0.0, 0.0, 0.0]
rows_l, cols_l, wlen = np.array(rows_l), np.array(cols_l), np.array(wlen, float)


def repair_dijkstra(lines, cov):
    """漏测块内Dijkstra补线: 禁入已覆盖区, 方向惩罚贴合局部地形
    对每个漏测连通块, 从块对应南边界区间到北边界区间找最短路径"""
    from scipy import ndimage
    for _ in range(6):
        cov2 = cov.reshape(NY, NX)
        lab, cnt = ndimage.label(~cov2, structure=np.ones((3, 3)))
        if cnt == 0:
            break
        m = 0
        for k in range(1, cnt + 1):
            ys, xs = np.where(lab == k)
            if len(ys) < 60:             # 微小块忽略
                continue
            jmin, jmax = int(xs.min()), int(xs.max())
            C = np.zeros(n + 2)
            C[:n] = 1e9 * cov.ravel()    # 已覆盖区禁入
            rows2 = list(rows) + [src] * (jmax - jmin + 1) \
                + [dst] * (jmax - jmin + 1)
            cols2 = list(cols) + list(range(jmin, jmax + 1)) \
                + [(NY - 1) * NX + j for j in range(jmin, jmax + 1)]
            w2 = np.concatenate([wdir, np.zeros(2 * (jmax - jmin + 1))])
            w2 = w2 + C[cols2]
            M = csr_matrix((w2, (rows2, cols2)), shape=(n + 2, n + 2))
            _, pred = sp_dijkstra(M, directed=False, indices=src,
                                  return_predecessors=True)
            path, u = [], dst
            while u != src and u != -1:
                path.append(u)
                u = pred[u]
            path = [v for v in path[::-1] if v < n]
            if not path:
                continue
            line = np.stack([GRID[v] for v in path])
            lines.append(line)
            d = cdist(GRID, line).min(axis=1)
            cov |= (d <= Wr / 2)
            m += 1
        if m == 0:
            break
    return lines, cov


def idx_fn(pts):
    """坐标→网格索引"""
    return np.round(pts[:, 1] / DX).astype(int) * NX + \
        np.round(pts[:, 0] / DX).astype(int)


def smooth_line(pts, C):
    """移动平均平滑: 抹平8邻域锯齿, 平滑点进入更深重叠区则回退"""
    pts = pts.copy()
    m = len(pts)
    for _ in range(3):
        new = pts.copy()
        for k in range(1, m - 1):
            cand = pts[max(0, k - 2):k + 3].mean(axis=0)
            ic = idx_fn(cand[None, :])[0]
            ik = idx_fn(pts[k:k + 1])[0]
            if C[ic] <= C[ik] + 1e-9:        # 惩罚不增: 不深入重叠区
                new[k] = cand
        pts = new
    return pts


def solve_pws_smooth(cov_target=0.98, max_iter=90):
    """PWS-平滑: 迭代Dijkstra + 移动平均平滑(曲率约束), 线更直间距更匀"""
    C = np.zeros(n + 2)
    cov = np.zeros(n, bool)
    lines = []
    for _ in range(max_iter):
        w = wdir + C[cols]
        M = csr_matrix((w, (rows, cols)), shape=(n + 2, n + 2))
        _, pred = sp_dijkstra(M, directed=False, indices=src,
                              return_predecessors=True)
        path, u = [], dst
        while u != src and u != -1:
            path.append(u)
            u = pred[u]
        path = [v for v in path[::-1] if v < n]
        pts = np.stack([GRID[v] for v in path])
        pts = smooth_line(pts, C)              # 平滑: 减少蛇形
        d = cdist(GRID, pts).min(axis=1)
        cov |= (d <= Wr / 2)
        lines.append(pts)
        C[:n] += P_COV * np.clip(0.9 * Wr - d, 0, None) / Wr
        if cov.mean() >= cov_target:
            break
    return lines, cov


def solve_pws(cov_target=0.98, max_iter=90):
    """PWS: 迭代Dijkstra, 覆盖碗形惩罚(间距落在0.9W)"""
    C = np.zeros(n + 2)
    cov = np.zeros(n, bool)
    lines = []
    for _ in range(max_iter):
        w = wdir + C[cols]
        M = csr_matrix((w, (rows, cols)), shape=(n + 2, n + 2))
        _, pred = sp_dijkstra(M, directed=False, indices=src,
                              return_predecessors=True)
        path, u = [], dst
        while u != src and u != -1:
            path.append(u)
            u = pred[u]
        path = [v for v in path[::-1] if v < n]
        pts = np.stack([GRID[v] for v in path])
        d = cdist(GRID, pts).min(axis=1)
        cov |= (d <= Wr / 2)
        lines.append(pts)
        C[:n] += P_COV * np.clip(0.9 * Wr - d, 0, None) / Wr
        if cov.mean() >= cov_target:
            break
    return lines, cov


def solve_pws_guided(LAM=300.0, cov_target=0.985, max_iter=90, min_new=200):
    """PWS-覆盖引导: 边权=长度+重叠惩罚+已覆盖区禁入(LAM),
    新线被迫走未覆盖走廊中心, 间距由覆盖惩罚自适应"""
    cov = np.zeros(n, bool)
    cov_ext = np.zeros(n + 2, bool)
    C = np.zeros(n + 2)
    lines = []
    for _ in range(max_iter):
        cov_ext[:n] = cov
        w = wlen + C[cols] + LAM * cov_ext[cols]
        M = csr_matrix((w, (rows, cols)), shape=(n + 2, n + 2))
        _, pred = sp_dijkstra(M, directed=False, indices=src,
                              return_predecessors=True)
        path, u = [], dst
        while u != src and u != -1:
            path.append(u)
            u = pred[u]
        path = [v for v in path[::-1] if v < n]
        pts = np.stack([GRID[v] for v in path])
        d = cdist(GRID, pts).min(axis=1)
        new = (d <= Wr / 2) & ~cov
        cov |= (d <= Wr / 2)
        lines.append(pts)
        C[:n] += P_COV * np.clip(0.9 * Wr - d, 0, None) / Wr
        if cov.mean() >= cov_target or new.sum() < min_new:
            break
    return lines, cov


if __name__ == '__main__':
    th = scan_direction()
    print(f'[扫描] 最优测线方向: {th}°')
    lines, cov = solve_lines(th)
    lines, cov = repair_dijkstra(lines, cov)
    m1 = metrics(lines, cov, th)
    l_pws, c_pws = solve_pws()
    m2 = metrics(l_pws, c_pws, 90)
    l_pwss, c_pwss = solve_pws_smooth()
    m3 = metrics(l_pwss, c_pwss, 90)
    l_pwsg, c_pwsg = solve_pws_guided()
    m4 = metrics(l_pwsg, c_pwsg, 90)

    print('=' * 58)
    print(f'{"算法":<14}{"条数":>5}{"总长/海里":>10}{"漏测率":>9}'
          f'{"超20%长度/海里":>14}')
    print('-' * 58)
    print(f'贪心@{th}°(主方案){m1[3]:>5}{m1[0]:>10.2f}{m1[1]*100:>8.2f}%'
          f'{m1[2]:>14.2f}')
    print(f'PWS-贴合等深线  {m2[3]:>5}{m2[0]:>10.2f}{m2[1]*100:>8.2f}%'
          f'{m2[2]:>14.2f}')
    print(f'PWS-平滑(曲率)  {m3[3]:>5}{m3[0]:>10.2f}{m3[1]*100:>8.2f}%'
          f'{m3[2]:>14.2f}')
    print(f'PWS-覆盖引导    {m4[3]:>5}{m4[0]:>10.2f}{m4[1]*100:>8.2f}%'
          f'{m4[2]:>14.2f}')
    print('=' * 58)
