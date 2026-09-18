# ============================================================
# 有向网络推断 + SCNG-RC 预测
#   核心区别：不做对称化，保持有向
#   评估基准：真实的有向 W（原始 ww1）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import time
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from zxgj1.validpt import Vpt
from zxgj1.工具 import f变量变形

vpt = Vpt.vpt3
np.random.seed(42)

# ============================================================
# 1. 加载数据
# ============================================================
xx = np.load('data/llzc7.npz')
ww1, yy = xx['arr_0'], xx['arr_1'][::2]

xx1 = yy[1000:15000]
xx2 = yy[15000:20000]
suo = np.ceil(np.max(xx1) + 1)
xh1 = xx1 / suo
xh2 = xx2 / suo

N = ww1.shape[0]
L_state = 3
xh1gai = f变量变形(xh1)

# ★ 真实网络：保持有向
W_true_directed = ww1.astype(int)
n_edges_true_directed = int(W_true_directed.sum())
print(f'True directed edges: {n_edges_true_directed}')

# 真实无向边（用于参考）
W_sym = np.maximum(ww1, ww1.T).astype(int)
np.fill_diagonal(W_sym, 0)
n_edges_true_sym = int(W_sym.sum() / 2)
print(f'True undirected edges: {n_edges_true_sym}')

# ============================================================
# 2. 全局参数
# ============================================================
dLt = 3
dt = 1
start = 80
qstart = 100
lam_infer = 1e-3
n_pred = 1000


# ============================================================
# 3. 特征构造 + 训练 + 预测
# ============================================================
def build_features(xh1gai, cc, grla, start, qstart, n_pred):
    Sm = len(cc)
    l1 = xh1gai.shape[1]
    线性项个数 = dLt * 3

    总项个数 = []
    for ii in range(Sm):
        n_lin_self = 线性项个数
        n_lin_neigh = len(cc[ii]) * 线性项个数
        n_quad_self = int((线性项个数 + 1) * 线性项个数 / 2)
        n_cross = 线性项个数 * len(cc[ii]) * 线性项个数
        总项个数.append(n_lin_self + n_lin_neigh + n_quad_self + n_cross + 1)

    RR = [np.zeros((l1, 总项个数[ii])) for ii in range(Sm)]
    for zz in range(Sm):
        olin = np.c_[xh1gai[zz][start:, :],
        xh1gai[zz][start - dt:-dt, :],
        xh1gai[zz][start - 2 * dt:-2 * dt, :]]
        RR[zz][start:, :线性项个数] = olin

    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc[zz]:
            RR[zz][:, xu:xu + 线性项个数] = RR[ij][:, :线性项个数]
            xu += 线性项个数
        for i1 in range(线性项个数):
            for i2 in range(i1, 线性项个数):
                RR[zz][start:, xu] = (RR[zz][start:, :线性项个数][:, i1] *
                                      RR[zz][start:, :线性项个数][:, i2])
                xu += 1
        for i1 in range(线性项个数):
            for ij in cc[zz]:
                for jj in range(线性项个数):
                    RR[zz][:, xu] = RR[zz][:, i1] * RR[ij][:, jj]
                    xu += 1
        RR[zz][:, xu] = 1

    wout = []
    for ii in range(Sm):
        X = RR[ii][qstart:-1]
        Y = xh1gai[ii][qstart + 1:]
        F = X.shape[1]
        A_mat = X.T @ X + grla * np.eye(F)
        wout.append(np.linalg.solve(A_mat, X.T @ Y))

    rrzhong = [RR[ii][-1].copy() for ii in range(Sm)]
    yc = np.zeros((Sm, 100 + n_pred, 3))
    yc[:, :100] = xh1gai[:, -100:]

    for ii in range(100, 100 + n_pred):
        for zz in range(Sm):
            yc[zz][ii] = rrzhong[zz] @ wout[zz]
        for zz in range(Sm):
            olin = np.r_[yc[zz][ii, :], yc[zz][ii - dt, :], yc[zz][ii - 2 * dt, :]]
            rrzhong[zz][:线性项个数] = olin
        for zz in range(Sm):
            xu = 线性项个数
            for ij in cc[zz]:
                rrzhong[zz][xu:xu + 线性项个数] = rrzhong[ij][:线性项个数]
                xu += 线性项个数
            for i1 in range(线性项个数):
                for i2 in range(i1, 线性项个数):
                    rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[zz][i2]
                    xu += 1
            for i1 in range(线性项个数):
                for ij in cc[zz]:
                    for jj in range(线性项个数):
                        rrzhong[zz][xu] = rrzhong[zz][i1] * rrzhong[ij][jj]
                        xu += 1

    return yc, 总项个数[0]


# ============================================================
# 4. Stage 1: 格兰杰因果推断
# ============================================================
def compute_train_error(xh1gai, target, neighbors, grla, start, qstart):
    l1 = xh1gai.shape[1]
    线性项个数 = dLt * 3

    Lin = np.zeros((l1, 线性项个数 * (1 + len(neighbors))))
    Lin[start:, :线性项个数] = np.c_[
        xh1gai[target][start:, :],
        xh1gai[target][start - dt:-dt, :],
        xh1gai[target][start - 2 * dt:-2 * dt, :]
    ]
    for k, ij in enumerate(neighbors):
        base = (k + 1) * 线性项个数
        Lin[start:, base:base + 线性项个数] = np.c_[
            xh1gai[ij][start:, :],
            xh1gai[ij][start - dt:-dt, :],
            xh1gai[ij][start - 2 * dt:-2 * dt, :]
        ]

    n_quad = int((线性项个数 + 1) * 线性项个数 / 2)
    quad = np.zeros((l1, n_quad))
    xu = 0
    for i1 in range(线性项个数):
        for i2 in range(i1, 线性项个数):
            quad[start:, xu] = Lin[start:, i1] * Lin[start:, i2]
            xu += 1

    n_cross = 线性项个数 * len(neighbors) * 线性项个数
    cross = np.zeros((l1, n_cross))
    xu = 0
    for i1 in range(线性项个数):
        for k in range(len(neighbors)):
            for jj in range(线性项个数):
                cross[:, xu] = Lin[:, i1] * Lin[:, (k + 1) * 线性项个数 + jj]
                xu += 1

    RR = np.hstack([Lin, quad, cross, np.ones((l1, 1))])

    X = RR[qstart:-1]
    Y = xh1gai[target][qstart + 1:]
    F = X.shape[1]
    A_mat = X.T @ X + grla * np.eye(F)
    w = np.linalg.solve(A_mat, X.T @ Y)
    pred = X @ w
    return np.mean((Y - pred) ** 2)


print('\n' + '=' * 60)
print('Stage 1: 格兰杰因果推断')
print('=' * 60)

t0 = time.time()
E_self = np.zeros(N)
E_pair = np.zeros((N, N))

for i in range(N):
    E_self[i] = compute_train_error(xh1gai, i, [], lam_infer, start, qstart)
    for j in range(N):
        if j == i:
            continue
        E_pair[i, j] = compute_train_error(xh1gai, i, [j], lam_infer,
                                           start, qstart)
    print(f'  node {i}: E_self={E_self[i]:.4e}, '
          f'E_min(≠i)={E_pair[i][np.arange(N) != i].min():.4e}')

print(f'推断用时: {time.time() - t0:.1f}s')

gain = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if j != i:
            gain[i, j] = (E_self[i] - E_pair[i, j]) / E_self[i]

print('\nGain 矩阵：')
print(np.round(gain, 4))


# ============================================================
# 5. ★ 有向推断：不做对称化
# ============================================================
def infer_ratio_directed(gain, ratio):
    """有向推断：每行独立阈值化，不进行对称化"""
    A = np.zeros((N, N))
    for i in range(N):
        row = gain[i].copy()
        row[i] = -np.inf
        tau = ratio * row.max()
        for j in range(N):
            if j != i and row[j] > tau:
                A[i, j] = 1  # i 的入邻居包括 j
    np.fill_diagonal(A, 0)
    return A


def evaluate_directed(A_true, A_infer):
    """有向网络评估：直接逐元素比较"""
    tp = int(((A_infer == 1) & (A_true == 1)).sum())
    fp = int(((A_infer == 1) & (A_true == 0)).sum())
    fn = int(((A_infer == 0) & (A_true == 1)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    return P, R, F1, tp, fp, fn


def evaluate_undirected(A_true_sym, A_infer):
    """无向网络评估：对称化后比较"""
    A_sym = np.maximum(A_infer, A_infer.T).astype(int)
    np.fill_diagonal(A_sym, 0)
    tp = int(((A_sym == 1) & (A_true_sym == 1)).sum() / 2)
    fp = int(((A_sym == 1) & (A_true_sym == 0)).sum() / 2)
    fn = int(((A_sym == 0) & (A_true_sym == 1)).sum() / 2)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    return P, R, F1, tp, fp, fn


ratios = [0.1, 0.05, 0.01]

print('\n' + '=' * 80)
print('有向推断结果（与真实有向 W 比较）')
print('=' * 80)
print(f'{"ratio":>8s} | {"边数":>5s} | {"TP":>3s} | {"FP":>3s} | '
      f'{"FN":>3s} | {"P":>5s} | {"R":>5s} | {"F1":>5s}')
print('-' * 80)

infer_results = {}
for ratio in ratios:
    A = infer_ratio_directed(gain, ratio)
    P, R, F1, tp, fp, fn = evaluate_directed(W_true_directed, A)
    n_edges = int(A.sum())

    # 同时用无向评估基准看一下
    A_sym = np.maximum(A, A.T).astype(int)
    np.fill_diagonal(A_sym, 0)
    P_u, R_u, F1_u, tp_u, fp_u, fn_u = evaluate_undirected(W_sym, A)

    cc_use = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                cc_use[i].append(j)

    infer_results[ratio] = {
        'A_directed': A,
        'A_sym': A_sym,
        'cc': cc_use,
        'P_directed': P, 'R_directed': R, 'F1_directed': F1,
        'P_undirected': P_u, 'R_undirected': R_u, 'F1_undirected': F1_u,
        'n_edges_directed': n_edges,
        'n_edges_undirected': int(A_sym.sum() / 2),
        'tp': tp, 'fp': fp, 'fn': fn,
    }
    print(f'{ratio:>8.3f} | {n_edges:>5d} | {tp:>3d} | {fp:>3d} | '
          f'{fn:>3d} | {P:.3f} | {R:.3f} | {F1:.3f}')

print('\n参考：无向评估基准下（对称化后）的 F1')
print(f'{"ratio":>8s} | {"无向边数":>8s} | {"P":>5s} | {"R":>5s} | {"F1":>5s}')
print('-' * 50)
for ratio in ratios:
    r = infer_results[ratio]
    print(f'{ratio:>8.3f} | {r["n_edges_undirected"]:>8d} | '
          f'{r["P_undirected"]:.3f} | {r["R_undirected"]:.3f} | '
          f'{r["F1_undirected"]:.3f}')

# ============================================================
# 6. Stage 2: 各推断网络 λ 扫描
# ============================================================
lambdas = np.logspace(-8, -5, 9)
print(f'\nλ 网格：{np.round(lambdas, 12)}')

print('\n' + '=' * 80)
print('各推断网络 λ 扫描 → 最优 VPS')
print('=' * 80)

pred_results = {}
for ratio in ratios:
    cc_use = infer_results[ratio]['cc']
    vpts = np.zeros(len(lambdas))
    n_feat = None

    for k, lam in enumerate(lambdas):
        yc, n_feat = build_features(xh1gai, cc_use, lam, start,
                                    qstart, n_pred)
        yct = f变量变形(yc)
        vpts[k] = vpt(yct[100:], xh2, 0.5)

    best_idx = np.argmax(vpts)
    best_lam = lambdas[best_idx]
    best_vpt = vpts[best_idx]

    pred_results[ratio] = {
        'vpt': best_vpt, 'best_lam': best_lam,
        'vpts': vpts, 'n_feat': n_feat,
    }
    print(f'ρ={ratio:.3f} | λ*={best_lam:.2e} | VPS={int(best_vpt):4d} | '
          f'features={n_feat:5d}')

# ============================================================
# 7. 真网络参考
# ============================================================
cc_true = [[] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if ww1[i, j] > 0.01:
            cc_true[i].append(j)

vpts = np.zeros(len(lambdas))
n_feat = None
for k, lam in enumerate(lambdas):
    yc, n_feat = build_features(xh1gai, cc_true, lam, start,
                                qstart, n_pred)
    yct = f变量变形(yc)
    vpts[k] = vpt(yct[100:], xh2, 0.5)

best_idx = np.argmax(vpts)
best_lam = lambdas[best_idx]
best_vpt = vpts[best_idx]

ref_results = {
    'True_directed': {
        'vpt': best_vpt, 'best_lam': best_lam,
        'n_feat': n_feat, 'n_edges': n_edges_true_directed,
        'vpts': vpts,
    }
}
print(f'\nTrue (directed) | edges={n_edges_true_directed:3d} | '
      f'λ*={best_lam:.2e} | VPS={int(best_vpt):5d} | '
      f'features={n_feat:5d}')


# ============================================================
# 8. 彩色拓扑图（有向）
# ============================================================
def color_code_directed(A_true, A_infer):
    M = np.zeros_like(A_true, dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t = A_true[i, j]
            p = A_infer[i, j]
            if t == 1 and p == 1:
                M[i, j] = 1  # TP
            elif t == 0 and p == 1:
                M[i, j] = 2  # FP
            elif t == 1 and p == 0:
                M[i, j] = 3  # FN
    return M


cmap_edges = mcolors.ListedColormap(['white', '#2ca02c', '#d62728', '#cccccc'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_edges = mcolors.BoundaryNorm(bounds, cmap_edges.N)

n_plots = 1 + len(ratios)
fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3.2), dpi=300)

axs[0].imshow(W_true_directed, cmap='Blues', vmin=0, vmax=1)
axs[0].set_title(f'(a) True (directed)\n{n_edges_true_directed} edges',
                 fontsize=9)
axs[0].set_xticks(range(N));
axs[0].set_yticks(range(N))
axs[0].tick_params(labelsize=6)

for k, ratio in enumerate(ratios):
    A = infer_results[ratio]['A_directed']
    M = color_code_directed(W_true_directed, A)
    ax = axs[k + 1]
    ax.imshow(M, cmap=cmap_edges, norm=norm_edges)
    ax.set_title(f'({chr(98 + k)}) $\\rho$={ratio}\n'
                 f'{infer_results[ratio]["n_edges_directed"]} edges, '
                 f'F1={infer_results[ratio]["F1_directed"]:.2f}', fontsize=8)
    ax.set_xticks(range(N));
    ax.set_yticks(range(N))
    ax.tick_params(labelsize=6)

legend_elements = [
    Patch(facecolor='#2ca02c', label='True Positive (TP)'),
    Patch(facecolor='#d62728', label='False Positive (FP)'),
    Patch(facecolor='#cccccc', label='False Negative (FN)'),
    Patch(facecolor='white', edgecolor='black', label='True Negative (TN)'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Directed adjacency matrices under different inference thresholds',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('7infer_directed.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# 9. VPS-λ 曲线
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

markers = ['o', 's', '^']
colors = ['b', 'g', 'r']
for (r, mk, c) in zip(ratios, markers, colors):
    ax.semilogx(lambdas, pred_results[r]['vpts'], mk + '-', color=c,
                label=f'ρ={r} (F1={infer_results[r]["F1_directed"]:.2f})',
                markersize=5, lw=1.2)

ax.semilogx(lambdas, ref_results['True_directed']['vpts'], 'k--',
            label='True (directed)', lw=1.2)

ax.set_xlabel(r'Regularization parameter $\lambda$')
ax.set_ylabel('Valid Prediction Steps (VPS)')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('7infer_vpt_directed.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# 10. 汇总
# ============================================================
print('\n' + '=' * 100)
print('汇总（有向推断）')
print('=' * 100)
print(f'{"方法":<22s} | {"有向边数":>8s} | {"TP":>3s} | {"FP":>3s} | '
      f'{"FN":>3s} | {"F1(有向)":>8s} | {"F1(无向)":>8s} | {"VPS":>5s}')
print('-' * 100)

print(f'{"True (directed)":<22s} | {n_edges_true_directed:>8d} | '
      f'{"—":>3s} | {"—":>3s} | {"—":>3s} | {"1.000":>8s} | '
      f'{"1.000":>8s} | {int(ref_results["True_directed"]["vpt"]):>5d}')

for ratio in ratios:
    r = infer_results[ratio]
    print(f'{"Inferred ρ=" + str(ratio):<22s} | {r["n_edges_directed"]:>8d} | '
          f'{r["tp"]:>3d} | {r["fp"]:>3d} | {r["fn"]:>3d} | '
          f'{r["F1_directed"]:>8.3f} | {r["F1_undirected"]:>8.3f} | '
          f'{int(pred_results[ratio]["vpt"]):>5d}')