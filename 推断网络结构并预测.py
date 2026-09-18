# ============================================================
# 格兰杰因果推断 + 不同假阳性数量的 VPT 对比
#   阈值：ρ = 0.1, 0.05, 0.01
#   拓扑图用彩色：TP=绿, FP=红, FN=灰, 真边=蓝
#   每个网络单独做 λ 扫描，报告最优 VPT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import itertools
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
print(f'N = {N}, T_train = {xh1gai.shape[1]}, L_state = {L_state}')

# 真实无向边
W_sym = np.maximum(ww1, ww1.T).astype(int)
np.fill_diagonal(W_sym, 0)
n_edges_true = int(W_sym.sum() / 2)
print(f'True undirected edges: {n_edges_true}')
print(f'Degrees: {W_sym.sum(axis=1)}')

true_edges = set()
for i in range(N):
    for j in range(i+1, N):
        if W_sym[i, j] == 1:
            true_edges.add((i, j))
print(f'True edges: {sorted(true_edges)}')

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
                     xh1gai[zz][start-dt:-dt, :],
                     xh1gai[zz][start-2*dt:-2*dt, :]]
        RR[zz][start:, :线性项个数] = olin

    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc[zz]:
            RR[zz][:, xu:xu+线性项个数] = RR[ij][:, :线性项个数]
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
        Y = xh1gai[ii][qstart+1:]
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
            olin = np.r_[yc[zz][ii, :], yc[zz][ii-dt, :], yc[zz][ii-2*dt, :]]
            rrzhong[zz][:线性项个数] = olin
        for zz in range(Sm):
            xu = 线性项个数
            for ij in cc[zz]:
                rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
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
        xh1gai[target][start-dt:-dt, :],
        xh1gai[target][start-2*dt:-2*dt, :]
    ]
    for k, ij in enumerate(neighbors):
        base = (k + 1) * 线性项个数
        Lin[start:, base:base+线性项个数] = np.c_[
            xh1gai[ij][start:, :],
            xh1gai[ij][start-dt:-dt, :],
            xh1gai[ij][start-2*dt:-2*dt, :]
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
                cross[:, xu] = Lin[:, i1] * Lin[:, (k+1)*线性项个数 + jj]
                xu += 1

    RR = np.hstack([Lin, quad, cross, np.ones((l1, 1))])

    X = RR[qstart:-1]
    Y = xh1gai[target][qstart+1:]
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
          f'E_min(≠i)={E_pair[i][np.arange(N)!=i].min():.4e}')

print(f'推断用时: {time.time()-t0:.1f}s')

gain = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if j != i:
            gain[i, j] = (E_self[i] - E_pair[i, j]) / E_self[i]

print('\nGain 矩阵：')
print(np.round(gain, 4))


# ============================================================
# 5. 不同 ratio → 不同假阳性数量
# ============================================================
def infer_ratio(gain, ratio):
    A = np.zeros((N, N))
    for i in range(N):
        row = gain[i].copy()
        row[i] = -np.inf
        tau = ratio * row.max()
        for j in range(N):
            if j != i and row[j] > tau:
                A[i, j] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def evaluate(A_true, A_infer):
    tp = int(((A_infer == 1) & (A_true == 1)).sum() / 2)
    fp = int(((A_infer == 1) & (A_true == 0)).sum() / 2)
    fn = int(((A_infer == 0) & (A_true == 1)).sum() / 2)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    return P, R, F1, tp, fp, fn


# ★ 只保留 3 个阈值
ratios = [0.1, 0.05, 0.01]

print('\n' + '=' * 80)
print('不同 ratio 下的推断质量')
print('=' * 80)
print(f'{"ratio":>8s} | {"边数":>5s} | {"TP":>3s} | {"FP":>3s} | '
      f'{"FN":>3s} | {"P":>5s} | {"R":>5s} | {"F1":>5s}')
print('-' * 80)

infer_results = {}
for ratio in ratios:
    A = infer_ratio(gain, ratio)
    A_sym = np.maximum(A, A.T).astype(int)
    np.fill_diagonal(A_sym, 0)

    P, R, F1, tp, fp, fn = evaluate(W_sym, A_sym)
    n_edges = int(A_sym.sum() / 2)

    cc_use = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A_sym[i, j] == 1:
                cc_use[i].append(j)

    infer_results[ratio] = {
        'A': A_sym, 'cc': cc_use,
        'P': P, 'R': R, 'F1': F1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_edges': n_edges,
    }
    print(f'{ratio:>8.3f} | {n_edges:>5d} | {tp:>3d} | {fp:>3d} | '
          f'{fn:>3d} | {P:.3f} | {R:.3f} | {F1:.3f}')


# ============================================================
# 6. Stage 2: 每个网络 λ 扫描找最优
# ============================================================
lambdas = np.logspace(-8, -5, 9)
print(f'\nλ 网格：{np.round(lambdas, 12)}')

print('\n' + '=' * 80)
print('各推断网络 λ 扫描 → 最优 VPT')
print('=' * 80)

pred_results = {}
for ratio in ratios:
    cc_use = infer_results[ratio]['cc']
    vpts = np.zeros(len(lambdas))
    n_feat = None

    for k, lam in enumerate(lambdas):
        t0 = time.time()
        yc, n_feat = build_features(xh1gai, cc_use, lam, start,
                                    qstart, n_pred)
        yct = f变量变形(yc)
        vpt_val = vpt(yct[100:], xh2, 0.5)
        vpts[k] = vpt_val
        print(f'  ρ={ratio:.3f} | λ={lam:.2e} | VPT={int(vpt_val):4d} | '
              f'time={time.time() - t0:.1f}s')

    best_idx = np.argmax(vpts)
    best_lam = lambdas[best_idx]
    best_vpt = vpts[best_idx]

    pred_results[ratio] = {
        'vpt': best_vpt, 'best_lam': best_lam,
        'vpts': vpts, 'n_feat': n_feat,
    }
    print(f'  → 最优 λ={best_lam:.2e}, VPT={int(best_vpt)}\n')


# ============================================================
# 7. 真网络参考（λ 扫描）
# ============================================================
cc_true = []
for ii in range(N):
    cc_true.append([jj for jj in range(N) if ww1[ii, jj] > 0.01])

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
    'True': {
        'vpt': best_vpt, 'best_lam': best_lam,
        'n_feat': n_feat, 'n_edges': n_edges_true,
        'vpts': vpts,
    }
}
print(f'True | edges={n_edges_true:3d} | λ*={best_lam:.2e} | '
      f'VPT={int(best_vpt):5d} | features={n_feat:5d}')


# ============================================================
# 8. ★ 彩色拓扑图（真网络 + 3 个 ratio）
# ============================================================
def color_code(A_true, A_infer):
    """颜色编码：
       0 = TN（白）, 1 = TP（绿）, 2 = FP（红）, 3 = FN（灰）
    """
    M = np.zeros_like(A_true, dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t = A_true[i, j]
            p = A_infer[i, j]
            if t == 1 and p == 1:
                M[i, j] = 1
            elif t == 0 and p == 1:
                M[i, j] = 2
            elif t == 1 and p == 0:
                M[i, j] = 3
    return M


cmap_edges = mcolors.ListedColormap(['white', '#2ca02c', '#d62728', '#cccccc'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_edges = mcolors.BoundaryNorm(bounds, cmap_edges.N)

n_plots = 1 + len(ratios)   # 真网络 + 3 个 ratio = 4
fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3.2), dpi=300)

# (a) 真网络
axs[0].imshow(W_sym, cmap='Blues', vmin=0, vmax=1)
axs[0].set_title(f'(a) True\n{n_edges_true} edges', fontsize=9)
axs[0].set_xticks(range(N)); axs[0].set_yticks(range(N))
axs[0].tick_params(labelsize=6)

# (b-d) 3 个 ratio
for k, ratio in enumerate(ratios):
    A_sym = infer_results[ratio]['A']
    M = color_code(W_sym, A_sym)
    ax = axs[k + 1]
    ax.imshow(M, cmap=cmap_edges, norm=norm_edges)
    ax.set_title(f'({chr(98+k)}) $\\rho$={ratio}\n'
                 f'{infer_results[ratio]["n_edges"]} edges, '
                 f'FP={infer_results[ratio]["fp"]}, '
                 f'F1={infer_results[ratio]["F1"]:.2f}', fontsize=8)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.tick_params(labelsize=6)

# 图例
legend_elements = [
    Patch(facecolor='#2ca02c', label='True Positive (TP)'),
    Patch(facecolor='#d62728', label='False Positive (FP)'),
    Patch(facecolor='#cccccc', label='False Negative (FN)'),
    Patch(facecolor='white', edgecolor='black', label='True Negative (TN)'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Adjacency matrices under different inference thresholds',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('7infer.pdf', bbox_inches='tight')
plt.show()


# ============================================================
# 9. ★ VPS-λ 曲线（3 个推断网络 + 真网络参考）
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

markers = ['o', 's', '^']
colors = ['b', 'g', 'r']
for (r, mk, c) in zip(ratios, markers, colors):
    ax.semilogx(lambdas, pred_results[r]['vpts'], mk + '-', color=c,
                label=f'ρ={r} (F1={infer_results[r]["F1"]:.2f})',
                markersize=5, lw=1.2)

# 真网络参考线
ax.semilogx(lambdas, ref_results['True']['vpts'], 'k--',
            label=f"True (F1=1.00)", lw=1.2)

ax.set_xlabel(r'Regularization parameter $\lambda$')
ax.set_ylabel('Valid Prediction Steps (VPS)')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('7infer_vpt.pdf', bbox_inches='tight')
plt.show()


# ============================================================
# 10. ★ VPS vs 推断边数（核心论证图）
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

edges_arr = np.array([infer_results[r]['n_edges'] for r in ratios])
vpt_arr2 = np.array([pred_results[r]['vpt'] for r in ratios])

ax.plot(edges_arr, vpt_arr2, 'go-', markersize=10, lw=1.5,
        label='Inferred (optimal λ)')

ax.axhline(ref_results['True']['vpt'], color='k', ls='--', lw=1.2,
           label=f"True ({int(ref_results['True']['vpt'])})")

for e, v, r in zip(edges_arr, vpt_arr2, ratios):
    lam_star = pred_results[r]['best_lam']
    ax.annotate(f'ρ={r}\nλ*={lam_star:.0e}', (e, v),
                textcoords='offset points', xytext=(5, -18), fontsize=8)

ax.set_xlabel('Number of inferred edges')
ax.set_ylabel('VPS')
ax.set_title('VPS vs edges (each point uses its optimal λ)')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('7infer_vpt_vs_edges.pdf', bbox_inches='tight')
plt.show()


# ============================================================
# 11. 汇总
# ============================================================
print('\n' + '=' * 80)
print('汇总')
print('=' * 80)
print(f'{"方法":<22s} | {"边数":>5s} | {"FP":>3s} | {"F1":>6s} | '
      f'{"λ*":>10s} | {"VPS":>5s} | {"特征数":>6s}')
print('-' * 90)

print(f'{"True":<22s} | {n_edges_true:>5d} | {"—":>3s} | {"1.000":>6s} | '
      f'{ref_results["True"]["best_lam"]:>10.2e} | '
      f'{int(ref_results["True"]["vpt"]):>5d} | '
      f'{ref_results["True"]["n_feat"]:>6d}')

for ratio in ratios:
    name = f'Inferred ρ={ratio}'
    print(f'{name:<22s} | {infer_results[ratio]["n_edges"]:>5d} | '
          f'{infer_results[ratio]["fp"]:>3d} | '
          f'{infer_results[ratio]["F1"]:>6.3f} | '
          f'{pred_results[ratio]["best_lam"]:>10.2e} | '
          f'{int(pred_results[ratio]["vpt"]):>5d} | '
          f'{pred_results[ratio]["n_feat"]:>6d}')