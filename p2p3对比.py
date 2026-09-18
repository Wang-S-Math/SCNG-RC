#%%
import numpy as np
from numpy import zeros, ones_like
from numpy import tanh, eye, sin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import plot, figure, cla
import time
import itertools
rng = np.random.default_rng(42)
from zxgj1.validpt import Vpt
from zxgj1.工具 import f变量变形
from zxgj1.下一代储层计算 import c下一代储层计算
vpt = Vpt.vpt3
plt.rcParams.update({'font.size': 6.5})

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

l1, l2 = xh1.shape
Sm = int(l2 / 3)
xh1gai = f变量变形(xh1)
L_state = xh1gai.shape[2]
print(f'N = {Sm}, T = {l1}, L_state = {L_state}')

cc = []
for ii in range(Sm):
    zz = []
    for jj in range(Sm):
        if ww1[ii, jj] > 0.01:
            zz.append(jj)
    cc.append(zz)
print(f'Neighbor lists: {cc}')

# ============================================================
# 2. 通用特征生成函数
# ============================================================
dLt, dt, start, qstart, n = 3, 1, 80, 100, 1000
线性项个数 = dLt * 3        # = 9


def build_feature_indices(L, deg, poly_order):
    """生成特征项索引：高阶项至少含一个自身变量（索引 < L）"""
    m = L * (1 + deg)
    terms = [('lin', (i,)) for i in range(m)]   # 线性项
    for d in range(2, poly_order + 1):
        for comb in itertools.combinations_with_replacement(range(m), d):
            if any(i < L for i in comb):
                terms.append(('poly', comb))
    return terms


def build_and_train(xh1gai, cc, poly_order, grla, start, qstart, n):
    """构造特征 + 训练 + 预测
       返回：yc (Sm, 100+n, 3)、特征数
    """
    Sm = len(cc)
    l1 = xh1gai.shape[1]
    线性项个数 = dLt * 3

    features_list = []
    for ii in range(Sm):
        features_list.append(
            build_feature_indices(线性项个数, len(cc[ii]), poly_order)
        )

    总项个数 = zeros(Sm, dtype=int)
    for ii in range(Sm):
        总项个数[ii] = len(features_list[ii]) + 1

    # ---- 构造特征矩阵 RR ----
    RR = [zeros((l1, 总项个数[ii])) for ii in range(Sm)]

    # 自身线性项
    for zz in range(Sm):
        olin = np.c_[xh1gai[zz][start:, :],
                     xh1gai[zz][start-dt:-dt, :],
                     xh1gai[zz][start-2*dt:-2*dt, :]]
        RR[zz][start:, :线性项个数] = olin

    # 邻居线性项
    for zz in range(Sm):
        xu = 线性项个数
        for ij in cc[zz]:
            RR[zz][:, xu:xu+线性项个数] = RR[ij][:, :线性项个数]
            xu += 线性项个数

    # 高阶项
    for zz in range(Sm):
        m = 线性项个数 * (1 + len(cc[zz]))
        V = zeros((l1, m))
        V[:, :线性项个数] = RR[zz][:, :线性项个数]
        xu_v = 线性项个数
        for ij in cc[zz]:
            V[:, xu_v:xu_v+线性项个数] = RR[ij][:, :线性项个数]
            xu_v += 线性项个数

        xu = m
        for term_kind, term_idx in features_list[zz]:
            if term_kind == 'lin':
                continue
            RR[zz][start:, xu] = np.prod(
                [V[start:, i] for i in term_idx], axis=0
            )
            xu += 1
        RR[zz][:, xu] = 1.0

    # ---- 训练 ----
    wout = []
    A = []
    rrzhong = []
    for ii in range(Sm):
        qrr = RR[ii][qstart:-1]
        qxx = xh1gai[ii][qstart+1:]
        rrzhong.append(RR[ii][-1].copy())
        A.append(c下一代储层计算())
        wout.append(A[ii].qiujie(qxx, qrr, grla))

    # ---- 预测 ----
    yc = zeros((Sm, 100 + n, 3))
    yc[:, :100] = xh1gai[:, -100:]

    for ii in range(100, 100 + n):
        for zz in range(Sm):
            yc[zz][ii] = wout[zz] @ rrzhong[zz]

        for zz in range(Sm):
            olin = np.r_[yc[zz][ii, :],
                         yc[zz][ii-dt, :],
                         yc[zz][ii-2*dt, :]]
            rrzhong[zz][:线性项个数] = olin

        for zz in range(Sm):
            xu = 线性项个数
            for ij in cc[zz]:
                rrzhong[zz][xu:xu+线性项个数] = rrzhong[ij][:线性项个数]
                xu += 线性项个数

            m = 线性项个数 * (1 + len(cc[zz]))
            V = zeros(m)
            V[:线性项个数] = rrzhong[zz][:线性项个数]
            xu_v = 线性项个数
            for ij in cc[zz]:
                V[xu_v:xu_v+线性项个数] = rrzhong[ij][:线性项个数]
                xu_v += 线性项个数

            xu = m
            for term_kind, term_idx in features_list[zz]:
                if term_kind == 'lin':
                    continue
                rrzhong[zz][xu] = np.prod([V[i] for i in term_idx])
                xu += 1

    return yc, int(总项个数[0])


# ============================================================
# 3. λ 扫描：p=2 vs p=3
# ============================================================
lambdas = np.logspace(-8, -5, 20)
print(f'\nλ 网格：{np.round(lambdas, 12)}')

results = {}

for poly_order in [2, 3]:
    print(f'\n' + '=' * 60)
    print(f'SCNG-RC p={poly_order}: λ 扫描')
    print('=' * 60)

    vpts = np.zeros(len(lambdas))
    n_feat = None

    for k, lam in enumerate(lambdas):
        t0 = time.time()
        try:
            yc, n_feat = build_and_train(xh1gai, cc, poly_order, lam,
                                          start, qstart, n)
            yct = f变量变形(yc)
            vpt_val = vpt(yct[100:], xh2, 0.5)
        except Exception as e:
            print(f'  λ={lam:.2e} 出错：{type(e).__name__}: {e}')
            vpt_val = 0
        vpts[k] = vpt_val
        print(f'  λ={lam:.2e} | 特征={n_feat:6d} | VPT={int(vpt_val):4d} | '
              f'用时={time.time()-t0:.2f}s')

    best_idx = np.argmax(vpts)
    best_lam = lambdas[best_idx]
    best_vpt = vpts[best_idx]

    results[f'p={poly_order}'] = {
        'vpts': vpts,
        'best_lam': best_lam,
        'best_vpt': best_vpt,
        'n_feat': n_feat,
    }
    print(f'  → 最优 λ={best_lam:.2e}, VPT={int(best_vpt)}')

# ============================================================
# 4. VPT-λ 曲线
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
markers = ['o', 's']
for (name, res), mk in zip(results.items(), markers):
    ax.semilogx(lambdas, res['vpts'], mk + '-', label=name,
                markersize=5, lw=1.2)
ax.set_xlabel(r'Regularization parameter $\lambda$')
ax.set_ylabel('Valid prediction steps (VPT)')
ax.set_title('Lorenz 7-node network: VPT vs λ')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# ============================================================
# 5. 汇总
# ============================================================
print('\n' + '=' * 70)
print('汇总（最优 λ 下）')
print('=' * 70)
print(f'{"方法":<12s} | {"最优 λ":>10s} | {"VPT":>6s} | {"特征数":>8s} | '
      f'{"VPT/特征":>10s}')
print('-' * 70)
for name, res in results.items():
    eff = res['best_vpt'] / res['n_feat']
    print(f'{name:<12s} | {res["best_lam"]:>10.2e} | '
          f'{int(res["best_vpt"]):>6d} | {res["n_feat"]:>8d} | '
          f'{eff:>10.4f}')

# ============================================================
# 6. 用最优 λ 预测并画图（p=2 vs p=3）
# ============================================================
print('\n重新用最优 λ 预测...')
preds = {}
for poly_order in [2, 3]:
    best_lam = results[f'p={poly_order}']['best_lam']
    yc, n_feat = build_and_train(xh1gai, cc, poly_order, best_lam,
                                   start, qstart, n)
    yct = f变量变形(yc)
    vpt_val = vpt(yct[100:], xh2, 0.5)
    preds[f'p={poly_order}'] = {
        'yct': yct * suo,
        'vpt': vpt_val,
        'lam': best_lam,
    }
    print(f'  p={poly_order}, λ*={best_lam:.2e}, VPT={vpt_val:.4f}')

# 预测对比图
yx2 = xx2
l1_plot = 600

fig, axs = plt.subplots(4, 1, figsize=(9, 7), dpi=150, sharex=True)
for k, ax in enumerate(axs):
    ax.plot(yx2[:l1_plot, k], 'r', lw=1.0, label='actual')
    ax.plot(preds['p=2']['yct'][100:100+l1_plot, k], 'b--', lw=0.9,
            label=f"p=2 (λ*={preds['p=2']['lam']:.1e}, VPT={preds['p=2']['vpt']:.0f})")
    ax.plot(preds['p=3']['yct'][100:100+l1_plot, k], 'g:', lw=0.9,
            label=f"p=3 (λ*={preds['p=3']['lam']:.1e}, VPT={preds['p=3']['vpt']:.0f})")
    ax.set_ylabel(f'$x_{k+1}$')
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)

axs[0].legend(fontsize=7, loc='upper right')
axs[-1].set_xlabel('t')
plt.suptitle('SCNG-RC p=2 vs p=3 on Lorenz 7-node network', fontsize=10)
plt.tight_layout()
plt.show()

# ============================================================
# 7. 误差热图
# ============================================================
wucha2 = np.abs(preds['p=2']['yct'][100:1100, :] - xx2[:1000, :])
wucha3 = np.abs(preds['p=3']['yct'][100:1100, :] - xx2[:1000, :])

fig = plt.figure(dpi=150, figsize=(9, 4))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.03],
                       hspace=0.2, wspace=0.05)

axs = [fig.add_subplot(gs[0, 0]),
       fig.add_subplot(gs[1, 0])]
cax = fig.add_subplot(gs[:, 1])   # colorbar 专用列

for ax, w, name in zip(axs,
                        [wucha2, wucha3],
                        [f"p=2 (VPT={preds['p=2']['vpt']:.0f})",
                         f"p=3 (VPT={preds['p=3']['vpt']:.0f})"]):
    im = ax.imshow(w[:500].T, cmap='viridis_r', vmax=30,
                   origin='lower', aspect='auto',
                   extent=[0.5, 500.5, 0.5, w.shape[1]+0.5])
    ax.set_ylabel('dim')
    ax.set_title(name, fontsize=8, loc='left')
    ax.tick_params(labelsize=7)

axs[-1].set_xlabel('t')
fig.colorbar(im, cax=cax)

plt.show()

np.savez('data/results_lorenz_p2p3.npz',
         wucha2=wucha2, wucha3=wucha3,
         vpt_p2=preds['p=2']['vpt'],
         vpt_p3=preds['p=3']['vpt'])
print('结果已保存到 data/results_lorenz_p2p3.npz')